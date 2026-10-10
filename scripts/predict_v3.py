"""
predict_v3.py

Bet recommender that builds on predict_v2's prediction pipeline but
replaces the bet trigger logic with an entropy + ml two-rule selection.

What this version does:
    - Two rules OR'd together. A horse fires if EITHER rule passes:
        Chalk rule:    (LOG OR RDG OR RAW trigger)
                       AND ml ≤ ML_MAX_FOR_BET (8.0)
                       AND race_entropy_norm ≥ q60 (top 40% flattest races)
        Longshot rule: (LOG OR RDG OR RAW trigger)
                       AND ml ≥ ML_MIN_LONGSHOT (12.0)
                       AND race_entropy_norm ≤ q40 (bottom 40% peakest races)
      Mid-ml zone (8 < ml < 12) is intentionally excluded — diagnostic
      showed it was the worst ROI bucket regardless of trigger.
    - Race entropy is the normalized Shannon entropy of ensemble_prob
      across the field (H/log(N), bounded in [0, 1]). High entropy =
      flat race, low = peaked.
    - The q40 / q60 thresholds are dynamic — computed by backtest_kfold
      from the loaded sample and written to a JSON file. predict_v3
      reads that file at run time.
    - RAW is now part of the trigger set (uses raw_value_model.json,
      same per-row calibration as predict_v2: bet if raw_value ≥ μ + 1σ).
    - LOG-triggered bets are W-only. Otherwise predict_v2's standard
      W/WP/WPS by ml rule applies. In practice:
          chalk rule (ml ≤ 8) → mostly W or WP
          longshot rule (ml ≥ 12) → WPS

Why two rules:
    - Chalk rule: in flat races, market prices spread out and a value
      edge has more headroom to overcome 17% takeout. Diagnostic shows
      this cell consistently profitable on the in-sample data.
    - Longshot rule: in peaked races (one obvious chalk), the rest of
      the field is overpriced. A longshot trigger here might be a
      genuine overlay where the public has overcommitted to the favorite.
      Diagnostic doesn't yet show this rule profitable on the current
      (favorite-heavy) sample, but the hypothesis is that it will work
      better in more longshot-heavy regimes. Treat as a prospective
      experiment, not validated edge.

Honest caveats:
    - Both q40 and q60 thresholds are computed from the same K-fold data
      the whole pipeline was tuned on. ROI estimates are upper bounds.
    - The longshot rule is prior-driven — recent in-sample ROI is mildly
      negative, the bet is that this reflects a chalk-heavy regime that
      will revert. Track prospectively before scaling stakes.
    - Thresholds drift as the backtest ingests new days. That's by
      design — keeps each rule's pass-rate stable. But it does mean
      "today's gate" depends on "yesterday's backtest," so keep them
      in sync (run backtest_kfold before predict_v3 after data refresh).

Usage:
    python predict_v3.py --date 2026-04-26 --track churchill-downs
    python predict_v3.py --date 2026-04-26 --track keeneland --num-sims 200000

Make sure backtest_kfold.py has run recently enough that
v3_thresholds.json reflects current data — this script reads that
file at startup and fails loudly if it's missing or schema-mismatched.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Reuse predict_v2's infrastructure wholesale. predict_v2 is a clean
# module — only top-level side effect is load_dotenv(), so importing
# is safe.
import predict_v2 as p2


# =============================================================================
# v3 CONSTANTS
# =============================================================================

# Single ml cap applied to both LOG and RDG triggers. Based on the
# 3×5 entropy×ml diagnostic finding that ml > 8 (mid/longshot) was
# the worst odds bucket regardless of trigger; capping at 8 excludes
# that zone. Round number, not a tuned value.
# ML bands for the two-rule gate.
#
# Chalk rule:    ml ≤ ML_MAX_FOR_BET  (favorites and short-mid odds)
# Longshot rule: ml ≥ ML_MIN_LONGSHOT (12/1 and up)
# Mid-band (ML_MAX_FOR_BET < ml < ML_MIN_LONGSHOT) is excluded from
# both rules. The 3×5 entropy×ml diagnostic showed mid-ml (~8-12) was
# the worst ROI bucket regardless of trigger or entropy band.
ML_MAX_FOR_BET   = 8.0   # chalk-rule upper bound
ML_MIN_LONGSHOT  = 12.0  # longshot-rule lower bound

# LOG-triggered bets are W-only (per Blake's choice). All other
# triggered horses use predict_v2's standard W/WP/WPS-by-ml rule.
# In the chalk rule, ml ≤ 8 < ML_WP_MAX (=9), so WPS rarely applies.
# In the longshot rule, ml ≥ 12 > ML_WP_MAX, so non-LOG bets are WPS.
LOG_TRIGGER_FORCE_W = True

# Entropy axis used by the gate. The DEFAULT is normalized cv_prob entropy
# ("how confident is our own model in this race"). The --use-ml-entropy
# flag swaps to ML-implied entropy (1/ml_decimal), which is independent of
# our model and the expert pipeline. ML entropy backtested WORSE — it's
# offered for diagnostics and prospective experiments, not as a default.
GATE_ENTROPY_KEY_DEFAULT = 'race_entropy_norm'
GATE_ENTROPY_KEY_ML      = 'race_entropy_ml'

# Path the backtest writes thresholds to. predict_v3 reads from here.
# Uses the DATA_PATH env var (same convention predict_v2 uses).
_DATA_PATH_ENV = os.environ.get('DATA_PATH')
V3_THRESHOLDS_PATH = (
    Path(_DATA_PATH_ENV) / 'predictions' / 'v3_thresholds.json'
    if _DATA_PATH_ENV else None
)


def _load_v3_thresholds(gate_key=GATE_ENTROPY_KEY_DEFAULT):
    """Read v3_thresholds.json, return (chalk_threshold, longshot_threshold,
    payload).

    The two-rule gate needs:
      - q60 of the gate axis  → chalk-rule threshold (entropy ≥ this)
      - q40 of the gate axis  → longshot-rule threshold (entropy ≤ this)

    `gate_key` selects which entropy axis to gate on:
      - 'race_entropy_norm' (default) — entropy of our ensemble_prob
      - 'race_entropy_ml'             — entropy of 1/ml_decimal (morning line)

    backtest_kfold writes q40/q60 for ALL three axes to the JSON file
    (race_entropy_norm, race_entropy_ml, race_entropy_raw), so any of
    them can be selected at run time without re-running the backtest.

    Fails loudly if:
      - DATA_PATH env var isn't set (path can't be constructed)
      - the file doesn't exist (backtest hasn't been run)
      - the file is malformed JSON
      - the requested axis's q60 or q40 key is missing (likely an old
        schema_version=1 file from before the multi-axis writeout)

    Silent fallbacks would let predict_v3 silently use stale or missing
    thresholds. Better to fail at startup than to bet with a
    misconfigured gate.

    Returns: (chalk_threshold, longshot_threshold, payload_dict).
    """
    if V3_THRESHOLDS_PATH is None:
        raise RuntimeError(
            "DATA_PATH env var is not set. predict_v3 needs it to find "
            "v3_thresholds.json."
        )
    if not V3_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"v3 thresholds file not found at {V3_THRESHOLDS_PATH}. "
            f"Run backtest_kfold.py first — it computes the q40/q60 "
            f"thresholds and writes the JSON file that predict_v3 reads."
        )
    try:
        with open(V3_THRESHOLDS_PATH) as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"v3 thresholds file at {V3_THRESHOLDS_PATH} is not valid JSON: {e}"
        ) from e

    thresholds = payload.get('thresholds', {})
    chalk_key    = f'{gate_key}_q60'
    longshot_key = f'{gate_key}_q40'
    missing = [k for k in (chalk_key, longshot_key) if k not in thresholds]
    if missing:
        schema = payload.get('schema_version', 'unknown')
        raise RuntimeError(
            f"v3 thresholds file at {V3_THRESHOLDS_PATH} is missing key(s) "
            f"for gate axis '{gate_key}': {missing}. "
            f"Found keys: {sorted(thresholds.keys())}. "
            f"Schema version: {schema}. Re-run backtest_kfold.py to "
            f"regenerate the file with all three axes."
        )
    return float(thresholds[chalk_key]), float(thresholds[longshot_key]), payload


# =============================================================================
# RACE-LEVEL ENTROPY
# =============================================================================

def compute_race_entropy(probs):
    """
    Normalized Shannon entropy: H/log(N), bounded in [0, 1].

    Higher = flatter race (model less peaked). Lower = peaked race
    (model has a clear pick). Same function as in backtest_kfold —
    kept identical so live and backtest agree on what they're
    computing.

    Inputs are re-normalized defensively in case probs don't sum to
    1 (scratches, rounding). Negatives floored to 0; NaN dropped.
    Returns NaN for degenerate races (N < 2 or all zero).
    """
    arr = np.asarray(probs, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = np.clip(arr, 0.0, None)
    if len(arr) < 2:
        return np.nan
    total = arr.sum()
    if total <= 0:
        return np.nan
    p = arr / total
    p = p[p > 0]
    if len(p) < 2:
        return np.nan
    h = -np.sum(p * np.log(p))
    return h / np.log(len(p))


def add_race_entropy_norm(df):
    """Add a 'race_entropy_norm' column with the normalized entropy
    of ensemble_prob across the race, broadcast to every row."""
    if 'ensemble_prob' not in df.columns or not df['ensemble_prob'].notna().any():
        df['race_entropy_norm'] = np.nan
        return df

    by_race = df.groupby('race')['ensemble_prob'].apply(compute_race_entropy)
    by_race.name = 'race_entropy_norm'
    df = df.merge(by_race.reset_index(), on='race', how='left',
                  suffixes=('', '_drop'))
    if 'race_entropy_norm_drop' in df.columns:
        df = df.drop(columns=['race_entropy_norm_drop'])
    return df


def add_race_entropy_ml(df):
    """Add a 'race_entropy_ml' column with the normalized entropy of
    1/ml_decimal across the race, broadcast to every row.

    ML-implied probability has overround (track's margin baked in) so it
    doesn't sum to 1. compute_race_entropy re-normalizes defensively, so
    this measures the SHAPE of the morning line — ignoring the constant
    overround multiplier — across the field.

    Compared to race_entropy_norm:
      - Independent of our model and the expert pipeline (no drift risk)
      - Reflects "what the track's odds setter thought the race would
        look like before betting started"
      - Backtested WORSE than race_entropy_norm in current sample —
        offered for diagnostics and prospective experiments.
    """
    if 'ml_decimal' not in df.columns or not df['ml_decimal'].notna().any():
        df['race_entropy_ml'] = np.nan
        return df

    work = df.copy()
    work['_ml_prob'] = np.where(
        work['ml_decimal'] > 0,
        1.0 / work['ml_decimal'],
        np.nan,
    )
    by_race = work.groupby('race')['_ml_prob'].apply(compute_race_entropy)
    by_race.name = 'race_entropy_ml'
    df = df.merge(by_race.reset_index(), on='race', how='left',
                  suffixes=('', '_drop'))
    if 'race_entropy_ml_drop' in df.columns:
        df = df.drop(columns=['race_entropy_ml_drop'])
    return df


# =============================================================================
# v3 BET LOGIC
# =============================================================================

def generate_bets_v3(df, takeout=p2.WIN_TAKEOUT,
                     entropy_q60=None, entropy_q40=None):
    """
    v3 bet recommender. Two rules, OR'd.

    Trigger condition (any model):
        log_bet (ev_per_2 ≥ LOGISTIC_EDGE_THRESHOLD) OR
        rdg_bet (ridge_value ≥ ridge_break_even × RIDGE_EDGE_MULTIPLIER) OR
        raw_bet (raw_value ≥ raw_cv_mean + RAW_EDGE_K × raw_cv_std)

    Chalk rule:
        race_entropy_norm ≥ entropy_q60  (top 40% flattest races)
        AND ml_decimal ≤ ML_MAX_FOR_BET  (favorites and short-mid odds)

    Longshot rule:
        race_entropy_norm ≤ entropy_q40  (bottom 40% peakest races)
        AND ml_decimal ≥ ML_MIN_LONGSHOT (12/1 and up)

    A horse fires if any trigger is true AND it passes either rule.
    Mid-ml zone (ML_MAX_FOR_BET < ml < ML_MIN_LONGSHOT) is excluded
    from both rules.

    Bet type:
        LOG fired             → W only (forced)
        else                  → predict_v2's standard W/WP/WPS by ml
        Chalk-rule bets land in W or WP (ml ≤ 8).
        Longshot-rule non-LOG bets land in WPS (ml ≥ 12 > ML_WP_MAX).

    triggered_by reflects which trigger(s) fired (LOG, RDG, RAW, or
    combos). rule reflects which rule passed (chalk or longshot).
    Both go into the bet record.

    Args:
        df: per-horse df from predict_v2's pipeline. Must include
            ensemble_prob, ev_per_2, ridge_value, raw_value, ml_decimal.
        takeout: win-pool takeout. Used to compute ridge break-even.
        entropy_q60: threshold for the chalk rule (entropy ≥ this).
        entropy_q40: threshold for the longshot rule (entropy ≤ this).
            Both must come from v3_thresholds.json. None values
            disable that rule (gate falls open) — useful for debugging
            but should not happen in production.

    Returns (bet_df, race_summaries) — same shape as predict_v2's
    generate_bets, so print_race_report works without changes.
    """
    bets = []
    race_summaries = {}

    # Make sure race_entropy_norm is present
    if 'race_entropy_norm' not in df.columns:
        df = add_race_entropy_norm(df)

    has_logistic = 'ev_per_2' in df.columns and df['ev_per_2'].notna().any()
    has_ridge    = 'ridge_value' in df.columns and df['ridge_value'].notna().any()
    has_raw      = 'raw_value' in df.columns and df['raw_value'].notna().any()
    takeout_gross = 1.0 / max(1e-6, 1.0 - takeout)

    # RAW edge calibration. predict_v2 stores cv_pred_mean/std on
    # df.attrs after apply_raw_value_model runs; fall back to
    # in-sample stats if missing (matches predict_v2's defensive path).
    raw_cv_mean = df.attrs.get('raw_cv_mean') if has_raw else None
    raw_cv_std  = df.attrs.get('raw_cv_std')  if has_raw else None
    if has_raw and (raw_cv_mean is None or raw_cv_std is None):
        raw_cv_mean = float(df['raw_value'].mean())
        raw_cv_std  = float(df['raw_value'].std())
    raw_edge_threshold = (
        raw_cv_mean + p2.RAW_EDGE_K * raw_cv_std
        if has_raw and raw_cv_mean is not None and raw_cv_std is not None
        else None
    )

    # Defensive: if either threshold wasn't provided, use a value that
    # makes that rule unable to pass (rather than fall fully open). For
    # chalk rule (entropy ≥ q60), use +inf so nothing clears it. For
    # longshot rule (entropy ≤ q40), use -inf so nothing clears it.
    # This is conservative — better to bet nothing than bet too much
    # in an unconfigured state.
    chalk_threshold    = entropy_q60 if entropy_q60 is not None else np.inf
    longshot_threshold = entropy_q40 if entropy_q40 is not None else -np.inf

    for race_num, race_df in df.groupby('race'):
        n_horses = len(race_df)
        race_entropy = (
            race_df['race_entropy_norm'].iloc[0]
            if 'race_entropy_norm' in race_df.columns
            else np.nan
        )

        # Per-race ridge break-even (same as v2)
        if has_ridge and n_horses > 0:
            ridge_raw_fair = p2.RIDGE_RACE_TOTAL / n_horses
            ridge_break_even = ridge_raw_fair * takeout_gross
            ridge_edge = ridge_break_even * p2.RIDGE_EDGE_MULTIPLIER
        else:
            ridge_break_even = ridge_edge = None

        # Race-level rule eligibility (entropy band only).
        # NaN entropy fails both rules — defensive.
        race_chalk_ok    = pd.notna(race_entropy) and race_entropy >= chalk_threshold
        race_longshot_ok = pd.notna(race_entropy) and race_entropy <= longshot_threshold

        # Per-horse filter evaluation
        race_bets = []
        for _, row in race_df.iterrows():
            ev_per_2    = row.get('ev_per_2', np.nan)
            ridge_value = row.get('ridge_value', np.nan)
            raw_value   = row.get('raw_value', np.nan)
            ml_dec      = row.get('ml_decimal', np.nan)

            # Per-horse trigger checks (without the gate).
            log_trigger = (
                has_logistic
                and pd.notna(ev_per_2)
                and ev_per_2 >= p2.LOGISTIC_EDGE_THRESHOLD
            )
            rdg_trigger = (
                has_ridge
                and pd.notna(ridge_value)
                and ridge_edge is not None
                and ridge_value >= ridge_edge
            )
            raw_trigger = (
                has_raw
                and pd.notna(raw_value)
                and raw_edge_threshold is not None
                and raw_value >= raw_edge_threshold
            )

            any_trigger = log_trigger or rdg_trigger or raw_trigger
            if not any_trigger:
                continue

            # ml-band check for each rule
            ml_chalk_ok    = pd.notna(ml_dec) and ml_dec <= ML_MAX_FOR_BET
            ml_longshot_ok = pd.notna(ml_dec) and ml_dec >= ML_MIN_LONGSHOT

            chalk_rule_passes    = race_chalk_ok    and ml_chalk_ok
            longshot_rule_passes = race_longshot_ok and ml_longshot_ok

            if not (chalk_rule_passes or longshot_rule_passes):
                continue

            triggers = []
            if log_trigger: triggers.append('LOG')
            if rdg_trigger: triggers.append('RDG')
            if raw_trigger: triggers.append('RAW')
            triggered_by = '+'.join(triggers)

            # Rule label. Both shouldn't fire simultaneously since ml
            # bands are disjoint, but if they somehow do, prefer chalk
            # (lower-variance, more validated).
            if chalk_rule_passes:
                rule = 'chalk'
            else:
                rule = 'longshot'

            # Bet type. LOG → W only. Else predict_v2 standard ml-based.
            # In chalk rule (ml ≤ 8 < ML_WP_MAX=9): WPS unreachable.
            # In longshot rule (ml ≥ 12 > ML_WP_MAX): WPS for non-LOG.
            if log_trigger and LOG_TRIGGER_FORCE_W:
                bet_type = 'W'
            elif pd.notna(ml_dec) and ml_dec <= p2.ML_WIN_ONLY_MAX:
                bet_type = 'W'
            elif pd.notna(ml_dec) and ml_dec <= p2.ML_WP_MAX:
                bet_type = 'WP'
            else:
                bet_type = 'WPS'

            bet_record = {
                'race': race_num,
                'horse_number': int(row['horse_number']),
                'bet_type': bet_type,
                'rule': rule,
                'ensemble_prob': round(row.get('ensemble_prob', np.nan), 4),
                'ml_prob': round(row.get('ml_win_prob_norm', np.nan), 4),
                'edge': round(row.get('edge', np.nan), 4),
                'ev_per_2': round(ev_per_2, 2) if pd.notna(ev_per_2) else np.nan,
                'ev_roi': round(row.get('ev_roi', np.nan), 3),
                'kelly': round(row.get('half_kelly', 0), 4),
                'ml_odds': row.get('ml_odds_raw', ''),
                'triggered_by': triggered_by,
                'race_entropy_norm': (
                    round(race_entropy, 3) if pd.notna(race_entropy) else np.nan
                ),
            }

            if has_ridge:
                bet_record['ridge_value'] = (
                    round(ridge_value, 2) if pd.notna(ridge_value) else np.nan
                )
                bet_record['ridge_edge'] = (
                    round(ridge_edge, 2) if ridge_edge is not None else np.nan
                )

            if has_raw and pd.notna(raw_value):
                bet_record['raw_value'] = round(float(raw_value), 2)
                if raw_edge_threshold is not None:
                    bet_record['raw_edge'] = round(float(raw_edge_threshold), 2)

            race_bets.append(bet_record)

        bets.extend(race_bets)

        race_summaries[race_num] = {
            'action': 'BET' if race_bets else 'PASS',
            'horses': [b['horse_number'] for b in race_bets],
            'flagged': [],  # v3 doesn't have a separate flagged tier
            'ridge_break_even': (
                round(ridge_break_even, 2) if ridge_break_even else None
            ),
            'race_entropy_norm': (
                round(race_entropy, 3) if pd.notna(race_entropy) else None
            ),
            'race_chalk_eligible':    bool(race_chalk_ok),
            'race_longshot_eligible': bool(race_longshot_ok),
        }

    bet_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    return bet_df, race_summaries


# =============================================================================
# EXACTA BOX RECOMMENDATIONS
# =============================================================================

def compute_exacta_boxes(bet_df):
    """
    Identify races where v3 emits 2 or 3 RDG-only triggered bets (no
    LOG fire), under the CHALK rule only. Build exacta box
    recommendations for those races.

    Backtest finding (in-sample): on multi-trigger RDG races within
    the chalk rule, exacta boxes captured meaningful edge. Caveat:
    in-sample on the same data v3 thresholds were tuned on — real
    prospective ROI will be lower. The longshot-rule version of
    this hasn't been validated, and boxing horses paying ml ≥ 12
    each would chew through bankroll fast — so we restrict to chalk.

    Box rules:
      2 RDG-only chalk-rule horses → $4 exacta box ($2 × 2 combos)
      3 RDG-only chalk-rule horses → $12 exacta box ($2 × 6 combos)
      4+ → no recommendation (sample too thin to justify cost)

    Returns: list of dicts, each:
        {race, horses (sorted ints), n, cost, race_entropy_norm}
    Empty list if no races qualify.
    """
    if bet_df is None or bet_df.empty:
        return []
    if 'triggered_by' not in bet_df.columns:
        return []

    boxes = []
    for race, race_df in bet_df.groupby('race'):
        # Restrict to chalk-rule bets only. The longshot rule's RDG
        # fires are operationally distinct (much higher variance,
        # untested for boxing).
        if 'rule' in race_df.columns:
            race_df = race_df[race_df['rule'] == 'chalk']
            if race_df.empty:
                continue

        # Any LOG fire in this race disqualifies the box pool
        has_log = race_df['triggered_by'].str.contains('LOG').any()
        if has_log:
            continue

        rdg_only = race_df[race_df['triggered_by'] == 'RDG']
        n = len(rdg_only)
        if n not in (2, 3):
            continue

        horses = sorted(int(h) for h in rdg_only['horse_number'])
        cost = 4 if n == 2 else 12
        race_entropy = (
            float(rdg_only['race_entropy_norm'].iloc[0])
            if 'race_entropy_norm' in rdg_only.columns
               and pd.notna(rdg_only['race_entropy_norm'].iloc[0])
            else None
        )

        boxes.append({
            'race': int(race),
            'horses': horses,
            'n': n,
            'cost': cost,
            'race_entropy_norm': race_entropy,
        })

    return boxes


def print_exacta_box_recommendations(boxes):
    """Pretty-print the box recommendations as a deployable list."""
    if not boxes:
        return

    print()
    print("=" * 80)
    print("EXACTA BOX RECOMMENDATIONS (multi-trigger RDG-only races)")
    print("=" * 80)
    print("  These supplement the per-horse W/WP/WPS bets above — they don't")
    print("  replace them. Backtest finding: when v3 RDG fires on 2-3 horses")
    print("  in a race (no LOG fire), boxing those horses captured meaningful")
    print("  edge in-sample. Real prospective ROI will be lower.")
    print()
    total_cost = sum(b['cost'] for b in boxes)
    print(f"  Total: {len(boxes)} box(es), ${total_cost} cost")
    print()
    print(f"    {'race':>4} {'n':>2} {'cost':>5}  {'horses':<14} "
          f"{'race_entropy':>12}")
    print(f"    {'-'*4} {'-'*2} {'-'*5}  {'-'*14} {'-'*12}")
    for b in boxes:
        horses_str = '/'.join(str(h) for h in b['horses'])
        ent_str = (
            f"{b['race_entropy_norm']:.3f}"
            if b['race_entropy_norm'] is not None else '—'
        )
        print(f"    R{b['race']:<3} {b['n']:>2} ${b['cost']:>3}  "
              f"{horses_str:<14} {ent_str:>12}")


# =============================================================================
# RACE BANNER ANNOTATION
# =============================================================================

def race_banner_extra(race_entropy, q60, q40):
    """Build the per-race annotation appended to the banner by predict_v2.

    Surfaces:
      - the entropy value
      - which rule(s) the race is eligible for (chalk / longshot / neither)

    "neither" means no v3 bet can fire on this race regardless of model
    triggers — useful to know at a glance when scanning the per-race tables.
    """
    if pd.isna(race_entropy):
        return "entropy=—"
    if race_entropy >= q60:
        tag = "CHALK-zone"
    elif race_entropy <= q40:
        tag = "LONGSHOT-zone"
    else:
        tag = "no-zone"
    return f"entropy={race_entropy:.3f} [{tag}]"


# =============================================================================
# HUMAN-POOL REPORT (--human-picks)
# =============================================================================

def print_human_pool_report(df, date_str, track, skip_tipmeerkat=True):
    """Print a per-race table of human_pool / human_prob per horse.

    The "human pool" is the regression model's view of expert consensus:
    each expert's per-horse logit_delta gets scaled by their Elo
    multiplier and summed into a per-horse pool value. The per-race
    softmax-like normalization of those pools is human_prob.

    Calculation lives in predict_v2._ensure_regression_features. We
    invoke it here on a copy of df (so we don't mutate the user's
    bet/prediction df), then format the result.

    Note: the regression's Elo weighting (linear, threshold at 1350) is
    DIFFERENT from the ensemble adjustment's logit weighting (rating-1500
    scaled by 1000). Don't expect human_pool ranks to match the
    ensemble's expert adjustments line-for-line.
    """
    if df is None or df.empty:
        print("\n[--human-picks] No data available — skipping human-pool report.")
        return

    # Compute on a copy so we don't pollute the saved predictions df.
    work = df.copy()
    work = p2._ensure_regression_features(
        work, date_str, track, skip_tipmeerkat=skip_tipmeerkat
    )

    if 'human_pool' not in work.columns or 'human_prob' not in work.columns:
        print("\n[--human-picks] human_pool/human_prob not computed — "
              "no expert picks found for this date/track?")
        return

    print()
    print("=" * 80)
    print("HUMAN POOL  (per-race expert-consensus regression feature)")
    print("=" * 80)
    print("  Per expert: pool += (logit_delta / 0.7) * 2.0 * elo_mult")
    print("              elo_mult = max(0.1, (rating - 1350) / 150)")
    print("  Per race:   human_prob = softmax-shifted normalization of pool.")
    if skip_tipmeerkat:
        print("  (tipmeerkat experts excluded for this report.)")
    print()

    has_name = 'horse_name' in work.columns
    for race_num in sorted(work['race'].unique()):
        race_df = work[work['race'] == race_num].sort_values(
            'human_pool', ascending=False
        )
        if race_df['human_pool'].abs().sum() == 0:
            print(f"  Race {race_num}: no expert picks for this race")
            print()
            continue

        print(f"  Race {race_num}  (N={len(race_df)})")
        if has_name:
            print(f"    {'#':>3}  {'horse':<28} {'pool':>8}  {'prob':>7}")
            print(f"    {'-'*3}  {'-'*28} {'-'*8}  {'-'*7}")
        else:
            print(f"    {'#':>3}  {'pool':>8}  {'prob':>7}")
            print(f"    {'-'*3}  {'-'*8}  {'-'*7}")
        for _, row in race_df.iterrows():
            hn = int(row['horse_number'])
            pool = float(row['human_pool'])
            prob = float(row['human_prob'])
            if has_name:
                name = str(row.get('horse_name', ''))[:28]
                print(f"    {hn:>3}  {name:<28} {pool:>+8.2f}  {prob:>6.1%}")
            else:
                print(f"    {hn:>3}  {pool:>+8.2f}  {prob:>6.1%}")
        print()


# =============================================================================
# v3 ORCHESTRATOR
# =============================================================================
#
# Approach: rather than duplicate predict_v2.run_predictions (which is
# ~130 lines of subtle data-loading and merging logic that we'd risk
# drifting from), we monkey-patch predict_v2.generate_bets to call our
# v3 logic, then call predict_v2.run_predictions normally. After v3
# returns, we restore the original generate_bets so other importers
# aren't affected.
#
# The catch is that generate_bets_v3 doesn't take all of v2's keyword
# args (raw_edge_k, raw_min_k, etc.). We provide a shim that accepts
# but ignores the v2 kwargs while computing race entropy and
# dispatching to our v3 logic.

# Module-level capture of the most recent v3 bet_df. Set by the shim,
# read by run_predictions_v3 after p2.run_predictions returns. We can't
# get the bet_df out via a return value because predict_v2 owns the
# orchestration loop and doesn't pass through what generate_bets returned.
_LAST_BET_DF = None

# Module-level capture of the full per-horse df at bet-generation time.
# Set by the shim, read by run_predictions_v3 for the optional --human-picks
# report (which needs human_pool/human_prob columns computed off the live df).
_LAST_FULL_DF = None

# Per-race banner annotations (entropy + zone tag). Populated by the shim
# during generate_bets; read by predict_v2's per-race print loop AFTER
# generate_bets returns. We use a module-level dict the shim mutates in
# place so the same object reference v3 passes to p2.run_predictions
# stays in sync.
_RACE_BANNER_EXTRAS = {}

# Module-level threshold cache — loaded once per run by run_predictions_v3,
# read by the shim. Avoids re-reading the JSON file on every race.
_ENTROPY_Q60 = None  # chalk-rule threshold (entropy ≥ this)
_ENTROPY_Q40 = None  # longshot-rule threshold (entropy ≤ this)

# Module-level gate-axis selection. Default is the cv-norm entropy that
# the v3 backtest validated. The --use-ml-entropy flag swaps it to ML
# entropy (1/ml_decimal) for diagnostic / prospective-experiment use.
# The shim writes the chosen axis's per-race entropy into the
# 'race_entropy_norm' column of df so the rest of the pipeline (gate
# check, banner display, bet records) reads from one canonical column
# regardless of which axis sourced it.
_GATE_AXIS = GATE_ENTROPY_KEY_DEFAULT  # 'race_entropy_norm' or 'race_entropy_ml'


def _generate_bets_shim(df, **kwargs):
    """Adapter that lets our v3 logic plug into predict_v2.run_predictions.

    Accepts (and ignores) all of v2's keyword args. Computes race
    entropy on the input df, then calls generate_bets_v3 with the
    two-rule thresholds loaded earlier. Also captures the full df
    so post-run reporters (e.g. --human-picks) can work off it, and
    populates _RACE_BANNER_EXTRAS so v2's per-race print loop can show
    entropy + zone tags in each race banner.

    Honors _GATE_AXIS — when set to 'race_entropy_ml', the ML-derived
    entropy is computed and written into df['race_entropy_norm'] so the
    downstream gate check (which reads that one column) operates on it
    transparently. The original 'race_entropy_norm' values, if any,
    are preserved alongside under 'race_entropy_norm_orig' for
    debugging — no other code reads it.
    """
    global _LAST_BET_DF, _LAST_FULL_DF

    # Compute the chosen gate axis's per-race entropy and write it into
    # the canonical 'race_entropy_norm' column. predict_v2 has just
    # finished computing ensemble_prob when it calls generate_bets, so
    # both axes' source columns should be available.
    if _GATE_AXIS == GATE_ENTROPY_KEY_ML:
        # Compute ML entropy. Stash any pre-existing 'race_entropy_norm'
        # under '_orig' (rare — the column shouldn't exist yet at this
        # point in v2's pipeline, but be defensive).
        ml_df = add_race_entropy_ml(df.copy())
        if 'race_entropy_norm' in df.columns:
            df['race_entropy_norm_orig'] = df['race_entropy_norm']
        df['race_entropy_norm'] = ml_df['race_entropy_ml']
    else:
        if 'race_entropy_norm' not in df.columns:
            df_with_ent = add_race_entropy_norm(df.copy())
            df['race_entropy_norm'] = df_with_ent['race_entropy_norm']

    takeout = kwargs.get('takeout', p2.WIN_TAKEOUT)
    bet_df, race_summaries = generate_bets_v3(
        df, takeout=takeout,
        entropy_q60=_ENTROPY_Q60, entropy_q40=_ENTROPY_Q40,
    )

    # Tag every bet record with which gate axis was used. This goes into
    # the saved bet_log.csv so post-hoc analysis can split results by
    # gate axis without ambiguity.
    if not bet_df.empty:
        bet_df['gate_axis'] = _GATE_AXIS

    _LAST_BET_DF = bet_df
    _LAST_FULL_DF = df

    # Build per-race banner extras for v2's per-race print loop.
    # In-place mutation so v3's reference stays current.
    _RACE_BANNER_EXTRAS.clear()
    for race_num, race_df in df.groupby('race'):
        race_entropy = (
            race_df['race_entropy_norm'].iloc[0]
            if 'race_entropy_norm' in race_df.columns else float('nan')
        )
        _RACE_BANNER_EXTRAS[int(race_num)] = race_banner_extra(
            race_entropy, _ENTROPY_Q60, _ENTROPY_Q40
        )
    return bet_df, race_summaries


def run_predictions_v3(date_str, track, num_sims=100000, use_regression=False,
                       skip_tipmeerkat=True, show_human_picks=False,
                       gate_axis=GATE_ENTROPY_KEY_DEFAULT):
    """
    Run the v3 prediction pipeline. Internally this is just
    predict_v2.run_predictions with generate_bets temporarily replaced
    by our v3 shim, which guarantees we use predict_v2's data-loading,
    scratch-handling, ML normalization, and ensemble logic exactly.

    Threshold loading happens up front: if v3_thresholds.json is missing
    or malformed, we fail loudly before doing any prediction work — no
    point spending compute on a run that can't apply the gate correctly.

    v3 trigger rules are printed at the top of the run (right after the
    threshold load) so the per-race output is read with them in mind.
    predict_v2's own trigger explanation is suppressed via the
    suppress_trigger_explanation kwarg (the explanation describes v2
    rules that don't apply here and would just contradict v3's).

    Args:
      skip_tipmeerkat: forwarded to predict_v2.run_predictions; excludes
        tm_* experts from both the ensemble adjustment and the regression
        human_pool. Their Elo ratings still update via expert_elo.py.
        Defaults to True (skip) because tipmeerkat data is currently
        bugged. Pass False to opt back in.
      show_human_picks: if True, after the run print a per-race table of
        human_pool / human_prob (the regression model's expert-consensus
        feature). Useful for sanity-checking how the expert picks are
        being aggregated.
      gate_axis: which entropy axis to gate on. Default
        'race_entropy_norm' (entropy of our ensemble_prob, what the v3
        backtest validated). 'race_entropy_ml' uses ML-implied entropy
        (1/ml_decimal) instead — backtested WORSE but useful for
        diagnostics or as a prospective experiment when the cv-norm
        distribution is drifting from calibration.
    """
    global _LAST_BET_DF, _LAST_FULL_DF, _ENTROPY_Q60, _ENTROPY_Q40, _GATE_AXIS
    _LAST_BET_DF = None
    _LAST_FULL_DF = None
    _GATE_AXIS = gate_axis

    # Fail-loudly threshold load. Done before any heavy lifting so a
    # missing file doesn't waste a long prediction run.
    q60, q40, payload = _load_v3_thresholds(gate_key=gate_axis)
    _ENTROPY_Q60 = q60
    _ENTROPY_Q40 = q40
    print(f"[v3] Loaded entropy thresholds from {V3_THRESHOLDS_PATH}")
    print(f"     gate axis = {gate_axis}")
    print(f"     chalk-rule q60 = {q60:.4f}  (entropy ≥ this)")
    print(f"     longshot-rule q40 = {q40:.4f}  (entropy ≤ this)")
    print(f"     computed_at = {payload.get('computed_at', '?')}, "
          f"n_races = {payload.get('n_races_in_sample', '?')}")

    # When the user opts into the ML gate, surface the tradeoff loudly
    # at the top AND tag the saved bet records (gate_axis column) so
    # post-hoc analysis can split results by gate without ambiguity.
    if gate_axis == GATE_ENTROPY_KEY_ML:
        print()
        print("!" * 78)
        print("!!  --use-ml-entropy active: gating on race_entropy_ml (1/ml_decimal)")
        print("!!  This axis backtested WORSE than the default race_entropy_norm.")
        print("!!  Bets emitted under this gate are tagged gate_axis='race_entropy_ml'")
        print("!!  in bet_log.csv so you can split prospective results by axis.")
        print("!" * 78)

    # Tipmeerkat status banner. Default behavior (skip) is silent — quiet
    # is the new normal. Loud warning if the user opted in via
    # --include-tipmeerkat, since that data is currently bugged.
    if not skip_tipmeerkat:
        print()
        print("!" * 78)
        print("!!  --include-tipmeerkat active: tipmeerkat (tm_*) experts WILL be used.")
        print("!!  This data is currently bugged and may skew results.")
        print("!!  Recommend running without --include-tipmeerkat (the default).")
        print("!" * 78)

    # Print v3 trigger rules up-front so the per-race output below is
    # read with them already in context. (We used to print this as a
    # contradictory footer AFTER predict_v2's own trigger explanation,
    # which produced two opposing rule statements per run. Now we print
    # ours once here and suppress v2's via suppress_trigger_explanation.)
    print()
    print("v3 trigger rules:")
    print(f"  Trigger:       log_bet (ev_per_2 ≥ ${p2.LOGISTIC_EDGE_THRESHOLD:.2f}) "
          f"OR rdg_bet (ridge_value ≥ {p2.RIDGE_EDGE_MULTIPLIER:.2f}× break-even)")
    print(f"                 OR raw_bet (raw_value ≥ μ + {p2.RAW_EDGE_K}σ)")
    print(f"  Chalk rule:    {gate_axis} ≥ {q60:.4f} "
          f"AND ml_decimal ≤ {ML_MAX_FOR_BET}")
    print(f"  Longshot rule: {gate_axis} ≤ {q40:.4f} "
          f"AND ml_decimal ≥ {ML_MIN_LONGSHOT}")
    print(f"  Mid-ml zone ({ML_MAX_FOR_BET} < ml < {ML_MIN_LONGSHOT}) "
          f"excluded from both rules.")
    print(f"  Bet type:      LOG → W only; else W/WP/WPS by ml "
          f"(chalk → W or WP; longshot → WPS)")

    original_generate_bets = p2.generate_bets
    p2.generate_bets = _generate_bets_shim
    try:
        p2.run_predictions(
            date_str, track,
            num_sims=num_sims,
            use_regression=use_regression,
            skip_tipmeerkat=skip_tipmeerkat,
            suppress_trigger_explanation=True,
            race_banner_extras=_RACE_BANNER_EXTRAS,
        )
    finally:
        p2.generate_bets = original_generate_bets

    # Exacta box recommendations for multi-trigger RDG-only races.
    if _LAST_BET_DF is not None and not _LAST_BET_DF.empty:
        boxes = compute_exacta_boxes(_LAST_BET_DF)
        print_exacta_box_recommendations(boxes)

    # Optional: per-race human-pool report
    if show_human_picks:
        print_human_pool_report(
            _LAST_FULL_DF, date_str, track,
            skip_tipmeerkat=skip_tipmeerkat
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bet recommender v3 with two-rule entropy×ml gate '
                    '(chalk corner + longshot corner)'
    )
    parser.add_argument('--date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--track', required=True)
    parser.add_argument('--num-sims', type=int, default=100000,
                        help='Beyer→prob simulation count (default 100000)')
    parser.add_argument('--use-regression', action='store_true',
                        help='Use regression-based probabilities '
                             '(passed through to predict_v2 helpers)')
    parser.add_argument('--skip-tipmeerkat', action='store_true',
                        default=None,
                        help='[DEPRECATED — skip is now the default behavior] '
                             'Exclude tipmeerkat (tm_*) experts from this run. '
                             'Has no effect; passing this flag is a no-op. '
                             'Pass --include-tipmeerkat to opt back in.')
    parser.add_argument('--include-tipmeerkat', action='store_true',
                        help='Include tipmeerkat (tm_*) experts in the bet '
                             'recommendations. NOT recommended — tipmeerkat '
                             "data is currently bugged and skewing results. "
                             'Default behavior is to exclude tipmeerkat.')
    parser.add_argument('--human-picks', action='store_true',
                        help='After the run, print a per-race table of '
                             'human_pool / human_prob (the regression model\'s '
                             'expert-consensus feature). Useful for sanity-'
                             'checking how expert picks aggregate.')
    parser.add_argument('--use-ml-entropy', action='store_true',
                        help='Gate on ML-implied entropy (1/ml_decimal) '
                             'instead of the default ensemble_prob entropy. '
                             'WARNING: this axis backtested WORSE than the '
                             'default. Useful as a diagnostic, or if the '
                             'default cv-norm distribution has drifted from '
                             'calibration. Bets emitted under this gate are '
                             'tagged gate_axis="race_entropy_ml" in '
                             'bet_log.csv for clean post-hoc splitting.')
    args = parser.parse_args()

    # Resolve tipmeerkat behavior. Default is to SKIP tipmeerkat (its data
    # is currently bugged). --include-tipmeerkat opts back in. The legacy
    # --skip-tipmeerkat flag is accepted as a no-op for backward compat.
    skip_tipmeerkat = not args.include_tipmeerkat
    if args.skip_tipmeerkat:
        # User passed the deprecated flag explicitly. Their intent matches
        # the new default, but warn so they know they can drop the flag.
        print("  Note: --skip-tipmeerkat is now the default and is no longer "
              "needed. Future versions may remove this flag.")

    run_predictions_v3(
        args.date, args.track,
        num_sims=args.num_sims,
        use_regression=args.use_regression,
        skip_tipmeerkat=skip_tipmeerkat,
        show_human_picks=args.human_picks,
        gate_axis=(GATE_ENTROPY_KEY_ML if args.use_ml_entropy
                   else GATE_ENTROPY_KEY_DEFAULT),
    )


if __name__ == '__main__':
    main()