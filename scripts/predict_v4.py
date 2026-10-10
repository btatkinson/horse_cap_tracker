"""
predict_v4_corner.py

A betting rule based on z-scored 3-model agreement geometry.

The rule
--------
Each horse has three model values:
    ev_per_2     (Logistic expected value per $2 win bet)
    ridge_value  (Log-Ridge synthetic 4·W+2·P+1·S, already per-race
                  normalized by predict_v2 to sum to RIDGE_RACE_TOTAL)
    raw_value    (Raw-Ridge synthetic W+P+S — NOT per-race normalized
                  by predict_v2; we per-race normalize here)

Per-race normalize raw_value (raw_value / race_mean(raw_value)) so it's
on a comparable scale across races, then z-score all three using pooled
mean/std from walk_forward_summary.csv:
    z_i = (value_i - mean_i) / std_i

For each horse, compute the corner score:
    score = (z1 + z2 + z3) * sqrt(z1² + z2² + z3²)
          = dot_product([z1,z2,z3], [1,1,1]) * magnitude([z1,z2,z3])

A horse is BET if its score >= the (1 - bet_fraction) quantile of
historical corner scores (computed from walk_forward_summary.csv).

With bet_fraction=0.34 (default), that's the q66 of historical scores,
i.e. the cutoff above which the top 34% of historical horses sat. On
any given live card, the number of bets depends on how today's models'
output compares to the calibration distribution — could be more or
fewer than 34% of today's horses.

Why per-race normalize raw_value
--------------------------------
Without per-race normalization, raw_value's scale across races is governed
by input prob features. Heavy favorites have systematically higher raw_value
than longshots regardless of price, so pooled z-scoring would surface
favorites at the top of the corner score (this was an observed bug in
earlier versions). Per-race normalization puts every race's raw_value on
an "above/below this race's average" scale before pooling.

Why no octant filter
--------------------
The dot product with (1,1,1) is just z1+z2+z3, which is naturally negative
when most z's are negative and positive when they're positive. Multiplying
by magnitude (always non-negative) preserves sign. So the top scores end up
in the all-positive corner naturally — explicitly filtering for the octant
just adds a redundant gate.

Why historical cutoff (not per-card)
------------------------------------
A historical cutoff is what you asked for. It anchors the bet rule to a
backtested distribution. The trade-off: when live models drift cooler
than backtest, you bet fewer horses (potentially zero); when hotter,
you bet more. The mean/std for z-scoring is also historical (one global
calibration per model column from walk_forward_summary).

Re-run with --recompute-thresholds whenever walk_forward_summary.csv is
refreshed (new cards added, models retrained), or when --bet-fraction
differs from what was last cached. The threshold file auto-detects the
bet_fraction mismatch and recomputes.

Pre-requisites
--------------
walk_forward_summary.csv must exist with the three value columns populated.
Run backtest_walk_forward.py first if missing.

Threshold calibration
---------------------
Persisted to v4_thresholds.json:
    {
      "ev_per_2_mean":   ...,    "ev_per_2_std":   ...,
      "ridge_value_mean": ...,   "ridge_value_std": ...,
      "raw_value_pernorm_mean":  ...,   "raw_value_pernorm_std":  ...,
      "score_cutoff":     ...,   # (1 - bet_fraction) quantile of history
      "bet_fraction_calibrated_at": 0.34,
      "n_calibration_rows": ...,
      "calibrated_at": "YYYY-MM-DD HH:MM"
    }

Note that raw_value_pernorm is per-race normalized raw_value
(raw_value / per-race mean of raw_value). predict_v2 doesn't do this
normalization — the v4 pipeline applies it to both the calibration
sample and live predictions before z-scoring. See add_per_race_normalized_raw
for the rationale.

Usage
-----
    python predict_v4_corner.py --date 2026-04-30 --track churchill-downs
    python predict_v4_corner.py --date 2026-04-30 --track churchill-downs --bet-fraction 0.50
    python predict_v4_corner.py --date 2026-04-30 --track churchill-downs --recompute-thresholds

Limitations / things to watch
-----------------------------
- Pooled z-scoring weights all three models equally on the corner score
  axes. If one model is a much better predictor than the others, this rule
  treats them as equal.
- Calibration drift: if live models systematically differ from backtest,
  bet count will skew. Same drift concern as v3's entropy thresholds.
  Re-run with --recompute-thresholds when walk_forward_summary changes.
- Bet records are tagged gate_axis='corner_v4' in bet_log.csv so they
  don't blur with v3's chalk/longshot bets in post-hoc analysis.
"""

import os
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])

import predict_v2 as p2

# =============================================================================
# CONFIG
# =============================================================================

# Default fraction of today's card to bet. 0.34 = top 34% by corner score.
# Override per-run via --bet-fraction.
DEFAULT_BET_FRACTION = 0.34

# The three model value columns. Must all be present in
# walk_forward_summary.csv and in live predictions.csv.
# The three model value axes used to compute the corner score.
# raw_value gets per-race normalized first (see add_per_race_normalized_raw)
# because predict_v2 doesn't normalize it and pooled z-scoring it as-is
# would systematically favor heavy favorites.
VALUE_COLS = ['ev_per_2', 'ridge_value', 'raw_value_pernorm']

# Flat bet size in dollars (matches v2's $2 win bets).
FLAT_BET = 2.0

# Where calibration is persisted.
THRESHOLDS_PATH = DATA_PATH / 'predictions' / 'v4_thresholds.json'

# Tag for bet_log.csv so corner-rule bets don't blur with v3's bets.
GATE_AXIS_TAG = 'corner_v4'


# =============================================================================
# PER-RACE NORMALIZATION
# =============================================================================
# raw_value (raw-ridge model) is NOT per-race normalized by predict_v2 —
# its scale across races is governed by the input prob features, so heavy
# favorites get systematically higher raw_value than longshots regardless
# of price. To put it on the same footing as ridge_value (already per-race
# normalized) and ev_per_2 (value-based by construction), divide each
# horse's raw_value by the per-race mean. Result is centered around 1.0
# per race; pooled z-scoring then measures "above/below this race's average
# raw_value" rather than absolute raw_value level.

def add_per_race_normalized_raw(df, race_group_cols=('race',)):
    """Add raw_value_pernorm column = raw_value / race_mean(raw_value).

    For LIVE use (one card), race_group_cols=('race',) is right because
    the input df has one card's worth of races.

    For CALIBRATION use (walk_forward_summary spanning many cards),
    race_group_cols=('date', 'track', 'race') uniquely identifies a race.

    Races with mean ≤ 0 (degenerate) get NaN to avoid division blow-ups.
    """
    if 'raw_value' not in df.columns:
        raise RuntimeError("Cannot per-race normalize: raw_value column missing")
    race_mean = df.groupby(list(race_group_cols))['raw_value'].transform('mean')
    df = df.copy()
    df['raw_value_pernorm'] = np.where(race_mean > 0,
                                        df['raw_value'] / race_mean,
                                        np.nan)
    return df


# =============================================================================
# 1. CALIBRATION — z-score stats + 75th-percentile of corner score
# =============================================================================

def _compute_corner_score(z_array):
    """Vectorized: dot([z1,z2,z3], [1,1,1]) * sqrt(z1²+z2²+z3²) for each row.

    z_array: (N, 3) numpy array of z-scores.
    Returns: (N,) array of scores. Negative when most z's are negative
    (the dot product z1+z2+z3 dominates the sign), positive when most
    are positive. The all-positive corner naturally produces the highest
    scores without an explicit filter.
    """
    dot = z_array.sum(axis=1)
    mag = np.sqrt((z_array ** 2).sum(axis=1))
    return dot * mag


def calibrate_thresholds(bet_fraction):
    """Compute and persist the v4 z-score calibration + score cutoff
    from walk_forward_summary.csv.

    Pulls out the three value columns and computes:
      - pooled mean/std for each (used to normalize live predictions)
      - the (1 - bet_fraction) quantile of historical corner scores,
        which becomes the bet threshold

    Both pieces are persisted together. If the user re-runs with a
    different --bet-fraction, the cutoff is recomputed; the means/stds
    are unchanged (and could be re-used) but for clarity we always
    recompute the full payload together.

    Raises RuntimeError on missing data with a clear message.
    """
    wf_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
    if not wf_path.exists():
        raise RuntimeError(
            f"walk_forward_summary.csv not found at {wf_path}. "
            f"Run backtest_walk_forward.py first to generate it."
        )

    df = pd.read_csv(wf_path)
    # raw_value must exist as the source for per-race normalization.
    raw_input_cols = ['ev_per_2', 'ridge_value', 'raw_value']
    missing_inputs = [c for c in raw_input_cols if c not in df.columns]
    if missing_inputs:
        raise RuntimeError(
            f"walk_forward_summary.csv missing required input columns: "
            f"{missing_inputs}. Schema may have drifted; re-run "
            f"backtest_walk_forward.py."
        )

    # Per-race normalize raw_value across the historical sample. Each
    # race in walk_forward_summary is identified by (date, track, race).
    df = add_per_race_normalized_raw(df, race_group_cols=('date', 'track', 'race'))

    # Drop rows with any NaN in the three value columns (incomplete predictions
    # or degenerate races where raw_value pernorm couldn't be computed).
    sample = df[VALUE_COLS].dropna()
    n_full = len(sample)
    if n_full < 200:
        raise RuntimeError(
            f"Only {n_full} rows in walk_forward_summary.csv have all three "
            f"value columns populated — too few to z-score against. Need 200+. "
            f"Re-run backtest_walk_forward.py over more cards."
        )

    means = sample.mean()
    stds  = sample.std()
    if (stds == 0).any():
        zero_cols = [c for c, s in zip(VALUE_COLS, stds) if s == 0]
        raise RuntimeError(
            f"Zero std in calibration columns {zero_cols} — "
            f"can't z-score (would divide by zero). Check data quality."
        )

    # Compute historical scores and pull the (1-bet_fraction) quantile.
    z = ((sample - means) / stds).values
    scores = _compute_corner_score(z)
    cutoff = float(np.quantile(scores, 1.0 - bet_fraction))

    payload = {
        'ev_per_2_mean':           float(means['ev_per_2']),
        'ev_per_2_std':            float(stds['ev_per_2']),
        'ridge_value_mean':        float(means['ridge_value']),
        'ridge_value_std':         float(stds['ridge_value']),
        'raw_value_pernorm_mean':  float(means['raw_value_pernorm']),
        'raw_value_pernorm_std':   float(stds['raw_value_pernorm']),
        'score_cutoff':            cutoff,
        'bet_fraction_calibrated_at': bet_fraction,
        'n_calibration_rows': n_full,
        'calibrated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def load_thresholds(bet_fraction, recompute=False):
    """Return calibration payload, recomputing it from walk_forward_summary
    if the file is missing, --recompute-thresholds was passed, the cached
    payload is missing required keys, OR the cached cutoff was calibrated
    for a different bet_fraction than the one we're running with now."""
    required_keys = {f'{c}_mean' for c in VALUE_COLS} | \
                    {f'{c}_std'  for c in VALUE_COLS} | \
                    {'score_cutoff', 'bet_fraction_calibrated_at'}
    if recompute or not THRESHOLDS_PATH.exists():
        return calibrate_thresholds(bet_fraction)
    try:
        payload = json.loads(THRESHOLDS_PATH.read_text())
    except json.JSONDecodeError:
        return calibrate_thresholds(bet_fraction)
    if not required_keys.issubset(payload.keys()):
        return calibrate_thresholds(bet_fraction)
    # If the persisted cutoff was calibrated for a different bet_fraction,
    # recompute. The cutoff is bet_fraction-dependent, the mean/stds aren't,
    # but we recompute the whole payload for simplicity.
    if payload['bet_fraction_calibrated_at'] != bet_fraction:
        return calibrate_thresholds(bet_fraction)
    return payload


# =============================================================================
# 2. SCORING — apply calibration to live predictions
# =============================================================================

def add_corner_score_columns(df, calib):
    """Add z_log, z_ridge, z_raw, and corner_score columns to df.

    df is one card's worth of horses (multiple races). raw_value is
    per-race normalized within this card before z-scoring against the
    historical mean/std stored in calib. ridge_value and ev_per_2 are
    used as-is (they're already on appropriate scales).

    Horses with NaN in any of the three values get NaN z-scores and a
    NaN corner_score (NOT zero — zero would put them mid-pack in the
    ranking, which is wrong). The bet rule excludes NaN scores.

    Modifies df in place and returns it.
    """
    # Per-race normalize raw_value within this card.
    # The live df has one card's races identified by 'race' alone.
    df = add_per_race_normalized_raw(df, race_group_cols=('race',))

    z_log = (df['ev_per_2']           - calib['ev_per_2_mean'])           / calib['ev_per_2_std']
    z_rdg = (df['ridge_value']        - calib['ridge_value_mean'])        / calib['ridge_value_std']
    z_raw = (df['raw_value_pernorm']  - calib['raw_value_pernorm_mean'])  / calib['raw_value_pernorm_std']

    df['z_log']   = z_log
    df['z_ridge'] = z_rdg
    df['z_raw']   = z_raw

    # Vectorized score. NaN propagates naturally — a horse missing any
    # input gets NaN score and is excluded from the cutoff calculation.
    z_arr = np.column_stack([z_log.values, z_rdg.values, z_raw.values])
    df['corner_score'] = _compute_corner_score(z_arr)

    return df


# =============================================================================
# 3. BET GENERATION — replaces v2's generate_bets via shim
# =============================================================================

def generate_bets_v4(df, calib, takeout=None):
    """Build bet records using the corner-rule with HISTORICAL cutoff.

    Mirrors v2's generate_bets signature shape (returns bet_df + race_summaries
    dict) so the v3-style monkey-patch shim slots in cleanly.

    df is the post-prediction-merge dataframe with all three value columns
    AND ensemble_prob populated (predict_v2.run_predictions calls
    generate_bets after both ridge_value and raw_value are merged).

    The cutoff lives in calib['score_cutoff'] — the (1 - bet_fraction)
    quantile of corner scores computed from walk_forward_summary.csv.
    Any horse with corner_score >= score_cutoff gets bet. NaN scores
    (missing inputs) are excluded.

    Returns:
        bet_df: rows = bets fired, columns include z-scores, corner_score,
                triggered_by='CORNER_V4', gate_axis tag.
        race_summaries: per-race dict for v2's per-race printer.
    """
    df = add_corner_score_columns(df.copy(), calib)
    cutoff = calib['score_cutoff']

    qualifies = (df['corner_score'].notna()) & (df['corner_score'] >= cutoff)

    bets = []
    race_summaries = {}

    for race_num, race_df in df.groupby('race'):
        race_qualifiers = race_df[qualifies.loc[race_df.index]]
        race_top = (float(race_df['corner_score'].max())
                    if race_df['corner_score'].notna().any() else float('nan'))

        if race_qualifiers.empty:
            race_summaries[race_num] = {
                'action': 'PASS',
                'top_score': race_top,
                'historical_cutoff': cutoff,
            }
            continue

        for _, row in race_qualifiers.iterrows():
            hn = int(row['horse_number'])
            prob = float(row['ensemble_prob']) if pd.notna(row['ensemble_prob']) else np.nan
            bet_record = {
                'race': int(race_num),
                'horse_number': hn,
                'bet_type': 'WPS',  # match v2's default; rule doesn't change bet type
                'ensemble_prob':  round(prob, 4) if pd.notna(prob) else np.nan,
                'ml_prob':        round(row.get('ml_win_prob_norm', np.nan), 4),
                'edge':           round(row.get('edge', np.nan), 4),
                'ev_per_2':       round(row.get('ev_per_2', np.nan), 2),
                'ev_roi':         round(row.get('ev_roi', np.nan), 3),
                'ml_odds':        row.get('ml_odds_raw', ''),
                'ridge_value':         round(row.get('ridge_value', np.nan), 2),
                'raw_value':           round(row.get('raw_value', np.nan), 3),
                'raw_value_pernorm':   round(float(row.get('raw_value_pernorm', np.nan)), 3) \
                                       if pd.notna(row.get('raw_value_pernorm', np.nan)) else np.nan,

                # v4-specific fields
                'z_log':              round(float(row['z_log']),   3),
                'z_ridge':            round(float(row['z_ridge']), 3),
                'z_raw':              round(float(row['z_raw']),   3),
                'corner_score':       round(float(row['corner_score']), 3),
                'historical_cutoff':  round(cutoff, 3),
                'bet_fraction':       calib['bet_fraction_calibrated_at'],

                # Compatibility fields with v2/v3 bet_log so existing
                # downstream tooling doesn't need to special-case v4.
                # n_models_positive is informational only — v4 doesn't
                # require all positive (the score's sign naturally surfaces
                # positive-octant horses at the top of the distribution).
                'triggered_by':       'CORNER_V4',
                'n_models_positive':  int((row[['z_log','z_ridge','z_raw']] > 0).sum()),
                'is_agree':           bool((row[['z_log','z_ridge','z_raw']] > 0).all()),
                'kelly':              0.0,  # not computed by this rule (flat bet)
                'gate_axis':          GATE_AXIS_TAG,
                'bet_amount':         FLAT_BET,
            }
            bets.append(bet_record)

        race_summaries[race_num] = {
            'action': 'BET',
            'n_bet': len(race_qualifiers),
            'horses': [int(h) for h in race_qualifiers['horse_number']],
            'historical_cutoff': cutoff,
            'top_score': race_top,
        }

    bet_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    return bet_df, race_summaries


# =============================================================================
# 4. SHIM — plug into predict_v2.run_predictions via monkey-patch
# =============================================================================

# Module-level state. Same pattern as v3 (see predict_v3.py for the
# rationale on why a return-value approach doesn't work cleanly).
_LAST_BET_DF = None
_LAST_FULL_DF = None
_RACE_BANNER_EXTRAS = {}
_CALIB = None  # set once per run by run_predictions_v4


def _generate_bets_shim(df, **kwargs):
    """Adapter: lets generate_bets_v4 plug into predict_v2.run_predictions.

    Accepts (and ignores) all of v2's keyword args — v2 passes things like
    raw_edge_k, raw_min_k, etc. that don't apply to the corner rule.

    Captures bet_df / full df into module state so the post-run reporter
    can access them, and populates _RACE_BANNER_EXTRAS so v2's per-race
    print loop shows corner-score info in each banner.
    """
    global _LAST_BET_DF, _LAST_FULL_DF

    if _CALIB is None:
        raise RuntimeError(
            "Calibration not loaded — run_predictions_v4 must set _CALIB "
            "before calling p2.run_predictions."
        )

    takeout = kwargs.get('takeout', p2.WIN_TAKEOUT)
    bet_df, race_summaries = generate_bets_v4(df, _CALIB, takeout=takeout)

    _LAST_BET_DF = bet_df
    _LAST_FULL_DF = df

    # Per-race banner: top score on the race vs the historical cutoff.
    _RACE_BANNER_EXTRAS.clear()
    for race_num, summary in race_summaries.items():
        top    = summary.get('top_score', float('nan'))
        cutoff = summary.get('historical_cutoff', float('nan'))
        n_bet  = summary.get('n_bet', 0)
        top_str    = f"{top:.2f}"    if not np.isnan(top)    else "n/a"
        cutoff_str = f"{cutoff:.2f}" if not np.isnan(cutoff) else "n/a"
        _RACE_BANNER_EXTRAS[int(race_num)] = (
            f"corner: top score={top_str} (hist cutoff={cutoff_str}, "
            f"{n_bet} bet)"
        )

    return bet_df, race_summaries


# =============================================================================
# 5. ENTRY POINT
# =============================================================================

def run_predictions_v4(date_str, track, num_sims=100000, use_regression=False,
                       recompute_thresholds=False,
                       bet_fraction=DEFAULT_BET_FRACTION):
    """Run v2's full pipeline with the corner-rule swapped in for bet generation.

    The shim approach mirrors v3's: predict_v2 owns the data-loading and
    merging logic (which is intricate); we monkey-patch only the betting
    decision so we don't drift from the upstream loader.
    """
    global _CALIB

    if not (0.0 < bet_fraction < 1.0):
        raise ValueError(f"bet_fraction must be in (0, 1), got {bet_fraction}")

    # Load (or recompute) calibration first so we fail fast if data is missing.
    # The cutoff in calib is bet_fraction-dependent — load_thresholds
    # detects mismatch with the cached payload's bet_fraction and recomputes.
    _CALIB = load_thresholds(bet_fraction=bet_fraction,
                             recompute=recompute_thresholds)

    print(f"\n{'='*72}")
    print(f"PREDICT V4 — corner-rule (dot × magnitude on z-scored 3-model values)")
    print(f"{'='*72}")
    print(f"  Calibration source: walk_forward_summary.csv")
    print(f"    n rows used:     {_CALIB['n_calibration_rows']}")
    print(f"    calibrated at:   {_CALIB['calibrated_at']}")
    print(f"  Z-score reference (per-model mean ± std from history):")
    for c in VALUE_COLS:
        print(f"    {c:13s}  μ={_CALIB[c+'_mean']:+.3f}  σ={_CALIB[c+'_std']:.3f}")
    pct = bet_fraction * 100
    print(f"  Bet rule: corner_score >= historical q{100-pct:.0f} "
          f"(= {_CALIB['score_cutoff']:.3f})")
    print(f"            i.e. score in top {pct:.0f}% of walk_forward history")
    print(f"            flat ${FLAT_BET:.0f} per bet, no octant filter")
    print()

    # Monkey-patch v2.generate_bets, call v2.run_predictions, restore.
    original_generate_bets = p2.generate_bets
    p2.generate_bets = _generate_bets_shim
    try:
        p2.run_predictions(
            date_str, track, num_sims=num_sims,
            use_regression=use_regression,
            race_banner_extras=_RACE_BANNER_EXTRAS,
            suppress_trigger_explanation=True,
        )
    finally:
        p2.generate_bets = original_generate_bets

    # Post-run summary
    if _LAST_BET_DF is not None and not _LAST_BET_DF.empty:
        n = len(_LAST_BET_DF)
        races_with_bet = _LAST_BET_DF['race'].nunique()
        avg_score = _LAST_BET_DF['corner_score'].mean()
        cutoff = _LAST_BET_DF['historical_cutoff'].iloc[0]
        n_all_pos = int(_LAST_BET_DF['is_agree'].sum())
        print(f"\n  v4 corner-rule summary:")
        print(f"    {n} bet(s) across {races_with_bet} race(s)")
        print(f"    historical score cutoff: {cutoff:.3f}")
        print(f"    avg corner_score on bets: {avg_score:.3f}")
        print(f"    avg z-scores: log={_LAST_BET_DF['z_log'].mean():+.2f}, "
              f"ridge={_LAST_BET_DF['z_ridge'].mean():+.2f}, "
              f"raw={_LAST_BET_DF['z_raw'].mean():+.2f}")
        print(f"    of {n} bets, {n_all_pos} have all 3 z>0 "
              f"(informational — not a bet requirement)")
    else:
        print(f"\n  v4 corner-rule: no bets fired on this card.")
        print(f"    No horse cleared the historical score cutoff "
              f"({_CALIB['score_cutoff']:.3f}).")
        print(f"    The day's models may be running cooler than backtest history.")
        print(f"    Options: (1) re-run with --recompute-thresholds if your")
        print(f"    walk_forward_summary.csv has been refreshed, (2) raise")
        print(f"    --bet-fraction to be more permissive (e.g. 0.50), or")
        print(f"    (3) just accept the pass — that's what the rule prescribes.")


def main():
    parser = argparse.ArgumentParser(
        description='predict_v4_corner — corner-rule bet selector '
                    '(historical cutoff)'
    )
    parser.add_argument('--date',  required=True, help='YYYY-MM-DD')
    parser.add_argument('--track', required=True, help='e.g. churchill-downs')
    parser.add_argument('--num-sims', type=int, default=100000)
    parser.add_argument('--use-regression', action='store_true')
    parser.add_argument('--bet-fraction', type=float, default=DEFAULT_BET_FRACTION,
                        help=f'historical-percentile fraction. '
                             f'0.34 = bet horses scoring >= q66 of historical scores. '
                             f'Default {DEFAULT_BET_FRACTION}.')
    parser.add_argument('--recompute-thresholds', action='store_true',
                        help='Refresh v4_thresholds.json from current '
                             'walk_forward_summary.csv before running')
    args = parser.parse_args()

    run_predictions_v4(
        args.date, args.track,
        num_sims=args.num_sims,
        use_regression=args.use_regression,
        recompute_thresholds=args.recompute_thresholds,
        bet_fraction=args.bet_fraction,
    )


if __name__ == '__main__':
    main()