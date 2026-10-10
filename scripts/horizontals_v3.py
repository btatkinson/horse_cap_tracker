"""
horizontals_v6.py

Pick 3/4/5/6 ticket recommender — saddle-scored variant of horizontals_v5.

What changed vs v5
------------------
v5 ranked candidates by `ridge_value` and gated each leg with `min_ridge`.
That treated every leg the same regardless of race shape, which produced
near-uniform "boxy" tickets (4×4×4) even when the data said different
race shapes deserved very different widths.

v6 replaces the ranking signal with a saddle-augmented score:

    score(h) = ridge_value(h) * max(0.1, 1 + λ * b_corner(h))

where b_corner is a 2D saddle defined on (race_entropy, horse_ml_decimal):

    x = (race_entropy   - hist_median_entropy) / σ_x
    y = (horse_ml_decimal - hist_median_ml)    / σ_y
    b_corner(h) = -x * y

    Sign analysis:
      flat race  (x>0) + chalk    horse (y<0) → b > 0   (flat+chalk corner)
      peaked race (x<0) + longshot horse (y>0) → b > 0   (peaked+longshot)
      flat       + longshot                    → b < 0
      peaked     + chalk                       → b < 0

The two positive corners are the +EV cells of the user's backtest
(flat+chalk +59% in-sample; peaked+longshot showing prospective edge).
The two negative corners are -EV cells; horses there get score-penalized
but not banned. Mid-band horses (near both medians) get b ≈ 0 → fall
back to raw ridge ranking, which by construction targets the dead zone
of the table where ROI is uniformly bad — and the existing strict-EV
gate filters most of them out.

Build loop changes:
  - Seed = top SCORE per leg (no longer top ridge). This lets a 3/1
    weak favorite in a flat race seed instead of a 4/5 chalk, because
    the chalk gets a smaller corner bonus AND has lower raw ridge.
  - Add candidates by marginal-SCORE-per-dollar (was ridge).
  - Stop when next add has score ≤ 2 (strict-EV gate, unchanged in
    spirit), OR cost would exceed --max-cost.
  - Hard quality floor: ridge_value < 1.0 horses excluded regardless
    of corner bonus. The corner can boost a 1.5-ridge horse but it
    can't rescue a 0.5-ridge horse from itself.
  - Hit-prob floor (--min-hit-prob) is a play/skip INDICATOR after
    the build, not a stop condition during the build.

Play/skip rule unchanged
------------------------
v5's seed_ridge_geomean play/skip rule was validated on 135 Pick-3
sequences built with v5's seed-by-ridge. v6 changes the seed, so the
geomean would be on a different distribution than what was calibrated.
To keep the rule's empirical guard intact, the play/skip evaluation
computes seed_ridge_geomean using v5's seed-by-top-ridge logic, even
though the actual v6 ticket uses seed-by-top-score. Conservative: if
v5 would have skipped, v6 also skips.

Tunables (CLI):
  --lambda           saddle weight in score formula (default 0.4)
  --max-cost         dollar cap on ticket combinations (per-bet-type
                     defaults: P3=24 P4=48 P5=100 P6=200)
  --min-hit-prob     hit-probability floor (default 0.005)
  --ridge-floor      hard exclusion below this raw ridge_value (default 1.0)

Usage:
    python horizontals_v6.py --date 2026-04-30 --track churchill-downs --races 4-6
    python horizontals_v6.py --date 2026-04-30 --track churchill-downs --races 4-7 --lambda 0.6
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])

# Defaults — picked from the conversation, all overridable via CLI.
DEFAULT_LAMBDA = 0.4
DEFAULT_MIN_HIT_PROB = 0.005
DEFAULT_RIDGE_FLOOR = 1.0
DEFAULT_MAX_COST = {3: 24, 4: 48, 5: 100, 6: 200}

# Score floor: max(0.1, 1 + λ·b) prevents a horse with strongly negative
# saddle from getting a negative score (which would invert ranking).
SCORE_FLOOR_FACTOR = 0.1

# Play/skip rule (preserved from v5)
PLAY_FRACTION = 0.33
MIN_HIST_FOR_BETTYPE_CUTOFF = 50


# =============================================================================
# 1. LOADING (consistent with v5)
# =============================================================================

def load_predictions(date_str, track):
    pred_path = DATA_PATH / 'predictions' / date_str / track / 'predictions.csv'
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions at {pred_path}")
    df = pd.read_csv(pred_path)
    df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
    if 'horse_number' in df.columns:
        df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').astype('Int64')
    return df


def discover_cards():
    pred_root = DATA_PATH / 'predictions'
    if not pred_root.exists():
        return []
    cards = []
    for date_dir in pred_root.iterdir():
        if not date_dir.is_dir() or not _looks_like_date(date_dir.name):
            continue
        for track_dir in date_dir.iterdir():
            if not track_dir.is_dir():
                continue
            pred = track_dir / 'predictions.csv'
            entries = DATA_PATH / 'entries' / f'{track_dir.name}_{date_dir.name}.csv'
            meta = DATA_PATH / 'meta' / f'{track_dir.name}_{date_dir.name}.csv'
            if pred.exists() and entries.exists() and meta.exists():
                cards.append((date_dir.name, track_dir.name))
    return sorted(cards)


def _looks_like_date(s):
    try:
        pd.to_datetime(s, format='%Y-%m-%d')
        return True
    except (ValueError, TypeError):
        return False


def parse_race_range(race_str):
    if '-' in race_str and ',' not in race_str:
        lo, hi = race_str.split('-')
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(r.strip()) for r in race_str.split(','))


# =============================================================================
# 2. SADDLE SCORING — historical centering
# =============================================================================

def compute_saddle_calibration():
    """Compute (median, IQR-based-σ) for race_entropy_norm and ml_decimal
    from walk_forward_summary.csv.

    Used to center+scale the saddle: anchoring on historical medians
    matches the entropy×ml backtest cuts (which used absolute thresholds
    derived from the same sample), so the saddle's "zero" lines up with
    the implicit zero of the validated ROI table.

    Returns dict with med_entropy, sigma_entropy, med_ml, sigma_ml.
    Raises RuntimeError if the file's missing or unusable — same fail-
    loud convention as predict_v3 (no silent fallback to a default
    that could quietly mis-calibrate the saddle).
    """
    wf_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
    if not wf_path.exists():
        raise RuntimeError(
            f"walk_forward_summary.csv not found at {wf_path}. "
            f"v6 needs it to compute the saddle's center+scale. "
            f"Run backtest_walk_forward.py to generate it."
        )

    df = pd.read_csv(wf_path)

    # Per-race entropy (one value per race). The saved CSV already has
    # ensemble_prob; recompute per-race entropy here so we don't depend
    # on whatever derived columns happen to be cached.
    if 'ensemble_prob' not in df.columns:
        raise RuntimeError(
            f"walk_forward_summary.csv missing 'ensemble_prob'. "
            f"Re-generate with the current predict_v2 output schema."
        )

    keys = ['date', 'track', 'race']
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise RuntimeError(
            f"walk_forward_summary.csv missing key columns: {missing}"
        )

    race_ent = (df.groupby(keys)['ensemble_prob']
                  .apply(_compute_race_entropy_norm).dropna())
    if race_ent.empty:
        raise RuntimeError("Couldn't compute any race entropies — empty sample.")

    if 'ml_decimal' not in df.columns:
        raise RuntimeError(
            f"walk_forward_summary.csv missing 'ml_decimal'."
        )
    ml = df['ml_decimal'].dropna()
    if ml.empty:
        raise RuntimeError("ml_decimal column has no valid values.")

    # IQR-based σ — robust to outliers. (Q75-Q25)/1.349 ≈ σ for normal.
    def _robust_sigma(s):
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        sig = (q75 - q25) / 1.349
        # If IQR is degenerate (constant data), fall back to std.
        return sig if sig > 1e-9 else float(s.std() or 1.0)

    return {
        'med_entropy':   float(race_ent.median()),
        'sigma_entropy': _robust_sigma(race_ent),
        'med_ml':        float(ml.median()),
        'sigma_ml':      _robust_sigma(ml),
        'n_races':       int(len(race_ent)),
        'n_horses':      int(len(ml)),
    }


def _compute_race_entropy_norm(probs):
    """H/log(N) of the ensemble_prob across one race. Matches the
    function used in predict_v3 / backtest_kfold so the calibration
    centers we compute here are on the same scale as their thresholds."""
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
    return -np.sum(p * np.log(p)) / np.log(len(p))


def add_score_column(pred_df, calib, lam):
    """Compute score = ridge_value * max(0.1, 1 + λ·b_corner) per horse.

    Adds three columns to pred_df:
        race_entropy_norm — entropy per race, broadcast to all horses
        b_corner          — saddle value per horse
        score             — final ranking score per horse

    b_corner uses normalized centered values:
        x = (race_entropy - med_entropy) / sigma_entropy
        y = (ml_decimal   - med_ml)      / sigma_ml
        b = -x * y

    For horses with missing ml_decimal, b_corner = 0 — they fall back
    to raw ridge ranking. (No bonus, no penalty; the saddle requires
    both axes to be defined.)

    pred_df is modified in place (race_entropy_norm, b_corner, score
    columns) and returned for chaining.
    """
    # Per-race entropy of ensemble_prob, broadcast to every horse in the race.
    if 'ensemble_prob' not in pred_df.columns:
        raise RuntimeError("predictions.csv missing 'ensemble_prob' column")

    by_race = (pred_df.groupby('race')['ensemble_prob']
                      .apply(_compute_race_entropy_norm))
    by_race.name = 'race_entropy_norm'
    pred_df = pred_df.merge(
        by_race.reset_index(), on='race', how='left', suffixes=('', '_drop')
    )
    if 'race_entropy_norm_drop' in pred_df.columns:
        pred_df = pred_df.drop(columns=['race_entropy_norm_drop'])

    # Saddle on normalized (x, y).
    x = (pred_df['race_entropy_norm'] - calib['med_entropy']) / calib['sigma_entropy']
    y_raw = pred_df.get('ml_decimal', pd.Series(np.nan, index=pred_df.index))
    y = (y_raw - calib['med_ml']) / calib['sigma_ml']

    pred_df['b_corner'] = -(x * y)
    # Where ml is missing, saddle is undefined — set to 0 (neither bonus
    # nor penalty). The horse still gets to compete on raw ridge.
    pred_df.loc[pred_df['b_corner'].isna(), 'b_corner'] = 0.0

    multiplier = (1.0 + lam * pred_df['b_corner']).clip(lower=SCORE_FLOOR_FACTOR)
    pred_df['score'] = pred_df['ridge_value'] * multiplier

    return pred_df


# =============================================================================
# 3. BUILD LOOP — score-driven greedy
# =============================================================================

def build_ticket_v6(pred_df, race_numbers, lam, max_cost, ridge_floor,
                    calib, _override_seed_signal=None):
    """Greedy: seed each leg with top-score horse, repeatedly add the
    candidate with highest marginal-score-per-dollar across all legs.

    Stops when:
      - no candidate has ridge_value ≥ ridge_floor (hard quality floor)
      - or no candidate has score > 2 (strict-EV gate, unchanged in spirit
        from v5 — adding a horse below score 2 means the ticket pays
        less than its added cost on the seed-only line)
      - or adding the next horse would exceed max_cost

    _override_seed_signal: internal hook used by the play/skip rule to
    reproduce v5's seed-by-top-ridge for empirical comparison. Production
    callers should leave it None (uses score-based seeding).

    Returns ticket dict with cost, hit_prob, allocation, selections, seeds.
    Returns None if any leg has no candidates clearing ridge_floor.
    """
    pred_df = pred_df.copy()
    if 'score' not in pred_df.columns:
        pred_df = add_score_column(pred_df, calib, lam)

    df_pick = pred_df[pred_df['race'].isin(race_numbers)].copy()
    if df_pick.empty:
        return None

    # Hard ridge floor — applied BEFORE any sorting. A horse below this
    # ridge can't get on the ticket regardless of corner bonus.
    df_pick = df_pick[df_pick['ridge_value'] >= ridge_floor]
    if df_pick.empty:
        return None

    needed_cols = ['ridge_value', 'ensemble_prob', 'score', 'horse_number']
    if not all(c in df_pick.columns for c in needed_cols):
        return None

    # The seed-signal is what we sort the per-leg candidate list by.
    # By default it's 'score'; the play/skip helper passes 'ridge_value'
    # to reproduce v5's behavior for the geomean indicator.
    seed_signal = _override_seed_signal or 'score'

    sorted_horses = {}
    for race_num in sorted(race_numbers):
        race_data = df_pick[df_pick['race'] == race_num].dropna(
            subset=['ridge_value', 'ensemble_prob', 'score']
        )
        if race_data.empty:
            return None
        race_data = race_data.sort_values(seed_signal, ascending=False)
        rows = []
        for _, r in race_data.iterrows():
            rows.append({
                'horse_number': int(r['horse_number']),
                'ridge_value':  float(r['ridge_value']),
                'ensemble_prob': float(r['ensemble_prob']),
                'score':        float(r['score']),
                'b_corner':     float(r.get('b_corner', 0.0)),
            })
        sorted_horses[race_num] = rows

    active_races = sorted(race_numbers)
    if any(r not in sorted_horses for r in active_races):
        return None

    allocation = [1] * len(active_races)
    selections = {r: [sorted_horses[r][0]['horse_number']] for r in active_races}

    while True:
        best_idx, best_mv = None, -np.inf
        for i, race_num in enumerate(active_races):
            next_pos = allocation[i]
            if next_pos >= len(sorted_horses[race_num]):
                continue
            cand = sorted_horses[race_num][next_pos]
            # Strict-EV stop: candidate's score must clear the per-$2-ticket
            # cost. With score = ridge·(1+λb), a corner-bonused horse with
            # raw ridge below 2 can clear here — that's intentional.
            if cand['score'] <= 2:
                continue
            other_product = 1
            for j in range(len(active_races)):
                if j != i:
                    other_product *= allocation[j]
            mv = (cand['score'] - 2) / other_product
            if mv > best_mv:
                best_mv = mv
                best_idx = i

        if best_idx is None or best_mv <= 0:
            break

        # Cost cap: prospective allocation if we add this candidate.
        new_alloc = allocation.copy()
        new_alloc[best_idx] += 1
        if int(np.prod(new_alloc)) > max_cost:
            break

        race_num = active_races[best_idx]
        next_pos = allocation[best_idx]
        added = sorted_horses[race_num][next_pos]
        selections[race_num].append(added['horse_number'])
        allocation[best_idx] += 1

    cost = int(np.prod(allocation))
    hit_prob = 1.0
    for i, race_num in enumerate(active_races):
        race_hit = sum(
            sorted_horses[race_num][j]['ensemble_prob']
            for j in range(allocation[i])
        )
        hit_prob *= max(race_hit, 1e-9)

    seeds = {r: sorted_horses[r][0] for r in active_races}

    return {
        'cost': cost,
        'hit_prob': hit_prob,
        'allocation': allocation,
        'selections': selections,
        'seeds': seeds,
    }


def compute_seed_ridge_geomean(ticket):
    """Geomean of seed-horse ridge_value across legs. Used by the
    play/skip indicator (v5's validated rule)."""
    seeds = ticket['seeds']
    seed_ridges = [max(seeds[r]['ridge_value'], 1e-9) for r in seeds]
    return float(np.exp(np.mean(np.log(seed_ridges))))


# =============================================================================
# 4. PLAY/SKIP — preserved from v5, applied to a v5-shaped ticket
# =============================================================================

def evaluate_play_skip(pred_df, race_numbers, calib, lam, ridge_floor,
                       max_cost):
    """Run v5's seed_ridge_geomean play/skip rule using a v5-shaped seed.

    The empirical rule was calibrated on tickets where seed = top ridge.
    To keep the calibration valid, this function builds a parallel ticket
    using top-ridge seeding (NOT v6's score seeding), computes the
    geomean of THAT seed list, and compares against the historical
    Pick-3 cutoff. Conservative: if v5 would have flagged the sequence
    as a skip, v6 also skips.

    Returns (decision, info) where decision is 'PLAY' or 'SKIP' and
    info is a dict with geomean, threshold, transferred-from-Pick-3 flag,
    historical sample size.
    """
    # Build a v5-shaped ticket purely for the geomean. Use raw ridge as
    # the seed signal so the seeds match v5's logic; everything else can
    # follow v6 since the geomean only needs the seed identities.
    v5_shape = build_ticket_v6(
        pred_df, race_numbers, lam=lam, max_cost=max_cost,
        ridge_floor=ridge_floor, calib=calib,
        _override_seed_signal='ridge_value',
    )
    if v5_shape is None:
        return 'SKIP', {'reason': 'no v5-shape ticket buildable'}

    geomean = compute_seed_ridge_geomean(v5_shape)
    thresh_info = compute_historical_threshold(len(race_numbers))
    if thresh_info is None:
        return 'PLAY', {'reason': 'no historical data — defaulting to play',
                        'geomean': geomean}

    threshold = thresh_info['threshold']
    decision = 'PLAY' if geomean <= threshold else 'SKIP'
    return decision, {
        'geomean': geomean,
        'threshold': threshold,
        'transferred': thresh_info.get('transferred', False),
        'n_seqs': thresh_info['n_seqs'],
        'source_legs': thresh_info['source_legs'],
        'own_legs_n': thresh_info.get('own_legs_n'),
    }


def compute_historical_threshold(bet_type_legs):
    """Same as v5: walk historical cards, build v5-shaped tickets for
    each consecutive N-race window with a real payout, return the
    PLAY_FRACTION quantile of seed_ridge_geomean.

    Falls back to Pick 3's distribution when bet-type sample is below
    MIN_HIST_FOR_BETTYPE_CUTOFF (the geomean is scale-invariant across
    leg counts).
    """
    cards = discover_cards()
    if not cards:
        return None

    own = _collect_geomeans_v5shape(cards, bet_type_legs)
    if len(own) >= MIN_HIST_FOR_BETTYPE_CUTOFF:
        return {
            'threshold':   float(np.quantile(own, PLAY_FRACTION)),
            'source_legs': bet_type_legs,
            'n_seqs':      len(own),
            'transferred': False,
        }

    if bet_type_legs == 3:
        if not own:
            return None
        return {
            'threshold':   float(np.quantile(own, PLAY_FRACTION)),
            'source_legs': 3,
            'n_seqs':      len(own),
            'transferred': False,
        }

    pick3 = _collect_geomeans_v5shape(cards, 3)
    if not pick3:
        return None
    return {
        'threshold':   float(np.quantile(pick3, PLAY_FRACTION)),
        'source_legs': 3,
        'n_seqs':      len(pick3),
        'transferred': True,
        'own_legs_n':  len(own),
    }


def _collect_geomeans_v5shape(cards, n_legs):
    """For each card, find every consecutive N-race window with a real
    horizontal payout in the meta CSV, build a v5-shaped ticket (top-ridge
    seed), compute seed_ridge_geomean. Returns the list."""
    payout_col = (f'pick_{n_legs}_$2_payout'
                  if n_legs > 2 else 'daily_double_$2_payout')
    geomeans = []

    # Saddle calibration is needed to pass through to build_ticket_v6,
    # but with override='ridge_value' the saddle doesn't affect the
    # seed-list — so we can build calibration once outside the loop.
    try:
        calib = compute_saddle_calibration()
    except RuntimeError:
        return geomeans  # graceful degradation: no historical = no rule

    for date_str, track in cards:
        try:
            pred = pd.read_csv(
                DATA_PATH / 'predictions' / date_str / track / 'predictions.csv'
            )
            meta = pd.read_csv(DATA_PATH / 'meta' / f'{track}_{date_str}.csv')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue

        pred['race'] = pd.to_numeric(pred['race'], errors='coerce').astype('Int64')
        if 'horse_number' in pred.columns:
            pred['horse_number'] = pd.to_numeric(
                pred['horse_number'], errors='coerce').astype('Int64')
        if 'race' in meta.columns:
            meta['race'] = pd.to_numeric(meta['race'], errors='coerce').astype('Int64')

        if payout_col not in meta.columns:
            continue

        races_in_card = sorted(pred['race'].dropna().unique())
        for i in range(len(races_in_card) - n_legs + 1):
            seq = list(races_in_card[i:i + n_legs])
            if not all(seq[j+1] == seq[j]+1 for j in range(len(seq)-1)):
                continue
            row = meta[meta['race'] == seq[-1]]
            if row.empty or pd.isna(row.iloc[0][payout_col]):
                continue

            ticket = build_ticket_v6(
                pred, seq, lam=DEFAULT_LAMBDA,
                max_cost=DEFAULT_MAX_COST.get(n_legs, 200),
                ridge_floor=DEFAULT_RIDGE_FLOOR,
                calib=calib,
                _override_seed_signal='ridge_value',
            )
            if ticket is None:
                continue
            geomeans.append(compute_seed_ridge_geomean(ticket))

    return geomeans


# =============================================================================
# 5. RECOMMEND
# =============================================================================

def recommend(date_str, track, race_numbers, lam, max_cost, min_hit_prob,
              ridge_floor):
    n_legs = len(race_numbers)
    bet_name = {3: 'Pick 3', 4: 'Pick 4', 5: 'Pick 5',
                6: 'Pick 6'}.get(n_legs, f'Pick {n_legs}')

    print(f"\n{'='*72}")
    print(f"{bet_name} RECOMMENDER v6  ({date_str} / {track} / races {race_numbers})")
    print(f"{'='*72}")
    print(f"  λ = {lam}   max_cost = ${max_cost}   "
          f"min_hit_prob = {min_hit_prob:.2%}   ridge_floor = {ridge_floor}")

    # --- Saddle calibration ---
    try:
        calib = compute_saddle_calibration()
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        return
    print(f"\n  Saddle calibration (from walk_forward_summary.csv):")
    print(f"    entropy: median={calib['med_entropy']:.4f}, "
          f"σ={calib['sigma_entropy']:.4f}  ({calib['n_races']} races)")
    print(f"    ml_dec : median={calib['med_ml']:.2f}, "
          f"σ={calib['sigma_ml']:.2f}  ({calib['n_horses']} horses)")

    # --- Load + score this card ---
    pred = load_predictions(date_str, track)
    available = sorted(pred['race'].dropna().unique())
    missing = [r for r in race_numbers if r not in available]
    if missing:
        print(f"\n  ERROR: races {missing} not in predictions; available={available}")
        return

    pred = add_score_column(pred, calib, lam)

    # --- Build the ticket ---
    ticket = build_ticket_v6(pred, race_numbers, lam=lam,
                             max_cost=max_cost, ridge_floor=ridge_floor,
                             calib=calib)
    if ticket is None:
        print("\n  ERROR: couldn't build ticket — every leg blocked by ridge_floor?")
        return

    # --- Print ticket ---
    print(f"\n  Proposed ticket (v6: seed by score, fill by score-marginal-EV)")
    print(f"    Cost: ${ticket['cost']}")
    print(f"    Predicted hit prob: {ticket['hit_prob']:.2%}")
    print(f"    Allocation: {ticket['allocation']}")
    for r in sorted(ticket['selections']):
        horses = ticket['selections'][r]
        seed = ticket['seeds'][r]
        # Per-race entropy from the merged column on pred.
        race_ent_series = pred[pred['race'] == r]['race_entropy_norm']
        race_ent = (float(race_ent_series.iloc[0])
                    if not race_ent_series.empty else float('nan'))
        ent_tag = ('flat' if race_ent > calib['med_entropy']
                   else 'peaked' if race_ent < calib['med_entropy']
                   else 'mid')
        print(f"      R{r}: {horses}")
        print(f"          seed=#{seed['horse_number']} "
              f"(score=${seed['score']:.2f}, ridge=${seed['ridge_value']:.2f}, "
              f"b_corner={seed['b_corner']:+.2f})  "
              f"race entropy={race_ent:.3f} [{ent_tag}]")

    # --- Hit-prob floor check ---
    if ticket['hit_prob'] < min_hit_prob:
        print(f"\n  ⚠ hit_prob ({ticket['hit_prob']:.3%}) is below the floor "
              f"({min_hit_prob:.2%}) — the ticket is too lottery-shaped.")
        print(f"    Either widen --max-cost, or accept the variance, or skip.")

    # --- Play/skip rule (v5's validated rule, applied to v5-shape) ---
    print(f"\n  Play/skip evaluation (v5 rule, on v5-shape seed_ridge_geomean):")
    decision, info = evaluate_play_skip(
        pred, race_numbers, calib=calib, lam=lam,
        ridge_floor=ridge_floor, max_cost=max_cost
    )
    if 'geomean' in info:
        print(f"    seed_ridge_geomean (v5-shape): {info['geomean']:.3f}")
    if 'threshold' in info:
        print(f"    Historical {PLAY_FRACTION*100:.0f}-pct threshold: "
              f"{info['threshold']:.3f}")
        if info.get('transferred'):
            print(f"    (Threshold from Pick 3; only {info['own_legs_n']} "
                  f"{bet_name} seqs available, need {MIN_HIST_FOR_BETTYPE_CUTOFF}+)")
        print(f"    (Based on {info['n_seqs']} historical sequences from "
              f"Pick {info['source_legs']})")

    print(f"\n  → {decision}")
    if decision == 'PLAY' and ticket['hit_prob'] >= min_hit_prob:
        print(f"    seed_ridge_geomean is in the bottom {PLAY_FRACTION*100:.0f}% "
              f"of historical sequences AND hit_prob clears the floor.")
    elif decision == 'PLAY':
        print(f"    seed_ridge_geomean clears the v5 rule, BUT the ticket's "
              f"hit_prob ({ticket['hit_prob']:.2%}) is below the floor "
              f"({min_hit_prob:.2%}). Consider widening or skipping.")
    else:
        if 'geomean' in info and 'threshold' in info:
            excess = info['geomean'] - info['threshold']
            print(f"    seed_ridge_geomean ({info['geomean']:.3f}) exceeds "
                  f"threshold ({info['threshold']:.3f}) by {excess:.2f}.")
            print(f"    v5 backtest: high-conviction-everywhere sequences")
            print(f"    historically lost money even with strict strategy.")

    # --- Caveats ---
    if n_legs > 3:
        print(f"\n  ⚠ The play/skip rule was validated on Pick 3. {bet_name} "
              f"validation is by transfer, not direct evidence.")
    if 'n_seqs' in info and info['n_seqs'] < 100:
        print(f"\n  ⚠ Threshold from only {info['n_seqs']} sequences — "
              f"borderline decisions are uncertain.")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pick 3/4/5/6 recommender with saddle-augmented scoring (v6)'
    )
    parser.add_argument('--date', required=True)
    parser.add_argument('--track', required=True)
    parser.add_argument('--races', required=True,
                        help='"4-6" for Pick 3, "4-7" Pick 4, etc.')
    parser.add_argument('--lambda', dest='lam', type=float, default=DEFAULT_LAMBDA,
                        help=f'saddle weight in score formula '
                             f'(default {DEFAULT_LAMBDA})')
    parser.add_argument('--max-cost', type=int, default=None,
                        help='cap on ticket combinations '
                             '(default: P3=24 P4=48 P5=100 P6=200)')
    parser.add_argument('--min-hit-prob', type=float,
                        default=DEFAULT_MIN_HIT_PROB,
                        help=f'hit-prob floor for play recommendation '
                             f'(default {DEFAULT_MIN_HIT_PROB})')
    parser.add_argument('--ridge-floor', type=float, default=DEFAULT_RIDGE_FLOOR,
                        help=f'hard exclusion below this raw ridge_value '
                             f'(default {DEFAULT_RIDGE_FLOOR})')
    args = parser.parse_args()

    race_numbers = parse_race_range(args.races)
    n_legs = len(race_numbers)
    if n_legs < 3 or n_legs > 6:
        print(f"--races must specify 3-6 consecutive races (got {n_legs})")
        return

    max_cost = args.max_cost if args.max_cost is not None \
        else DEFAULT_MAX_COST.get(n_legs, 200)

    recommend(args.date, args.track, race_numbers,
              lam=args.lam, max_cost=max_cost,
              min_hit_prob=args.min_hit_prob,
              ridge_floor=args.ridge_floor)


if __name__ == '__main__':
    main()