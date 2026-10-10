"""
predict_races.py

Generate race predictions and bet recommendations from AI picks + fitted params.

Usage:
    python predict_races.py --date 2026-04-08 --track keeneland
    python predict_races.py --date 2026-04-08 --track keeneland --sims 200000

Reads from:
    {DATA_PATH}/ai_picks/{date}/{track}/claude.csv   (and gemini.csv, openai.csv)
    {DATA_PATH}/upc_entries/{track}.csv               (scratches + ML odds)
    {DATA_PATH}/fitted_params.json                    (from backtest)
    {DATA_PATH}/regression_model.json                 (logistic CV model)
    {DATA_PATH}/ridge_value_model.json                (ridge value CV model)

Writes to:
    {DATA_PATH}/predictions/{date}/{track}/predictions.csv
    {DATA_PATH}/predictions/{date}/{track}/bet_log.csv

Bet trigger (when both logistic and ridge models are available):
    - Both models must show non-negative EV (logistic EV ≥ 0
      AND ridge ≥ per-race fair share)
    - At least one model must show ≥ 5% edge
      - Logistic: ev_per_2 ≥ 0.05  (on a $2 win bet, after takeout)
      - Ridge:    ridge_value ≥ 1.05 × (18.30 / N)
                  (where N is field size; predictions normalized per race
                  to sum to $18.30)

When only logistic is available, falls back to the old ev_roi > 0 trigger.
"""

import os, json, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])


# ── Defaults (overridden by fitted_params.json if present) ──────────
DEFAULT_STD = 5.0
DEFAULT_DF = 1.0
DEFAULT_WEIGHTS = {'claude': 0.35, 'gemini': 0.30, 'ml': 0.35}

# Minimum edge (probability units) to consider a bet
MIN_EDGE = 0.065

# Win pool takeout rate (Keeneland ~17%, varies by track)
WIN_TAKEOUT = 0.17

# ML odds thresholds for bet type
#   <= 4.5 (9/2):  Win only
#   5 to 9:        Win + Place
#   >= 10:         Win + Place + Show
ML_WIN_ONLY_MAX = 4.5
ML_WP_MAX = 9.0

# ── Ridge value target ──────────────────────────────────────────────
# Target = 4*win_paid + 2*place_paid + 1*show_paid (raw dollars).
#
# Expected sum of target across a whole race (constant, independent of
# field size): ~$18.30. Derivation — on a $2-base ticket in a pari-mutuel
# pool with 17% takeout, the weighted sum of payouts across all three
# board-hitters averages:
#   4 * win_paid   -> 6.64  (one winner, 83% of $8 fair return)
#   2 * place_paid -> 6.64  (one 2nd, same math via place pool)
#   1 * show_paid  -> 4.98  (one 3rd, same math via show pool)
#   total          -> 18.26
#
# We normalize every race's ridge predictions to sum to this constant,
# so break-even per horse = RIDGE_RACE_TOTAL / N, and edge threshold
# scales with field size automatically (8-horse field: $2.29 break-even,
# 12-horse field: $1.52 break-even).
RIDGE_RACE_TOTAL = 18.3
RIDGE_MIN_MULTIPLIER = 1.00   # per-horse qualification floor = fair share
RIDGE_EDGE_MULTIPLIER = 1.05  # 5% over per-horse fair share

# Small positive floor when normalizing a race (avoids div-by-zero and
# stops a single negative prediction from inverting everything).
RIDGE_FLOOR = 0.5

# Qualification floor (each horse must clear BOTH mins to be considered).
# Edge threshold (at least one model must clear its edge threshold to fire).
# Logistic defaults: non-negative EV floor, $0.05 edge.
# Ridge thresholds are now computed per-race from RIDGE_RACE_TOTAL / N.
LOGISTIC_MIN = 0.00           # $ per $2 (ev_per_2 floor)
LOGISTIC_EDGE_THRESHOLD = 0.05  # $ per $2


# =============================================================================
# 1. LOADING
# =============================================================================

def load_fitted_params():
    """Load fitted_params.json from DATA_PATH. Returns dict or None."""
    fp = DATA_PATH / 'fitted_params.json'
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return None


def load_ai_picks(date_str, track):
    """Load AI source CSVs for one date/track."""
    picks_dir = DATA_PATH / 'ai_picks' / date_str / track
    if not picks_dir.exists():
        # Try flat layout
        picks_dir = DATA_PATH / 'ai_picks' / date_str

    frames = {}
    for csv_path in picks_dir.glob('*.csv'):
        name = csv_path.stem  # 'claude', 'gemini', 'openai'
        if name in ('predictions', 'bet_log', 'backtest_results'):
            continue
        df = pd.read_csv(csv_path)

        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if 'beyer' in cl:
                col_map[c] = f'{name}_beyer'
            elif 'closing' in cl:
                col_map[c] = f'{name}_closing_prob'
            elif 'win_prob' in cl or 'prob' in cl:
                col_map[c] = f'{name}_win_prob'
        df = df.rename(columns=col_map)

        df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
        df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').astype('Int64')

        keep = ['race', 'horse_number']
        for col in [f'{name}_beyer', f'{name}_win_prob', f'{name}_closing_prob']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                keep.append(col)

        frames[name] = df[keep]

    if not frames:
        raise FileNotFoundError(f"No AI pick CSVs in {picks_dir}")

    names = list(frames.keys())
    merged = frames[names[0]]
    for n in names[1:]:
        merged = merged.merge(frames[n], on=['race', 'horse_number'], how='outer')

    merged['date'] = date_str
    merged['track'] = track
    return merged


def load_upc_entries(track):
    """Load upcoming entries with ML odds and running status."""
    upc_dir = DATA_PATH / 'upc_entries'
    # Try exact name, then with hyphens/underscores
    candidates = [
        upc_dir / f'{track}.csv',
        upc_dir / f'{track.replace("-", "_")}.csv',
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Normalize column names
            if 'horse' in df.columns and 'horse_number' not in df.columns:
                df['horse_number'] = pd.to_numeric(df['horse'], errors='coerce').astype('Int64')
            if 'race' in df.columns:
                df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
            return df
    return None


def _parse_ml_odds(odds_str):
    """Parse fractional/decimal ML odds to implied probability."""
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '':
            return np.nan
        s = str(odds_str).strip()
        if '/' in s:
            num, den = map(float, s.split('/'))
            return den / (num + den)
        val = float(s)
        if val >= 1:
            return 1 / (val + 1)
        return val
    except (ValueError, ZeroDivisionError):
        return np.nan


def _ml_odds_to_decimal(odds_str):
    """Parse fractional ML odds to decimal odds (e.g. '9/2' → 4.5)."""
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '':
            return np.nan
        s = str(odds_str).strip()
        if '/' in s:
            num, den = map(float, s.split('/'))
            return num / den
        return float(s)
    except (ValueError, ZeroDivisionError):
        return np.nan


# =============================================================================
# 2. PROBABILITY ENGINE (mirrors backtest_ai.py logic)
# =============================================================================

def sim_beyer_to_prob(df, beyer_col, std_dev, degrees_of_freedom, num_sims, seed=42):
    """T-dist MC sim per race. Returns Series of win probs."""
    result = pd.Series(np.nan, index=df.index)
    rng = np.random.RandomState(seed)
    t_dist = stats.t(df=degrees_of_freedom)

    for _, race_df in df.groupby('race'):
        idx = race_df.index
        n = len(idx)
        vals = race_df[beyer_col].values.copy()
        if n <= 1:
            result.loc[idx] = 1.0
            continue
        mean_b = np.nanmean(vals)
        if np.isnan(mean_b):
            continue
        vals = np.where(np.isnan(vals), mean_b, vals)
        noise = t_dist.rvs(size=(n, num_sims), random_state=rng) * std_dev
        winners = np.argmax(vals.reshape(n, 1) + noise, axis=0)
        counts = np.bincount(winners, minlength=n)
        probs = np.maximum(counts / num_sims, 1e-6)
        result.loc[idx] = probs / probs.sum()

    return result


def build_ensemble(df, std_dev, degrees_of_freedom, num_sims, weights,
                   ml_regularization=0.25):
    """Build 4 independent probability sources and blend into ensemble_prob.

    Sources:
      - claude_beyer: t-dist MC sim, ML-regularized
      - claude_prob:  direct LLM win probability
      - gemini_beyer: t-dist MC sim, ML-regularized
      - gemini_prob:  direct LLM win probability

    Each gets its own weight from fitted_params.json.
    Fallbacks: beyer backs up prob if missing, and vice versa.
    ML only regularizes the beyer sims.
    """
    AI_PROVIDERS = ['claude', 'gemini']

    df = df.copy()
    available = []

    has_ml = 'ml_win_prob_norm' in df.columns and df['ml_win_prob_norm'].notna().any()

    for provider in AI_PROVIDERS:
        bc = f'{provider}_beyer'
        pc = f'{provider}_win_prob'
        sim_col = f'{provider}_sim_prob'
        reg_col = f'{provider}_reg_sim'
        beyer_source = f'{provider}_beyer'
        prob_source = f'{provider}_prob'

        # Beyer path: sim → regularize
        if bc in df.columns and df[bc].notna().any():
            df[sim_col] = sim_beyer_to_prob(df, bc, std_dev, degrees_of_freedom, num_sims)
        else:
            df[sim_col] = np.nan

        if has_ml and df[sim_col].notna().any():
            ml_fill = df['ml_win_prob_norm'].fillna(df[sim_col])
            df[reg_col] = (1 - ml_regularization) * df[sim_col] + ml_regularization * ml_fill
        else:
            df[reg_col] = df[sim_col]

        # Beyer source (fallback to direct prob if beyer missing)
        beyer_col = f'{beyer_source}_src'
        df[beyer_col] = df[reg_col]
        beyer_missing = df[beyer_col].isna()
        if pc in df.columns and beyer_missing.any():
            df.loc[beyer_missing, beyer_col] = df.loc[beyer_missing, pc]
        if df[beyer_col].notna().any():
            available.append(beyer_source)

        # Prob source (fallback to reg sim if direct prob missing)
        prob_col = f'{prob_source}_src'
        if pc in df.columns:
            df[prob_col] = df[pc]
        else:
            df[prob_col] = np.nan
        prob_missing = df[prob_col].isna()
        if prob_missing.any() and df[reg_col].notna().any():
            df.loc[prob_missing, prob_col] = df.loc[prob_missing, reg_col]
        if df[prob_col].notna().any():
            available.append(prob_source)

    # Logit-space weighted ensemble
    if not available:
        df['ensemble_prob'] = np.nan
        return df

    w = {s: weights.get(s, 0) for s in available}
    tw = sum(w.values())
    if tw == 0:
        tw = len(w)
        w = {s: 1.0 / tw for s in w}
    else:
        w = {s: v / tw for s, v in w.items()}

    clip_lo, clip_hi = 1e-4, 1 - 1e-4
    logit_sum = np.zeros(len(df))
    for s, wt in w.items():
        p = df[f'{s}_src'].fillna(0).clip(clip_lo, clip_hi)
        logit_sum += np.log(p / (1 - p)) * wt

    df['ensemble_prob'] = 1.0 / (1.0 + np.exp(-logit_sum))

    # Normalize within each race
    for _, race_df in df.groupby('race'):
        idx = race_df.index
        total = df.loc[idx, 'ensemble_prob'].sum()
        if total > 0:
            df.loc[idx, 'ensemble_prob'] /= total

    return df


# =============================================================================
# 3. EXPERT ADJUSTMENTS
# =============================================================================

def load_expert_ratings():
    """Load expert_ratings.json."""
    path = DATA_PATH / 'expert_ratings.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def apply_expert_adjustments(df, date_str, track):
    """
    Apply expert logit deltas to ensemble_prob, weighted by Elo rating.

    For each expert with picks for this date/track:
      1. Load their logit deltas from expert_picks/{date}/{track}/{expert}.csv
      2. Look up their Elo weight from expert_ratings.json
      3. Apply: logit(ensemble) += weight * logit_delta per horse
      4. Renormalize per race

    Experts without ratings get a small default weight based on INITIAL_RATING
    from expert_elo.py (effectively near-zero influence).
    """
    expert_dir = DATA_PATH / 'expert_picks' / date_str / track
    if not expert_dir.exists():
        return df

    csv_files = sorted(expert_dir.glob('*.csv'))
    if not csv_files:
        return df

    ratings = load_expert_ratings()

    # Derive weights: (rating - 1500) / scale_factor
    # Scale so that a rating of 1535 (proven sharp) gives weight ~0.035
    # and 1510 (mildly useful) gives ~0.01
    BASELINE = 1500
    SCALE = 1000  # rating points per unit of logit influence

    print(f"\nApplying expert adjustments:")

    df = df.copy()
    clip = 1e-4

    # Convert ensemble to logit space
    ens_probs = df['ensemble_prob'].clip(clip, 1 - clip)
    logits = np.log(ens_probs / (1 - ens_probs)).values

    any_applied = False
    for csv_path in csv_files:
        expert_name = csv_path.stem
        expert_df = pd.read_csv(csv_path)

        if expert_df.empty:
            continue

        # Get weight from rating
        if expert_name in ratings:
            rating = ratings[expert_name]['rating']
        else:
            rating = BASELINE  # unknown expert, neutral weight

        weight = (rating - BASELINE) / SCALE
        if abs(weight) < 0.0001:
            print(f"  {expert_name:<22s} rating={rating:.0f}  weight≈0, skipping")
            continue

        # Build delta map: {(race, horse_number): logit_delta}
        n_applied = 0
        for _, row in expert_df.iterrows():
            race = int(row['race'])
            hn = int(row['horse_number'])
            delta = float(row['logit_delta'])

            mask = (df['race'] == race) & (df['horse_number'] == hn)
            idx = df.index[mask]
            if len(idx) > 0:
                logits[idx[0]] += weight * delta
                n_applied += 1

        print(f"  {expert_name:<22s} rating={rating:.1f}  weight={weight:+.4f}  "
              f"adjusted {n_applied} horses")
        any_applied = True

    if not any_applied:
        print("  No expert adjustments applied")
        return df

    # Convert back from logit
    df['ensemble_prob'] = 1.0 / (1.0 + np.exp(-logits))

    # Renormalize per race
    for _, race_df in df.groupby('race'):
        idx = race_df.index
        total = df.loc[idx, 'ensemble_prob'].sum()
        if total > 0:
            df.loc[idx, 'ensemble_prob'] /= total

    return df


# =============================================================================
# 4. EDGE CALCULATION
# =============================================================================

def calculate_edges(df, takeout=WIN_TAKEOUT):
    """
    Compute edge and expected value accounting for track takeout.

    For a $2 win bet on a horse at ML odds of X/1:
      - Payout if win: $2 * (1 + X) but pool has takeout
      - Approximate payout: $2 / ml_prob * (1 - takeout)
      - EV per $2 = 2 * (ensemble_prob / ml_prob) * (1 - takeout) - 2
      - ROI = (ensemble_prob / ml_prob) * (1 - takeout) - 1

    Positive ROI = positive expected value bet.
    Breakeven requires ensemble_prob > ml_prob / (1 - takeout).

    Kelly uses the takeout-adjusted odds.
    """
    df = df.copy()
    if 'ml_win_prob_norm' not in df.columns:
        df['edge'] = np.nan
        df['ev_roi'] = np.nan
        df['ev_per_2'] = np.nan
        df['kelly'] = np.nan
        return df

    proj = df['ensemble_prob']
    ml = df['ml_win_prob_norm']

    # Raw probability edge (before takeout)
    df['edge'] = proj - ml

    # Expected ROI on a $2 win bet (after takeout)
    # ROI = (our_prob / ml_prob) * (1 - takeout) - 1
    df['ev_roi'] = np.where(ml > 0, (proj / ml) * (1 - takeout) - 1, np.nan)

    # EV in dollars per $2 bet
    df['ev_per_2'] = np.where(ml > 0, 2 * df['ev_roi'], np.nan)

    # Kelly on takeout-adjusted odds
    # Effective decimal odds = (1 / ml) * (1 - takeout)
    # Kelly = (p * odds - 1) / (odds - 1)
    eff_odds = np.where(ml > 0, (1 / ml) * (1 - takeout), 0)
    df['kelly'] = np.where(
        eff_odds > 1,
        np.clip((proj * eff_odds - 1) / (eff_odds - 1), 0, 1),
        0
    )
    df['half_kelly'] = df['kelly'] / 2

    return df


# =============================================================================
# 5. BET RECOMMENDATIONS
# =============================================================================

def generate_bets(df,
                  logistic_min=LOGISTIC_MIN,
                  logistic_edge=LOGISTIC_EDGE_THRESHOLD,
                  ridge_edge_multiplier=RIDGE_EDGE_MULTIPLIER,
                  ridge_min_multiplier=1.0,
                  takeout=WIN_TAKEOUT):
    """
    Generate bet recommendations using dual-model agreement.

    Ridge thresholds are per-race (computed from field size + takeout),
    since ridge predictions are normalized within each race to sum to
    RIDGE_RACE_TOTAL ($18.30).

    The naive per-horse fair share is $18.30 / N — but that's what
    bettors actually get back (post-takeout). To actually beat the house,
    a horse must pay out gross-of-takeout, i.e. $18.30 / N / (1 - takeout).
    That's the threshold a ridge prediction has to clear just to be a
    break-even bet against the pool:

        break_even = (RIDGE_RACE_TOTAL / N) / (1 - takeout)
        ridge_min  = break_even * ridge_min_multiplier
        ridge_edge = break_even * ridge_edge_multiplier

    With takeout=0.17, this inflates per-horse thresholds by ~20%,
    making the ridge gate meaningfully tighter — without this, every
    race shows at least one "positive value" horse because the mean
    horse trivially clears its own mean.

    When ridge_value column is present (dual-model mode):
      - Qualification gate: both models must clear their floor
        (ev_per_2 >= logistic_min AND ridge_value >= per-race ridge_min)
      - Edge gate: at least one must clear its edge threshold
        (ev_per_2 >= logistic_edge OR ridge_value >= per-race ridge_edge)

    When only logistic is available:
      - Falls back to ev_per_2 >= logistic_min (plus edge check)

    Bet type by ML odds (unchanged):
      - ML <= 9/2:      W
      - ML 5/1 to 9/1:  WP
      - ML >= 10/1:     WPS

    Exotic bets (exacta/trifecta) have been dropped per request.

    Returns (bet_log DataFrame, summary dict per race).
    """
    bets = []
    race_summaries = {}

    has_ridge = 'ridge_value' in df.columns and df['ridge_value'].notna().any()
    has_logistic = 'ev_per_2' in df.columns and df['ev_per_2'].notna().any()

    # Gross-up factor: 1 / (1 - takeout). With 17% takeout, ≈ 1.2048.
    takeout_gross = 1.0 / max(1e-6, 1.0 - takeout)

    for race_num, race_df in df.groupby('race'):
        # Per-race break-even: raw fair share $18.30/N, grossed up for takeout
        n_horses = len(race_df)
        raw_fair = RIDGE_RACE_TOTAL / n_horses if n_horses > 0 else 0
        break_even = raw_fair * takeout_gross  # what ridge must predict to be +EV
        ridge_min = break_even * ridge_min_multiplier
        ridge_edge = break_even * ridge_edge_multiplier

        # Qualification mask
        if has_ridge and has_logistic:
            # Both models need to clear their floor
            base_qual = (
                race_df['ev_per_2'].notna()
                & race_df['ridge_value'].notna()
                & (race_df['ev_per_2'] >= logistic_min)
                & (race_df['ridge_value'] >= ridge_min)
            )
            # At least one must clear its edge threshold
            edge_qual = (
                (race_df['ev_per_2'] >= logistic_edge)
                | (race_df['ridge_value'] >= ridge_edge)
            )
            qualified = base_qual & edge_qual
        elif has_logistic:
            qualified = (
                race_df['ev_per_2'].notna()
                & (race_df['ev_per_2'] >= logistic_min)
                & (race_df['ev_per_2'] >= logistic_edge)
            )
        else:
            qualified = pd.Series(False, index=race_df.index)

        edge_horses = race_df[qualified].copy()

        # Sort: primary by logistic EV if available, else ridge value
        if has_ridge and has_logistic:
            # Combined score for ranking: normalize each relative to its
            # break-even and sum. Ridge side uses per-race break-even so
            # the ranking is comparable across races of different sizes.
            edge_horses['_rank_score'] = (
                edge_horses['ev_per_2'].fillna(0)
                + (edge_horses['ridge_value'].fillna(0) - break_even) / 2
            )
            edge_horses = edge_horses.sort_values('_rank_score', ascending=False)
        elif has_logistic:
            edge_horses = edge_horses.sort_values('ev_per_2', ascending=False)

        if len(edge_horses) == 0:
            race_summaries[race_num] = {'action': 'PASS', 'horses': []}
            continue

        race_bets = []
        for _, row in edge_horses.iterrows():
            hn = int(row['horse_number'])
            ml_dec = row.get('ml_decimal', np.nan)
            prob = row['ensemble_prob']
            kelly = row.get('half_kelly', 0)

            # Determine bet type by ML odds
            if pd.notna(ml_dec) and ml_dec <= ML_WIN_ONLY_MAX:
                bet_type = 'W'
            elif pd.notna(ml_dec) and ml_dec <= ML_WP_MAX:
                bet_type = 'WP'
            else:
                bet_type = 'WPS'

            bet_record = {
                'race': race_num,
                'horse_number': hn,
                'bet_type': bet_type,
                'ensemble_prob': round(prob, 4),
                'ml_prob': round(row.get('ml_win_prob_norm', np.nan), 4),
                'edge': round(row.get('edge', np.nan), 4),
                'ev_per_2': round(row.get('ev_per_2', np.nan), 2),
                'ev_roi': round(row.get('ev_roi', np.nan), 3),
                'kelly': round(kelly, 4),
                'ml_odds': row.get('ml_odds_raw', ''),
            }
            if has_ridge:
                rv = row.get('ridge_value', np.nan)
                bet_record['ridge_value'] = round(rv, 2) if pd.notna(rv) else np.nan
                bet_record['ridge_break_even'] = round(break_even, 2)
                bet_record['ridge_edge'] = round(rv - break_even, 2) if pd.notna(rv) else np.nan
                # Which model(s) triggered the edge?
                triggers = []
                if pd.notna(row.get('ev_per_2')) and row['ev_per_2'] >= logistic_edge:
                    triggers.append('LOG')
                if pd.notna(rv) and rv >= ridge_edge:
                    triggers.append('RDG')
                bet_record['triggered_by'] = '+'.join(triggers) if triggers else ''

            race_bets.append(bet_record)

        bets.extend(race_bets)
        race_summaries[race_num] = {
            'action': 'BET',
            'n_edge_horses': len(edge_horses),
            'horses': [int(h) for h in edge_horses['horse_number']],
        }

    bet_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    return bet_df, race_summaries


# =============================================================================
# 6. REPORTING
# =============================================================================

def print_race_report(df, race_num, bet_df=None):
    """Print formatted report for one race."""
    race = df[df['race'] == race_num].sort_values('ensemble_prob', ascending=False)
    if race.empty:
        return

    has_ridge = 'ridge_value' in race.columns and race['ridge_value'].notna().any()

    print(f"\n{'='*80}")
    print(f"RACE {race_num}")
    print(f"{'='*80}")

    # Build header
    src_cols = [c for c in race.columns if c.endswith('_beyer') and not c.startswith('ensemble')]
    header = f"{'#':>3} "
    for col in src_cols:
        header += f"{col.replace('_beyer','')[:6]:>6} "
    header += f"{'Prob':>7} "
    if 'ml_win_prob_norm' in race.columns:
        header += f"{'ML%':>6} {'Edge':>7} {'EV$2':>6} "
    if has_ridge:
        header += f"{'Ridge$':>7} "
    header += f"{'½K':>5} {'ML':>6}"
    print(header)
    print("-" * len(header))

    for _, row in race.iterrows():
        hn = int(row['horse_number'])
        line = f"{hn:>3} "
        for col in src_cols:
            v = row.get(col, np.nan)
            line += f"{v:>6.0f} " if pd.notna(v) else f"{'—':>6} "
        line += f"{row['ensemble_prob']:>6.1%} "
        if 'ml_win_prob_norm' in race.columns:
            ml = row.get('ml_win_prob_norm', np.nan)
            edge = row.get('edge', np.nan)
            ev2 = row.get('ev_per_2', np.nan)
            line += f"{ml:>5.1%} " if pd.notna(ml) else f"{'—':>5} "
            line += f"{edge:>+6.1%} " if pd.notna(edge) else f"{'—':>7} "
            line += f"{ev2:>+5.2f} " if pd.notna(ev2) else f"{'—':>6} "
        if has_ridge:
            rv = row.get('ridge_value', np.nan)
            line += f"{rv:>6.2f} " if pd.notna(rv) else f"{'—':>7} "
        hk = row.get('half_kelly', np.nan)
        ml_raw = row.get('ml_odds_raw', '')
        line += f"{hk:>5.3f}" if pd.notna(hk) else f"{'—':>5}"
        line += f" {ml_raw:>6}" if ml_raw else ""

        # Mark horses that qualify as bets — checked via bet_df membership
        # so the display exactly matches the actual bet log
        if bet_df is not None and not bet_df.empty:
            is_bet = (
                (bet_df['race'] == race_num)
                & (bet_df['horse_number'] == hn)
            ).any()
            if is_bet:
                line += "  ◀ BET"
        print(line)

    # Print bet recommendations for this race
    if bet_df is not None and not bet_df.empty:
        race_bets = bet_df[bet_df['race'] == race_num]
        if not race_bets.empty:
            print(f"\n  Bets:")
            for _, b in race_bets.iterrows():
                hn = str(b['horse_number'])
                parts = [f"    {b['bet_type']:<5s} #{hn:<4s}"]
                if pd.notna(b.get('ev_per_2')):
                    parts.append(f"LogEV=${b['ev_per_2']:+.2f}")
                if 'ridge_value' in b and pd.notna(b.get('ridge_value')):
                    parts.append(f"Ridge=${b['ridge_value']:.2f}")
                if 'triggered_by' in b and b.get('triggered_by'):
                    parts.append(f"[{b['triggered_by']}]")
                print("  ".join(parts))


def percentage_to_american_odds(p):
    """Convert probability to American odds string."""
    if pd.isna(p) or p <= 0 or p >= 1:
        return '—'
    if p >= 0.5:
        return f"{int(-100 * p / (1 - p)):+d}"
    return f"+{int(100 * (1 - p) / p)}"


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def run_predictions(date_str, track, num_sims=100000, use_regression=False):
    """Full forward-looking prediction pipeline."""

    # --- Load fitted params ---
    params = load_fitted_params()
    if params:
        std_dev = params['std_dev']
        degrees_of_freedom = params['degrees_of_freedom']
        ml_reg = params.get('ml_regularization', 0.25)
        weights = params['weights']
        print(f"Loaded fitted params (std={std_dev:.2f}, df={degrees_of_freedom:.2f}, "
              f"ml_reg={ml_reg:.2f})")
        print(f"  Weights: {weights}")
    else:
        std_dev = DEFAULT_STD
        degrees_of_freedom = DEFAULT_DF
        ml_reg = 0.25
        weights = DEFAULT_WEIGHTS
        print(f"No fitted_params.json found, using defaults")

    # --- Load AI picks ---
    print(f"\nLoading AI picks for {date_str} / {track}...")
    df = load_ai_picks(date_str, track)
    print(f"  {len(df)} horses across {df['race'].nunique()} races")

    # --- Load UPC entries (scratches + ML) ---
    upc = load_upc_entries(track)
    if upc is not None:
        print(f"Loading entries from upc_entries/{track}.csv...")

        # Filter to running horses
        if 'running' in upc.columns:
            scratched = upc[upc['running'] != 1]
            if len(scratched) > 0:
                scratch_set = set(zip(scratched['race'], scratched['horse_number']))
                before = len(df)
                df = df[~df.apply(lambda r: (r['race'], r['horse_number']) in scratch_set, axis=1)]
                print(f"  Removed {before - len(df)} scratches")
            else:
                print(f"  No scratches")
        else:
            print(f"  No 'running' column — assuming all horses running")

        # Merge ML odds
        if 'ml_odds' in upc.columns:
            upc_ml = upc[['race', 'horse_number', 'ml_odds']].drop_duplicates()

            # Diagnostic: check for merge mismatches
            ai_keys = set(zip(df['race'], df['horse_number']))
            upc_keys = set(zip(upc_ml['race'], upc_ml['horse_number']))
            ai_only = ai_keys - upc_keys
            if ai_only:
                print(f"  WARNING: {len(ai_only)} AI horses not in UPC entries (no ML odds):")
                for r, h in sorted(ai_only):
                    print(f"    R{r} #{h}")

            df = df.merge(upc_ml, on=['race', 'horse_number'], how='left')
            df['ml_odds_raw'] = df['ml_odds']
            df['ml_decimal'] = df['ml_odds'].apply(_ml_odds_to_decimal)
            df['ml_win_prob'] = df['ml_odds'].apply(_parse_ml_odds)

            # Normalize ML within each race
            df['ml_win_prob_norm'] = np.nan
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                raw = race_df['ml_win_prob'].values
                valid = ~np.isnan(raw)
                if valid.any():
                    probs = raw[valid]
                    probs = probs / probs.sum()
                    df.loc[idx[valid], 'ml_win_prob_norm'] = probs
            print(f"  ML odds loaded for {df['ml_win_prob'].notna().sum()} horses")
    else:
        print(f"  No upc_entries/{track}.csv found — running without ML odds")

    # --- Build probabilities ---
    if use_regression:
        print(f"\nBuilding regression probabilities...")
        df = build_regression_probs(df, date_str, track)
    else:
        print(f"\nBuilding ensemble (sims={num_sims:,})...")
        df = build_ensemble(df, std_dev, degrees_of_freedom, num_sims, weights,
                            ml_regularization=ml_reg)
        # Apply expert adjustments (only in ensemble mode — regression includes them)
        df = apply_expert_adjustments(df, date_str, track)

    # --- Edge calculation ---
    df = calculate_edges(df)

    # --- Ridge value predictions (additional signal, runs alongside logistic) ---
    df = apply_ridge_value_model(df, date_str, track)

    # --- Bet recommendations ---
    bet_df, race_summaries = generate_bets(
        df,
        logistic_min=LOGISTIC_MIN,
        logistic_edge=LOGISTIC_EDGE_THRESHOLD,
        ridge_edge_multiplier=RIDGE_EDGE_MULTIPLIER,
        ridge_min_multiplier=RIDGE_MIN_MULTIPLIER,
        takeout=WIN_TAKEOUT,
    )

    # --- Print reports ---
    for race_num in sorted(df['race'].unique()):
        print_race_report(df, race_num, bet_df)

    # --- Summary ---
    n_bet_races = sum(1 for v in race_summaries.values() if v['action'] == 'BET')
    n_pass_races = sum(1 for v in race_summaries.values() if v['action'] == 'PASS')

    print(f"\n{'='*80}")
    print(f"SUMMARY: {date_str} / {track}")
    print(f"{'='*80}")
    print(f"  Races: {df['race'].nunique()}")
    print(f"  Bet races: {n_bet_races}  |  Pass: {n_pass_races}")

    has_ridge = 'ridge_value' in df.columns and df['ridge_value'].notna().any()
    if has_ridge:
        # Show the range of per-race ridge break-evens across the card
        field_sizes = df.groupby('race').size()
        takeout_gross = 1.0 / max(1e-6, 1.0 - WIN_TAKEOUT)
        min_be = (RIDGE_RACE_TOTAL / field_sizes.max()) * takeout_gross
        max_be = (RIDGE_RACE_TOTAL / field_sizes.min()) * takeout_gross
        print(f"\n  Bet trigger: logistic EV ≥ ${LOGISTIC_MIN:.2f} "
              f"AND ridge ≥ per-race break-even,")
        print(f"               plus at least one of "
              f"(logistic EV ≥ ${LOGISTIC_EDGE_THRESHOLD:.2f}) or "
              f"(ridge ≥ {RIDGE_EDGE_MULTIPLIER:.2f}× break-even)")
        print(f"  Ridge break-even this card: ${min_be:.2f}–${max_be:.2f} "
              f"(${RIDGE_RACE_TOTAL:.2f}/N grossed up for {WIN_TAKEOUT:.0%} takeout)")
    else:
        print(f"\n  Bet trigger: logistic EV ≥ ${LOGISTIC_MIN:.2f} "
              f"and ≥ ${LOGISTIC_EDGE_THRESHOLD:.2f} edge "
              f"(ridge model not available)")

    if not bet_df.empty:
        print(f"  Total bets: {len(bet_df)}")
        if 'triggered_by' in bet_df.columns:
            trig_counts = bet_df['triggered_by'].value_counts()
            for trig, n in trig_counts.items():
                print(f"    Triggered by {trig:<8s} {n}")

        print(f"\n  Bet Log:")
        for _, b in bet_df.iterrows():
            parts = [
                f"    R{int(b['race']):>2}",
                f"{b['bet_type']:<4s}",
                f"#{str(b['horse_number']):<4s}",
                f"prob={b['ensemble_prob']:.1%}",
            ]
            if pd.notna(b.get('ev_per_2')):
                parts.append(f"LogEV=${b['ev_per_2']:+.2f}")
            if 'ridge_value' in b and pd.notna(b.get('ridge_value')):
                parts.append(f"Ridge=${b['ridge_value']:.2f}")
            if 'triggered_by' in b and b.get('triggered_by'):
                parts.append(f"[{b['triggered_by']}]")
            print("  ".join(parts))

    # --- Save outputs ---
    out_dir = DATA_PATH / 'predictions' / date_str / track
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / 'predictions.csv'
    df.to_csv(pred_path, index=False, encoding='utf-8')
    print(f"\n  Predictions → {pred_path}")

    if not bet_df.empty:
        bet_path = out_dir / 'bet_log.csv'
        bet_df.to_csv(bet_path, index=False, encoding='utf-8')
        print(f"  Bet log     → {bet_path}")

    return df, bet_df


def apply_ridge_value_model(df, date_str, track):
    """
    Apply ridge_value_model.json (from regression_analysis.py) as a second
    independent signal. Does NOT replace ensemble_prob — adds a 'ridge_value'
    column that predicts expected dollar return on the 4/2/1 weighted target.

    If the ridge model isn't present, returns df unchanged (dual-model bet
    logic will fall back to logistic-only).
    """
    model_path = DATA_PATH / 'ridge_value_model.json'
    if not model_path.exists():
        print(f"\n  No ridge_value_model.json — using logistic EV only")
        return df

    with open(model_path) as f:
        model = json.load(f)

    feature_cols = model['feature_cols']
    feature_is_prob = model.get('feature_is_prob', [True] * len(feature_cols))
    coefs = np.array(model['coefficients'])
    intercept = model['intercept']

    print(f"\n  Loading ridge value model (α={model.get('regularization_alpha')}, "
          f"CV ρ={model.get('cv_spearman')})")

    df = df.copy()

    # Same feature-construction logic as build_regression_probs — we need
    # human_prob, beyer sims, inv_field_size, ml_x_fieldsize if the ridge
    # model uses them.
    df = _ensure_regression_features(df, date_str, track)

    col_map = {
        'ml_prob_norm': 'ml_win_prob_norm',
        'claude_prob': 'claude_win_prob',
        'gemini_prob': 'gemini_win_prob',
        'openai_prob': 'openai_win_prob',
        'claude_beyer_sim': 'claude_beyer_sim',
        'gemini_beyer_sim': 'gemini_beyer_sim',
        'openai_beyer_sim': 'openai_beyer_sim',
        'openai_closing': 'openai_closing_prob',
        'claude_closing': 'claude_closing_prob',
        'gemini_closing': 'gemini_closing_prob',
        'human_prob': 'human_prob',
        'inv_field_size': 'inv_field_size',
        'ml_x_fieldsize': 'ml_x_fieldsize',
    }

    # Normalize probability features per race (to match training)
    for fc, is_prob in zip(feature_cols, feature_is_prob):
        if not is_prob:
            continue
        df_col = col_map.get(fc, fc)
        if df_col in df.columns:
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                vals = df.loc[idx, df_col].values
                valid = ~np.isnan(vals)
                if valid.any() and np.nansum(vals[valid]) > 0:
                    df.loc[idx, df_col] = np.where(
                        valid, vals / np.nansum(vals), np.nan
                    )

    # Build feature matrix (same transform rules as training)
    X_cols = []
    for fc in feature_cols:
        df_col = col_map.get(fc, fc)
        if df_col not in df.columns:
            print(f"    WARNING: ridge feature '{fc}' not found, using 0")
            df[df_col] = 0
        X_cols.append(df_col)

    X_raw = df[X_cols].fillna(0).values
    clip_lo, clip_hi = 1e-4, 1 - 1e-4

    input_space = model.get('input_space', 'mixed')
    if input_space == 'mixed':
        X = np.zeros_like(X_raw, dtype=float)
        for i, is_prob in enumerate(feature_is_prob):
            if is_prob:
                clipped = np.clip(X_raw[:, i], clip_lo, clip_hi)
                X[:, i] = np.log(clipped / (1 - clipped))
            else:
                X[:, i] = X_raw[:, i]
    elif input_space == 'logit':
        X_clipped = np.clip(X_raw, clip_lo, clip_hi)
        X = np.log(X_clipped / (1 - X_clipped))
    else:
        X = X_raw

    # Linear prediction (ridge is linear regression)
    df['ridge_value_raw'] = X @ coefs + intercept

    # If the model was trained on log1p(dollars), invert back to dollar space
    target_transform = model.get('target_transform')
    if target_transform == 'log1p':
        df['ridge_value_raw'] = np.expm1(df['ridge_value_raw'])

    # Normalize each race to sum to RIDGE_RACE_TOTAL (~$18.30). The raw
    # ridge output often doesn't sum to the theoretically-correct total
    # because the intercept is trained on the *average* horse; when a
    # race has more or fewer favorites than average, the raw sum drifts.
    # A proportional rescale fixes this without distorting the relative
    # ranking within the race.
    #
    # Horses with floor-level predictions (ridge says ~nothing) get a
    # RIDGE_FLOOR bump before the sum is computed, so a single negative
    # prediction can't make the race sum tiny (or zero) and blow up the
    # scale factor for everyone else.
    df['ridge_value'] = np.nan
    for race_num, race_df in df.groupby('race'):
        idx = race_df.index
        raw = df.loc[idx, 'ridge_value_raw'].values.astype(float)
        floored = np.where(np.isnan(raw), RIDGE_FLOOR,
                           np.maximum(raw, RIDGE_FLOOR))
        total = floored.sum()
        if total > 0:
            df.loc[idx, 'ridge_value'] = floored * (RIDGE_RACE_TOTAL / total)

    return df


def _ensure_regression_features(df, date_str, track):
    """
    Compute human_prob, beyer sims, inv_field_size, ml_x_fieldsize if they
    aren't already present. Idempotent — safe to call more than once.

    This exists so apply_ridge_value_model() can run after either the
    ensemble path (which doesn't build these) or the regression path (which
    does). Either way we get the features the ridge model expects.
    """
    if 'human_prob' not in df.columns:
        df['human_pool'] = 0.0
        expert_dir = DATA_PATH / 'expert_picks' / date_str / track
        ratings_path = DATA_PATH / 'expert_ratings.json'
        ratings = {}
        if ratings_path.exists():
            with open(ratings_path) as f_r:
                ratings = json.load(f_r)

        if expert_dir.exists():
            for csv_path in expert_dir.glob('*.csv'):
                expert_name = csv_path.stem
                try:
                    edf = pd.read_csv(csv_path)
                except Exception:
                    continue
                if edf.empty:
                    continue

                elo = ratings.get(expert_name, {}).get('rating', 1500)
                elo_mult = max(0.1, (elo - 1350) / 150)

                for _, row in edf.iterrows():
                    race = int(row['race'])
                    hn = int(row['horse_number'])
                    delta = float(row['logit_delta'])
                    pool_amount = (delta / 0.7) * 2.0 * elo_mult
                    mask = (df['race'] == race) & (df['horse_number'] == hn)
                    df.loc[mask, 'human_pool'] += pool_amount

        df['human_prob'] = 0.0
        for _, race_df in df.groupby('race'):
            idx = race_df.index
            pool = df.loc[idx, 'human_pool'].values
            shifted = pool - pool.min() + 0.01
            total = shifted.sum()
            if total > 0:
                df.loc[idx, 'human_prob'] = shifted / total

    # Beyer sim columns
    for beyer_col, sim_col in [('claude_beyer', 'claude_beyer_sim'),
                                ('gemini_beyer', 'gemini_beyer_sim'),
                                ('openai_beyer', 'openai_beyer_sim')]:
        if sim_col in df.columns and df[sim_col].notna().any():
            continue
        df[sim_col] = np.nan
        if beyer_col in df.columns:
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                vals = df.loc[idx, beyer_col].values
                valid = ~np.isnan(vals)
                if valid.sum() >= 2:
                    centered = vals[valid] - np.mean(vals[valid])
                    exp_vals = np.exp(centered * 0.1)
                    probs = exp_vals / exp_vals.sum()
                    full = np.full(len(vals), np.nan)
                    full[valid] = probs
                    df.loc[idx, sim_col] = full

    # inv_field_size
    if 'inv_field_size' not in df.columns:
        df['inv_field_size'] = np.nan
        for _, race_df in df.groupby('race'):
            idx = race_df.index
            df.loc[idx, 'inv_field_size'] = 1.0 / len(race_df)

    # ml_x_fieldsize interaction
    if 'ml_x_fieldsize' not in df.columns and 'ml_win_prob_norm' in df.columns:
        clip_lo_fs, clip_hi_fs = 1e-4, 1 - 1e-4
        ml_logit = np.log(
            df['ml_win_prob_norm'].clip(clip_lo_fs, clip_hi_fs) /
            (1 - df['ml_win_prob_norm'].clip(clip_lo_fs, clip_hi_fs))
        )
        df['ml_x_fieldsize'] = ml_logit * df['inv_field_size']

    return df


def build_regression_probs(df, date_str, track):
    """
    Build probability estimates using the fitted logistic regression model
    (from regression_analysis.py). Uses averaged K-fold coefficients applied
    to ML, Claude/Gemini/OpenAI Prob, Claude/Gemini/OpenAI Beyer, Human Pool,
    plus interaction features.

    Returns df with 'ensemble_prob' column populated.
    """
    model_path = DATA_PATH / 'regression_model.json'
    if not model_path.exists():
        print(f"  No regression_model.json found — run regression_analysis.py first")
        return df

    with open(model_path) as f:
        model = json.load(f)

    feature_cols = model['feature_cols']
    coefs = np.array(model['coefficients'])
    intercept = model['intercept']

    print(f"  Regression model: {model['n_folds']}-fold avg, "
          f"CV ρ={model['cv_spearman']:.4f}")

    df = df.copy()
    df = _ensure_regression_features(df, date_str, track)

    col_map = {
        'ml_prob_norm': 'ml_win_prob_norm',
        'claude_prob': 'claude_win_prob',
        'gemini_prob': 'gemini_win_prob',
        'openai_prob': 'openai_win_prob',
        'claude_beyer_sim': 'claude_beyer_sim',
        'gemini_beyer_sim': 'gemini_beyer_sim',
        'openai_beyer_sim': 'openai_beyer_sim',
        'openai_closing': 'openai_closing_prob',
        'claude_closing': 'claude_closing_prob',
        'gemini_closing': 'gemini_closing_prob',
        'human_prob': 'human_prob',
        'inv_field_size': 'inv_field_size',
        'ml_x_fieldsize': 'ml_x_fieldsize',
    }

    feature_is_prob = model.get('feature_is_prob', [True] * len(feature_cols))

    # Normalize probability features per race
    for fc, is_prob in zip(feature_cols, feature_is_prob):
        if not is_prob:
            continue
        df_col = col_map.get(fc, fc)
        if df_col in df.columns:
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                vals = df.loc[idx, df_col].values
                valid = ~np.isnan(vals)
                if valid.any() and np.nansum(vals[valid]) > 0:
                    df.loc[idx, df_col] = np.where(valid, vals / np.nansum(vals), np.nan)

    # Build feature matrix
    X_cols = []
    for fc in feature_cols:
        df_col = col_map.get(fc, fc)
        if df_col in df.columns:
            X_cols.append(df_col)
        else:
            print(f"    WARNING: feature '{fc}' ({df_col}) not found, using 0")
            df[df_col] = 0
            X_cols.append(df_col)

    input_space = model.get('input_space', 'raw')
    clip_lo, clip_hi = 1e-4, 1 - 1e-4
    X_raw = df[X_cols].fillna(0).values

    if input_space == 'mixed':
        X = np.zeros_like(X_raw, dtype=float)
        for i, is_prob in enumerate(feature_is_prob):
            if is_prob:
                clipped = np.clip(X_raw[:, i], clip_lo, clip_hi)
                X[:, i] = np.log(clipped / (1 - clipped))
            else:
                X[:, i] = X_raw[:, i]
    elif input_space == 'logit':
        X_clipped = np.clip(X_raw, clip_lo, clip_hi)
        X = np.log(X_clipped / (1 - X_clipped))
    else:
        X = X_raw

    logits = X @ coefs + intercept
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    df['ensemble_prob'] = raw_probs

    # Normalize per race
    for _, race_df in df.groupby('race'):
        idx = race_df.index
        total = df.loc[idx, 'ensemble_prob'].sum()
        if total > 0:
            df.loc[idx, 'ensemble_prob'] /= total

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate race predictions + bet recommendations")
    parser.add_argument('--date', type=str, required=True, help='Date (e.g. 2026-04-08)')
    parser.add_argument('--track', type=str, required=True, help='Track (e.g. keeneland)')
    parser.add_argument('--sims', type=int, default=100000, help='MC simulations')
    parser.add_argument('--logistic-min', type=float, default=LOGISTIC_MIN,
                        help=f'Logistic EV floor $ per $2 (default {LOGISTIC_MIN}). '
                             f'Horse must clear this on logistic to qualify.')
    parser.add_argument('--logistic-edge', type=float, default=LOGISTIC_EDGE_THRESHOLD,
                        help=f'Logistic EV edge $ per $2 (default {LOGISTIC_EDGE_THRESHOLD}). '
                             f'At least one model must clear its edge.')
    parser.add_argument('--ridge-min-mult', type=float, default=RIDGE_MIN_MULTIPLIER,
                        help=f'Ridge qualification floor as multiple of per-race '
                             f'fair share (default {RIDGE_MIN_MULTIPLIER}, i.e. '
                             f'fair share). Horse must clear this on ridge to qualify.')
    parser.add_argument('--ridge-edge-mult', type=float, default=RIDGE_EDGE_MULTIPLIER,
                        help=f'Ridge edge threshold as multiple of per-race '
                             f'fair share (default {RIDGE_EDGE_MULTIPLIER}, '
                             f'i.e. 5%% over fair share)')
    parser.add_argument('--takeout', type=float, default=WIN_TAKEOUT,
                        help='Win pool takeout rate (default 0.17)')
    parser.add_argument('--regression', action='store_true',
                        help='Use regression model instead of ensemble for ensemble_prob')
    args = parser.parse_args()

    WIN_TAKEOUT = args.takeout
    LOGISTIC_MIN = args.logistic_min
    LOGISTIC_EDGE_THRESHOLD = args.logistic_edge
    RIDGE_MIN_MULTIPLIER = args.ridge_min_mult
    RIDGE_EDGE_MULTIPLIER = args.ridge_edge_mult

    run_predictions(args.date, args.track, num_sims=args.sims,
                    use_regression=args.regression)