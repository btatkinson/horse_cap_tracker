"""
predict_races.py

Generate race predictions and bet recommendations from AI picks + fitted params.

Usage:
    python predict_races.py --date 2026-04-08 --track keeneland
    python predict_races.py --date 2026-04-08 --track keeneland --sims 200000

Reads from:
    {DATA_PATH}/ai_picks/{date}/{track}/claude.csv   (and gemini.csv)
    {DATA_PATH}/upc_entries/{track}.csv               (scratches + ML odds)
    {DATA_PATH}/fitted_params.json                    (from backtest)

Writes to:
    {DATA_PATH}/predictions/{date}/{track}/predictions.csv
    {DATA_PATH}/predictions/{date}/{track}/bet_log.csv
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
# 4. BET RECOMMENDATIONS
# =============================================================================

def generate_bets(df, min_ev_roi=0.0):
    """
    Generate bet recommendations per race.

    Logic:
      - Only bet horses with positive expected value (ev_roi > min_ev_roi)
        after accounting for track takeout
      - ML <= 9/2 (4.5):   Win only
      - ML 5/1 to 9/1:     Win + Place
      - ML >= 10/1:         Win + Place + Show
      - 2 value horses in same race:  exacta box
      - 3+ value horses:             trifecta box (top 3 by EV)

    Returns (bet_log DataFrame, summary dict per race).
    """
    bets = []
    race_summaries = {}

    for race_num, race_df in df.groupby('race'):
        # Filter to positive EV horses
        has_ev = race_df['ev_roi'].notna() & (race_df['ev_roi'] > min_ev_roi)
        edge_horses = race_df[has_ev].sort_values('ev_roi', ascending=False)

        if len(edge_horses) == 0:
            race_summaries[race_num] = {'action': 'PASS', 'horses': []}
            continue

        race_bets = []

        # --- Straight bets per horse ---
        for _, row in edge_horses.iterrows():
            hn = int(row['horse_number'])
            ml_dec = row.get('ml_decimal', np.nan)
            edge = row['edge']
            prob = row['ensemble_prob']
            kelly = row.get('half_kelly', 0)

            # Determine bet type by ML odds
            if pd.notna(ml_dec) and ml_dec <= ML_WIN_ONLY_MAX:
                bet_type = 'W'
            elif pd.notna(ml_dec) and ml_dec <= ML_WP_MAX:
                bet_type = 'WP'
            else:
                bet_type = 'WPS'

            race_bets.append({
                'race': race_num,
                'horse_number': hn,
                'bet_type': bet_type,
                'ensemble_prob': round(prob, 4),
                'ml_prob': round(row.get('ml_win_prob_norm', np.nan), 4),
                'edge': round(edge, 4),
                'ev_roi': round(row.get('ev_roi', np.nan), 3),
                'ev_per_2': round(row.get('ev_per_2', np.nan), 2),
                'kelly': round(kelly, 4),
                'ml_odds': row.get('ml_odds_raw', ''),
            })

        # --- Exotic bets ---
        if len(edge_horses) >= 2:
            top2 = edge_horses.head(2)
            h1, h2 = int(top2.iloc[0]['horse_number']), int(top2.iloc[1]['horse_number'])
            ex_prob = top2.iloc[0]['ensemble_prob'] * top2.iloc[1]['ensemble_prob'] * 2  # box
            race_bets.append({
                'race': race_num,
                'horse_number': f'{h1}/{h2}',
                'bet_type': 'EX BOX',
                'ensemble_prob': round(ex_prob, 4),
                'ml_prob': np.nan,
                'edge': np.nan,
                'pct_edge': np.nan,
                'kelly': np.nan,
                'ml_odds': '',
            })

        if len(edge_horses) >= 3:
            top3 = edge_horses.head(3)
            tri_nums = [int(top3.iloc[i]['horse_number']) for i in range(3)]
            # Box probability ≈ sum of all permutations
            tri_prob = 1
            for i in range(3):
                tri_prob *= top3.iloc[i]['ensemble_prob']
            tri_prob *= 6  # 3! permutations for box
            tri_prob = min(tri_prob, 1.0)
            race_bets.append({
                'race': race_num,
                'horse_number': '/'.join(str(h) for h in tri_nums),
                'bet_type': 'TRI BOX',
                'ensemble_prob': round(tri_prob, 4),
                'ml_prob': np.nan,
                'edge': np.nan,
                'pct_edge': np.nan,
                'kelly': np.nan,
                'ml_odds': '',
            })

        bets.extend(race_bets)
        race_summaries[race_num] = {
            'action': 'BET',
            'n_edge_horses': len(edge_horses),
            'horses': [int(h) for h in edge_horses['horse_number']],
        }

    bet_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    return bet_df, race_summaries


# =============================================================================
# 5. REPORTING
# =============================================================================

def print_race_report(df, race_num, bet_df=None):
    """Print formatted report for one race."""
    race = df[df['race'] == race_num].sort_values('ensemble_prob', ascending=False)
    if race.empty:
        return

    print(f"\n{'='*72}")
    print(f"RACE {race_num}")
    print(f"{'='*72}")

    # Build header
    src_cols = [c for c in race.columns if c.endswith('_beyer') and not c.startswith('ensemble')]
    header = f"{'#':>3} "
    for col in src_cols:
        header += f"{col.replace('_beyer','')[:6]:>6} "
    header += f"{'Prob':>7} "
    if 'ml_win_prob_norm' in race.columns:
        header += f"{'ML%':>6} {'Edge':>7} {'EV$2':>6} {'½K':>5} {'ML':>6}"
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
            hk = row.get('half_kelly', np.nan)
            ml_raw = row.get('ml_odds_raw', '')
            line += f"{ml:>5.1%} " if pd.notna(ml) else f"{'—':>5} "
            line += f"{edge:>+6.1%} " if pd.notna(edge) else f"{'—':>7} "
            line += f"{ev2:>+5.2f} " if pd.notna(ev2) else f"{'—':>6} "
            line += f"{hk:>5.3f}" if pd.notna(hk) else f"{'—':>5}"
            line += f" {ml_raw:>6}" if ml_raw else ""

        # Mark positive EV horses
        if pd.notna(row.get('ev_roi')) and row['ev_roi'] > 0:
            line += "  ◀ BET"
        print(line)

    # Print bet recommendations for this race
    if bet_df is not None and not bet_df.empty:
        race_bets = bet_df[bet_df['race'] == race_num]
        if not race_bets.empty:
            print(f"\n  Bets:")
            for _, b in race_bets.iterrows():
                hn = str(b['horse_number'])
                if pd.notna(b.get('ev_per_2')):
                    print(f"    {b['bet_type']:<10s} #{hn:<8s} "
                          f"EV=${b['ev_per_2']:+.2f}/bet  ROI={b['ev_roi']:+.0%}")
                else:
                    print(f"    {b['bet_type']:<10s} #{hn:<8s}")


def percentage_to_american_odds(p):
    """Convert probability to American odds string."""
    if pd.isna(p) or p <= 0 or p >= 1:
        return '—'
    if p >= 0.5:
        return f"{int(-100 * p / (1 - p)):+d}"
    return f"+{int(100 * (1 - p) / p)}"


# =============================================================================
# 6. MAIN PIPELINE
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

    # --- Bet recommendations ---
    bet_df, race_summaries = generate_bets(df)

    # --- Print reports ---
    for race_num in sorted(df['race'].unique()):
        print_race_report(df, race_num, bet_df)

    # --- Summary ---
    n_bet_races = sum(1 for v in race_summaries.values() if v['action'] == 'BET')
    n_pass_races = sum(1 for v in race_summaries.values() if v['action'] == 'PASS')

    print(f"\n{'='*72}")
    print(f"SUMMARY: {date_str} / {track}")
    print(f"{'='*72}")
    print(f"  Races: {df['race'].nunique()}")
    print(f"  Bet races: {n_bet_races}  |  Pass: {n_pass_races}")

    if not bet_df.empty:
        straight = bet_df[bet_df['bet_type'].isin(['W', 'WP', 'WPS'])]
        exotics = bet_df[~bet_df['bet_type'].isin(['W', 'WP', 'WPS'])]
        print(f"  Straight bets: {len(straight)}")
        print(f"  Exotic bets: {len(exotics)}")

        print(f"\n  Bet Log:")
        for _, b in bet_df.iterrows():
            ev_str = f"EV=${b['ev_per_2']:+.2f}" if pd.notna(b.get('ev_per_2')) else ""
            print(f"    R{int(b['race']):>2} {b['bet_type']:<10s} #{str(b['horse_number']):<8s} "
                  f"prob={b['ensemble_prob']:.1%}  {ev_str}")

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

def build_regression_probs(df, date_str, track):
    """
    Build probability estimates using the fitted regression model
    (from regression_analysis.py). Uses averaged K-fold logistic
    regression coefficients applied to: ML, Claude Prob, Gemini Prob,
    Claude Beyer, Gemini Beyer, Human Pool.

    Returns df with 'ensemble_prob' column populated.
    """
    import json

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

    # Build features that the regression expects
    df = df.copy()

    # ML prob norm should already exist
    # Claude/Gemini prob and beyer should already be loaded

    # Build human pool
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

            print(f"    Expert: {expert_name} (elo={elo:.0f}, mult={elo_mult:.2f})")

    # Normalize human pool to probability per race
    df['human_prob'] = 0.0
    for _, race_df in df.groupby('race'):
        idx = race_df.index
        pool = df.loc[idx, 'human_pool'].values
        shifted = pool - pool.min() + 0.01
        total = shifted.sum()
        if total > 0:
            df.loc[idx, 'human_prob'] = shifted / total

    # Map regression feature names to df columns
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

    # Convert beyers to within-race softmax probabilities
    for beyer_col, sim_col in [('claude_beyer', 'claude_beyer_sim'),
                                ('gemini_beyer', 'gemini_beyer_sim'),
                                ('openai_beyer', 'openai_beyer_sim')]:
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

    # Compute inv_field_size and interaction features
    df['inv_field_size'] = np.nan
    for _, race_df in df.groupby('race'):
        idx = race_df.index
        df.loc[idx, 'inv_field_size'] = 1.0 / len(race_df)

    # ML × 1/field_size interaction
    clip_lo_fs, clip_hi_fs = 1e-4, 1 - 1e-4
    if 'ml_win_prob_norm' in df.columns:
        ml_logit = np.log(
            df['ml_win_prob_norm'].clip(clip_lo_fs, clip_hi_fs) /
            (1 - df['ml_win_prob_norm'].clip(clip_lo_fs, clip_hi_fs))
        )
        df['ml_x_fieldsize'] = ml_logit * df['inv_field_size']

    # Normalize probability features per race (skip non-prob features)
    feature_is_prob = model.get('feature_is_prob',
                                [True] * len(feature_cols))  # default all prob

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

    # Transform: prob features → logit, raw features stay raw
    input_space = model.get('input_space', 'raw')
    clip_lo, clip_hi = 1e-4, 1 - 1e-4

    X_raw = df[X_cols].fillna(0).values

    if input_space == 'mixed':
        X = np.zeros_like(X_raw)
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

    # Apply logistic regression: P(win) = sigmoid(X @ coefs + intercept)
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
    parser.add_argument('--min-roi', type=float, default=0.0,
                        help='Min EV ROI to bet (default 0 = any positive EV)')
    parser.add_argument('--takeout', type=float, default=WIN_TAKEOUT,
                        help='Win pool takeout rate (default 0.17)')
    parser.add_argument('--regression', action='store_true',
                        help='Use regression model instead of ensemble')
    args = parser.parse_args()

    WIN_TAKEOUT = args.takeout
    run_predictions(args.date, args.track, num_sims=args.sims,
                    use_regression=args.regression)