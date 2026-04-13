"""
expert_elo.py

Elo rating system for expert handicappers. Replays all historical data
from initial ratings to produce current ratings + full history.

Every run scans expert_picks/ for all dates with matching entries results,
replays them chronologically from EXPERT_INITIAL_RATINGS, and writes
the final state to expert_ratings.json.

Usage:
    python expert_elo.py                     # full replay (fails if file exists)
    python expert_elo.py --force             # full replay, overwrite existing
    python expert_elo.py --report            # print current ratings table
    python expert_elo.py --weights           # print ensemble weights

Reads from:
    {DATA_PATH}/expert_picks/{date}/{track}/*.csv   (logit deltas)
    {DATA_PATH}/entries/*.csv                        (results + ML odds)

Writes to:
    {DATA_PATH}/expert_ratings.json
"""

import os, json, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
RATINGS_PATH = DATA_PATH / 'expert_ratings.json'
EXPERT_PICKS_DIR = DATA_PATH / 'expert_picks'
ENTRIES_DIR = DATA_PATH / 'entries'

# ── Elo parameters ──────────────────────────────────────────────────
INITIAL_RATING = 1500
K_INITIAL = 32
K_FLOOR = 8
K_DECAY_RATE = 0.01

# ── Per-expert initial ratings ──────────────────────────────────────
EXPERT_INITIAL_RATINGS = {
    ### known strong experts
    'kyle_herron':      1750,
    'joe_kristufek':    1700,
    'mike_beer':        1700,
    'mike_somich':      1700,
    'david_aragona':    1650,
    ## generic expert with good credentials
    'scott_hazelton':   1575,
    'jeremy_plonk':     1575,
    'tom_leach':        1575,
    'ellis_starr':      1575,
    'eric_solomon':     1575,
    'kim_nelson':       1575,
    'roberto_rodriguez':1575,
    'tony_calo':        1575,
    'kaitlin_benson':   1575,
    ## ron seemed more useful between two gulfstream experts
    'ron_nicoletti':    1600,
    'brian_nadeau':     1550,
    ## media people intialized as average
    'kyle_hampton':     1500,
    'kevin_kilroy':     1500,
    'samantha_perry':   1500,
    'ed_derosa':        1450,
}

# ── Track aliases ───────────────────────────────────────────────────
_TRACK_ALIASES = {
    'gulfstream-park': 'gulfstream-park', 'gulfstream park': 'gulfstream-park', 'gulfstream': 'gulfstream-park',
    'churchill-downs': 'churchill', 'churchill downs': 'churchill',
    'santa-anita': 'santa-anita', 'santa anita': 'santa-anita',
    'del-mar': 'del-mar', 'del mar': 'del-mar',
    'fair-grounds': 'fair-grounds', 'fair grounds': 'fair-grounds',
    'tampa-bay': 'tampa', 'tampa bay': 'tampa', 'tampa bay downs': 'tampa',
    'oaklawn-park': 'oaklawn', 'oaklawn park': 'oaklawn',
    'laurel-park': 'laurel', 'laurel park': 'laurel',
    'belmont-park': 'belmont', 'belmont park': 'belmont',
}
 
def _normalize_track(t):
    if pd.isna(t): return t
    t = str(t).strip().lower()
    return _TRACK_ALIASES.get(t, t)
 
 
# =============================================================================
# 1. PERSISTENCE
# =============================================================================
 
def load_ratings():
    if RATINGS_PATH.exists():
        with open(RATINGS_PATH) as f:
            return json.load(f)
    return {}
 
def save_ratings(ratings):
    RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RATINGS_PATH, 'w') as f:
        json.dump(ratings, f, indent=2)
 
 
# =============================================================================
# 2. DATA LOADING
# =============================================================================
 
def _parse_ml_odds(odds_str):
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
 
 
def load_all_entries():
    """Load all entries CSVs into one DataFrame."""
    if not ENTRIES_DIR.exists():
        return pd.DataFrame()
    dfs = []
    for f in ENTRIES_DIR.glob('*.csv'):
        try: dfs.append(pd.read_csv(f))
        except Exception: continue
    if not dfs:
        return pd.DataFrame()
 
    df = pd.concat(dfs, ignore_index=True)
    if 'running' in df.columns:
        df = df[df['running'] == 1].copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    if 'track' in df.columns:
        df['track'] = df['track'].apply(_normalize_track)
    if 'horse' in df.columns and 'horse_number' not in df.columns:
        df['horse_number'] = pd.to_numeric(df['horse'], errors='coerce')
    if 'finish' in df.columns:
        df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
        df['won'] = (df['finish'] == 1).astype(int)
    if 'ml_odds' in df.columns:
        df['ml_prob'] = df['ml_odds'].apply(_parse_ml_odds)
    for col in ['win_paid', 'place_paid', 'show_paid']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True),
                errors='coerce')
    return df
 
 
def discover_scoreable_days():
    """Find all (date, track) combos with both expert picks AND results."""
    if not EXPERT_PICKS_DIR.exists():
        return []
 
    entries = load_all_entries()
    if entries.empty or 'won' not in entries.columns:
        return []
 
    entry_keys = set()
    for _, row in entries[['date', 'track']].drop_duplicates().iterrows():
        if pd.notna(row['date']) and pd.notna(row['track']):
            entry_keys.add((row['date'], row['track']))
 
    days = []
    for date_dir in sorted(EXPERT_PICKS_DIR.iterdir()):
        if not date_dir.is_dir(): continue
        date_str = date_dir.name
        for track_dir in sorted(date_dir.iterdir()):
            if not track_dir.is_dir(): continue
            track = track_dir.name
            if list(track_dir.glob('*.csv')) and (date_str, track) in entry_keys:
                days.append((date_str, track))
 
    return sorted(days)
 
 
# =============================================================================
# 3. SCORING
# =============================================================================
 
def apply_logit_deltas(ml_probs, deltas):
    clip = 1e-4
    p = ml_probs.clip(clip, 1 - clip)
    logits = np.log(p / (1 - p))
    adjusted_logits = logits + deltas.fillna(0)
    adjusted_probs = 1 / (1 + np.exp(-adjusted_logits))
    total = adjusted_probs.sum()
    if total > 0:
        adjusted_probs /= total
    return adjusted_probs
 
 
def score_expert_race(race_df, expert_deltas):
    """Score one expert on one race. Returns (ml_loss, expert_loss, n) or None."""
    if 'won' not in race_df.columns or race_df['won'].sum() != 1:
        return None
 
    mentioned = {int(r['horse_number']) for _, r in expert_deltas.iterrows()}
    if not mentioned:
        return None
 
    ml_raw = race_df['ml_prob'].values
    if np.any(np.isnan(ml_raw)):
        return None
    ml_norm = ml_raw / ml_raw.sum()
 
    delta_map = {int(r['horse_number']): r['logit_delta']
                 for _, r in expert_deltas.iterrows()}
    deltas = race_df['horse_number'].map(lambda h: delta_map.get(int(h), 0.0))
    expert_probs = apply_logit_deltas(pd.Series(ml_norm, index=race_df.index), deltas)
 
    winner_idx = race_df[race_df['won'] == 1].index[0]
    ml_loss = -np.log(max(ml_norm[race_df.index.get_loc(winner_idx)], 1e-6))
    expert_loss = -np.log(max(expert_probs.loc[winner_idx], 1e-6))
    return ml_loss, expert_loss, len(mentioned)
 
 
def get_k_factor(n_races_scored):
    return max(K_FLOOR, K_INITIAL * np.exp(-K_DECAY_RATE * n_races_scored))
 
 
# =============================================================================
# 4. FULL REPLAY
# =============================================================================
 
def run_full_replay(verbose=True):
    """
    Replay all historical data from initial ratings, day by day.
 
    For each (date, track) with both expert picks and results:
      1. Record each expert's rating going INTO that day
      2. Score their picks against results
      3. Update rating
      4. Move to next day
 
    Writes final state to expert_ratings.json.
    """
    days = discover_scoreable_days()
 
    entries_df = load_all_entries()
 
    # Initialize all known experts at starting ratings
    ratings = {}
 
    if not days:
        if verbose:
            print("No scoreable days found (need expert_picks + entries with results).")
            print("Initializing from EXPERT_INITIAL_RATINGS only.")
        # Still write initial ratings so predict_races can use them
        for name, init_rating in EXPERT_INITIAL_RATINGS.items():
            ratings[name] = {
                'rating': init_rating,
                'initial_rating': init_rating,
                'races_scored': 0,
                'total_improvement': 0.0,
                'history': [],
            }
        save_ratings(ratings)
        if verbose:
            print(f"\nInitialized {len(ratings)} experts → {RATINGS_PATH}")
            for name, data in sorted(ratings.items(), key=lambda x: -x[1]['rating']):
                print(f"  {name:<25s} {data['rating']:.0f}")
        return ratings
 
    if verbose:
        print(f"Replaying {len(days)} day(s)...")
 
    for date_str, track in days:
        expert_dir = EXPERT_PICKS_DIR / date_str / track
        csv_files = sorted(expert_dir.glob('*.csv'))
 
        day_entries = entries_df[
            (entries_df['date'] == date_str) & (entries_df['track'] == track)
        ]
        if day_entries.empty:
            continue
 
        if verbose:
            print(f"\n  {date_str} / {track}:")
 
        for csv_path in csv_files:
            expert_name = csv_path.stem
            expert_df = pd.read_csv(csv_path)
            if expert_df.empty:
                continue
 
            # Initialize if first encounter
            if expert_name not in ratings:
                init_rating = EXPERT_INITIAL_RATINGS.get(expert_name, INITIAL_RATING)
                ratings[expert_name] = {
                    'rating': init_rating,
                    'initial_rating': init_rating,
                    'races_scored': 0,
                    'total_improvement': 0.0,
                    'history': [],
                }
 
            expert = ratings[expert_name]
            rating_before = expert['rating']
            n_before = expert['races_scored']
 
            # Score each race the expert covered
            race_results = []
            for race_num in sorted(expert_df['race'].unique()):
                race_df = day_entries[day_entries['race'] == race_num].copy()
                if race_df.empty:
                    continue
                expert_race = expert_df[expert_df['race'] == race_num]
                result = score_expert_race(race_df, expert_race)
                if result:
                    race_results.append(result[:2])
 
            if not race_results:
                continue
 
            # Day aggregates
            ml_total = sum(r[0] for r in race_results)
            expert_total = sum(r[1] for r in race_results)
            improvement = ml_total - expert_total
 
            # Elo update with expected score:
            # Higher-rated experts are expected to improve more on ML.
            # A neutral day (0 improvement) should pull toward 1500.
            # expected_improvement = n_races * (rating - 1500) / ELO_SCALE
            ELO_SCALE = 3000
            k = get_k_factor(n_before)
            expected = len(race_results) * (rating_before - INITIAL_RATING) / ELO_SCALE
            surprise = improvement - expected
            new_rating = rating_before + k * surprise
 
            # Record history
            expert['rating'] = round(new_rating, 1)
            expert['races_scored'] = n_before + len(race_results)
            expert['total_improvement'] = round(
                expert['total_improvement'] + improvement, 4)
            expert['history'].append({
                'date': date_str,
                'track': track,
                'races': len(race_results),
                'ml_loss': round(ml_total, 4),
                'expert_loss': round(expert_total, 4),
                'improvement': round(improvement, 4),
                'k_factor': round(k, 2),
                'rating_before': round(rating_before, 1),
                'rating_after': round(new_rating, 1),
            })
 
            if verbose:
                arrow = '▲' if improvement > 0 else '▼'
                print(f"    {expert_name:<22s} {len(race_results):>2d} races  "
                      f"imp={improvement:+.3f}  K={k:.1f}  "
                      f"{rating_before:.1f}→{new_rating:.1f} {arrow}")
 
    # Also ensure any EXPERT_INITIAL_RATINGS experts without picks get initialized
    for name, init_rating in EXPERT_INITIAL_RATINGS.items():
        if name not in ratings:
            ratings[name] = {
                'rating': init_rating,
                'initial_rating': init_rating,
                'races_scored': 0,
                'total_improvement': 0.0,
                'history': [],
            }
 
    save_ratings(ratings)
    if verbose:
        print(f"\nRatings saved to {RATINGS_PATH}")
        print(f"  {len(ratings)} experts tracked, "
              f"{sum(1 for r in ratings.values() if r['races_scored'] > 0)} with history")
 
    return ratings
 
 
# =============================================================================
# 5. REPORTING
# =============================================================================
 
def print_report():
    ratings = load_ratings()
    if not ratings:
        print("No expert ratings yet. Run: python expert_elo.py --force")
        return
 
    print(f"\n{'='*72}")
    print(f"EXPERT RATINGS")
    print(f"{'='*72}")
    print(f"  {'Expert':<25s} {'Rating':>7s} {'Init':>6s} {'Races':>6s} "
          f"{'Improv':>8s} {'K':>5s} {'imp/r':>7s}")
    print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*6} {'-'*8} {'-'*5} {'-'*7}")
 
    for name, data in sorted(ratings.items(), key=lambda x: -x[1]['rating']):
        n = data['races_scored']
        k = get_k_factor(n)
        imp_per_race = data['total_improvement'] / n if n > 0 else 0
        init = data.get('initial_rating', INITIAL_RATING)
        scored = '—' if n == 0 else str(n)
        print(f"  {name:<25s} {data['rating']:>7.1f} {init:>6.0f} {scored:>6s} "
              f"{data['total_improvement']:>+7.3f} {k:>5.1f} {imp_per_race:>+6.3f}")
 
    print(f"\n  > 1500: improves on ML  |  < 1500: faded  |  imp/r: log-loss gain per race")
 
 
def derive_weights(min_races=10):
    """Derive ensemble weights from Elo ratings. Negative = fade."""
    ratings = load_ratings()
    weights = {}
    for name, data in ratings.items():
        if data['races_scored'] < min_races:
            continue
        weights[name] = data['rating'] - INITIAL_RATING
    total_abs = sum(abs(v) for v in weights.values())
    if total_abs > 0:
        weights = {k: v / total_abs for k, v in weights.items()}
    return weights
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main():
    parser = argparse.ArgumentParser(description='Expert Elo rating system — full replay')
    parser.add_argument('--report', action='store_true', help='Print current ratings')
    parser.add_argument('--weights', action='store_true', help='Print ensemble weights')
    parser.add_argument('--force', action='store_true', help='Replay from scratch')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-day output')
    args = parser.parse_args()
 
    if args.report:
        print_report()
    elif args.weights:
        w = derive_weights()
        if w:
            print("Expert weights (min 10 races):")
            for name, weight in sorted(w.items(), key=lambda x: -x[1]):
                print(f"  {name:<25s} {weight:+.3f}")
        else:
            print("No experts with enough history yet.")
    else:
        if RATINGS_PATH.exists() and not args.force:
            print(f"Ratings file exists: {RATINGS_PATH}")
            print(f"  Use --force to replay from scratch")
            print(f"  Use --report to view current ratings")
            return
        run_full_replay(verbose=not args.quiet)
 
 
if __name__ == '__main__':
    main()