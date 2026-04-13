"""
view_expert_history.py

View detailed per-race scoring history for an expert handicapper.
Shows the logit math: ML%, delta applied, adjusted%, finish, winner.

Usage:
    python view_expert_history.py kyle_herron
    python view_expert_history.py kyle_herron --date 2026-04-04
    python view_expert_history.py --list

Reads from:
    {DATA_PATH}/expert_ratings.json
    {DATA_PATH}/expert_picks/{date}/{track}/*.csv
    {DATA_PATH}/entries/*.csv
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

_TRACK_ALIASES = {
    'gulfstream-park': 'gulfstream', 'gulfstream park': 'gulfstream',
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


def _parse_ml_odds(odds_str):
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '':
            return np.nan
        s = str(odds_str).strip()
        if '/' in s:
            num, den = map(float, s.split('/'))
            return den / (num + den)
        val = float(s)
        return 1 / (val + 1) if val >= 1 else val
    except (ValueError, ZeroDivisionError):
        return np.nan


def load_all_entries():
    if not ENTRIES_DIR.exists():
        return pd.DataFrame()
    dfs = []
    for f in ENTRIES_DIR.glob('*.csv'):
        try: dfs.append(pd.read_csv(f))
        except: continue
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
    if 'horse_name' in df.columns:
        df['horse_name'] = df['horse_name'].astype(str).str.strip()
    return df


def apply_logit_deltas(ml_probs, deltas):
    clip = 1e-4
    p = ml_probs.clip(clip, 1 - clip)
    logits = np.log(p / (1 - p))
    adjusted = 1 / (1 + np.exp(-(logits + deltas.fillna(0))))
    total = adjusted.sum()
    if total > 0:
        adjusted /= total
    return adjusted


def show_expert(expert_name, filter_date=None):
    if not RATINGS_PATH.exists():
        print(f"No ratings file. Run: python expert_elo.py --force")
        return

    with open(RATINGS_PATH) as f:
        ratings = json.load(f)

    if expert_name not in ratings:
        print(f"Unknown expert: {expert_name}")
        print(f"Available: {', '.join(sorted(ratings.keys()))}")
        return

    data = ratings[expert_name]

    print(f"\n{'='*72}")
    print(f"EXPERT: {expert_name}")
    print(f"{'='*72}")
    print(f"  Initial: {data.get('initial_rating', '?')}  "
          f"Current: {data['rating']}  "
          f"Races: {data['races_scored']}  "
          f"Total improvement: {data['total_improvement']:+.4f}")

    if not data['history']:
        print("\n  No scored history yet.")
        return

    # Day-level summary
    history = data['history']
    if filter_date:
        history = [h for h in history if h['date'] == filter_date]
        if not history:
            print(f"\n  No history for {filter_date}")
            return

    print(f"\n  {'Date':<12s} {'Track':<12s} {'Races':>5s} {'ML_loss':>8s} "
          f"{'Exp_loss':>9s} {'Improv':>8s} {'K':>5s} {'Before':>7s} {'After':>7s}")
    print(f"  {'-'*12} {'-'*12} {'-'*5} {'-'*8} {'-'*9} {'-'*8} {'-'*5} {'-'*7} {'-'*7}")
    for h in history:
        print(f"  {h['date']:<12s} {h['track']:<12s} {h['races']:>5d} "
              f"{h['ml_loss']:>8.3f} {h['expert_loss']:>9.3f} "
              f"{h['improvement']:>+7.3f} {h['k_factor']:>5.1f} "
              f"{h['rating_before']:>7.1f} {h['rating_after']:>7.1f}")

    # Per-race detail
    entries_df = load_all_entries()
    if entries_df.empty:
        print("\n  (No entries loaded — can't show per-race detail)")
        return

    has_names = 'horse_name' in entries_df.columns

    for h in history:
        date_str, track = h['date'], h['track']
        expert_csv = EXPERT_PICKS_DIR / date_str / track / f"{expert_name}.csv"
        if not expert_csv.exists():
            continue

        expert_df = pd.read_csv(expert_csv)
        day_entries = entries_df[
            (entries_df['date'] == date_str) & (entries_df['track'] == track)
        ]
        if day_entries.empty:
            continue

        print(f"\n  --- {date_str} / {track} ---")
        name_col = f"  {'Name':<18s}" if has_names else ""
        print(f"  {'R':>3s} {'#':>3s} {'Delta':>6s} {'Sent':<10s} "
              f"{'ML%':>6s} {'Adj%':>6s} {'Fin':>4s}{name_col}")

        for race_num in sorted(expert_df['race'].unique()):
            race_df = day_entries[day_entries['race'] == race_num].copy()
            if race_df.empty:
                continue

            expert_race = expert_df[expert_df['race'] == race_num]

            ml_raw = race_df['ml_prob'].values
            if np.any(np.isnan(ml_raw)):
                continue
            ml_norm = ml_raw / ml_raw.sum()

            delta_map = {int(r['horse_number']): (r['logit_delta'], r.get('sentiment', ''))
                         for _, r in expert_race.iterrows()}
            deltas_series = race_df['horse_number'].map(
                lambda h: delta_map.get(int(h), (0.0, ''))[0])

            expert_probs = apply_logit_deltas(
                pd.Series(ml_norm, index=race_df.index), deltas_series)

            winner_hn = None
            if 'won' in race_df.columns and race_df['won'].sum() == 1:
                winner_hn = int(race_df.loc[race_df['won'] == 1, 'horse_number'].iloc[0])

            mentioned_hns = set(delta_map.keys())
            show_hns = mentioned_hns.copy()
            if winner_hn is not None:
                show_hns.add(winner_hn)

            for _, row in race_df.iterrows():
                hn = int(row['horse_number'])
                if hn not in show_hns:
                    continue

                idx = row.name
                ml_p = ml_norm[race_df.index.get_loc(idx)]
                adj_p = expert_probs.loc[idx]
                delta_val, sentiment = delta_map.get(hn, (0.0, ''))
                finish = int(row['finish']) if pd.notna(row.get('finish')) else ''
                marker = ' ◀WIN' if hn == winner_hn else ''
                if hn not in mentioned_hns and hn == winner_hn:
                    sentiment = '(missed)'

                name_str = ''
                if has_names:
                    hname = row.get('horse_name', '')
                    name_str = f"  {str(hname)[:18]:<18s}"

                print(f"  {race_num:>3d} {hn:>3d} {delta_val:>+5.2f} {sentiment:<10s} "
                      f"{ml_p:>5.1%} {adj_p:>5.1%} {str(finish):>4s}{name_str}{marker}")

            # Separator between races
            print()


def main():
    parser = argparse.ArgumentParser(description='View expert pick history with per-race detail')
    parser.add_argument('expert', nargs='?', help='Expert name (e.g. kyle_herron)')
    parser.add_argument('--date', type=str, help='Filter to one date (e.g. 2026-04-04)')
    parser.add_argument('--list', action='store_true', help='List all tracked experts')
    args = parser.parse_args()

    if args.list:
        if not RATINGS_PATH.exists():
            print("No ratings file yet.")
            return
        with open(RATINGS_PATH) as f:
            ratings = json.load(f)
        for name, data in sorted(ratings.items(), key=lambda x: -x[1]['rating']):
            n = data['races_scored']
            print(f"  {name:<25s} {data['rating']:>7.1f}  ({n} races)")
    elif args.expert:
        show_expert(args.expert, filter_date=args.date)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()