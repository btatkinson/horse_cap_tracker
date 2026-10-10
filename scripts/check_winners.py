"""
check_winners.py

Verify every race in walk_forward_summary.csv (or regression_predictions.csv)
has exactly one winner. Diagnoses several failure modes:

  1. Zero winners — race-finish parsing failed for that race
  2. Multiple winners — duplicate finish=1 rows or merge issues
  3. Race in entries but no predictions for the winner — merge dropped winner

Usage:
    python check_winners.py
    python check_winners.py --source cv  # check regression_predictions.csv instead
"""

import os, argparse
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
ENTRIES_DIR = DATA_PATH / 'entries'


def _parse_payout(val):
    if pd.isna(val):
        return None
    s = str(val).strip().replace('$', '').replace(',', '')
    if s == '' or s.lower() in ('nan', 'none', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_predictions_with_won(source='auto'):
    """Load predictions.csv-style data and merge `won` from entries if missing."""
    wf_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
    cv_path = DATA_PATH / 'regression_predictions.csv'

    if source == 'auto':
        path = wf_path if wf_path.exists() else cv_path
    elif source == 'walk_forward':
        path = wf_path
    else:
        path = cv_path

    if not path.exists():
        raise FileNotFoundError(f"{path}")

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}")

    # Merge `won` from entries if missing
    if 'won' not in df.columns or 'finish' not in df.columns:
        if not ENTRIES_DIR.exists():
            print(f"WARNING: ENTRIES_DIR {ENTRIES_DIR} doesn't exist")
            return df, None

        entry_dfs = []
        for f in ENTRIES_DIR.glob('*.csv'):
            try:
                entry_dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"  skip {f.name}: {e}")
        entries = pd.concat(entry_dfs, ignore_index=True) if entry_dfs else None

        if entries is None:
            return df, None

        # Normalize keys
        for col in ('date', 'track'):
            if col in entries.columns:
                entries[col] = entries[col].astype(str).str.strip().str.lower()
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        entries['race'] = pd.to_numeric(entries['race'], errors='coerce')
        if 'horse' in entries.columns and 'horse_number' not in entries.columns:
            entries['horse_number'] = pd.to_numeric(entries['horse'], errors='coerce')
        else:
            entries['horse_number'] = pd.to_numeric(
                entries.get('horse_number'), errors='coerce')
        df['race'] = pd.to_numeric(df['race'], errors='coerce')
        df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')

        if 'finish' in entries.columns:
            entries['_finish'] = pd.to_numeric(entries['finish'], errors='coerce')
            entries['won'] = (entries['_finish'] == 1).astype(int)

        bring = ['date', 'track', 'race', 'horse_number']
        for c in ('won', '_finish'):
            if c in entries.columns and c not in df.columns:
                bring.append(c)

        df = df.merge(
            entries[bring].drop_duplicates(['date', 'track', 'race', 'horse_number']),
            on=['date', 'track', 'race', 'horse_number'],
            how='left',
        )
        df['won'] = df.get('won', pd.Series([0]*len(df))).fillna(0).astype(int)

        return df, entries

    return df, None


def main():
    parser = argparse.ArgumentParser(description='Audit winner attribution')
    parser.add_argument('--source', type=str, default='auto',
                       choices=['auto', 'walk_forward', 'cv'])
    args = parser.parse_args()

    df, entries = load_predictions_with_won(args.source)

    if 'won' not in df.columns:
        print("ERROR: cannot determine winners — `won` column missing and entries unavailable")
        return

    print(f"\n{'='*72}")
    print(f"WINNER ATTRIBUTION AUDIT")
    print(f"{'='*72}")

    # Per-race winner counts
    race_winners = df.groupby(['date', 'track', 'race'])['won'].sum()
    race_horses = df.groupby(['date', 'track', 'race']).size()

    n_total = len(race_winners)
    n_one = (race_winners == 1).sum()
    n_zero = (race_winners == 0).sum()
    n_multi = (race_winners > 1).sum()

    print(f"  Total races: {n_total}")
    print(f"  ✓ Exactly one winner: {n_one} ({n_one/n_total:.1%})")
    if n_zero > 0:
        print(f"  ✗ ZERO winners:       {n_zero} ({n_zero/n_total:.1%})")
    if n_multi > 0:
        print(f"  ✗ MULTIPLE winners:   {n_multi} ({n_multi/n_total:.1%})")

    # Detail on zero-winner races (most concerning)
    if n_zero > 0:
        print(f"\n{'─'*72}")
        print(f"ZERO-WINNER RACES (model never sees a winner here, dragging calibration)")
        print(f"{'─'*72}")
        zero_races = race_winners[race_winners == 0]
        for (date, track, race) in zero_races.index[:30]:
            n_horses = race_horses.loc[(date, track, race)]
            # Check if this race exists in entries with a real winner
            entries_msg = ""
            if entries is not None:
                ent = entries[(entries['date'] == date) &
                             (entries['track'] == track) &
                             (entries['race'] == race)]
                if len(ent) == 0:
                    entries_msg = " | NOT in entries either"
                else:
                    winners_in_entries = ent[ent['_finish'] == 1] \
                        if '_finish' in ent.columns else pd.DataFrame()
                    if len(winners_in_entries) == 0:
                        entries_msg = f" | entries has {len(ent)} horses but no finish=1"
                    else:
                        w_horses = winners_in_entries['horse_number'].tolist()
                        # Check if winner is in our predictions df
                        pred_horses = df[(df['date'] == date) &
                                        (df['track'] == track) &
                                        (df['race'] == race)]['horse_number'].tolist()
                        missing = [h for h in w_horses if h not in pred_horses]
                        if missing:
                            entries_msg = (f" | WINNER #{missing[0]} in entries "
                                         f"but NOT in predictions ({len(pred_horses)} horses)")
                        else:
                            entries_msg = (f" | winner #{w_horses[0]} IS in predictions "
                                         f"but won=0 (merge issue?)")
            print(f"  {date} {track} R{race} ({n_horses} horses){entries_msg}")
        if len(zero_races) > 30:
            print(f"  ... and {len(zero_races) - 30} more")

    # Detail on multi-winner races
    if n_multi > 0:
        print(f"\n{'─'*72}")
        print(f"MULTI-WINNER RACES (duplicate finish=1 rows or merge dup)")
        print(f"{'─'*72}")
        multi_races = race_winners[race_winners > 1]
        for (date, track, race), n_winners in multi_races.head(20).items():
            n_horses = race_horses.loc[(date, track, race)]
            winners = df[(df['date'] == date) & (df['track'] == track) &
                        (df['race'] == race) & (df['won'] == 1)]
            horse_nums = winners['horse_number'].tolist()
            print(f"  {date} {track} R{race}: {n_winners} winners "
                  f"(#{horse_nums}) of {n_horses} horses")

    # Summary impact
    print(f"\n{'─'*72}")
    print(f"IMPACT ON CALIBRATION")
    print(f"{'─'*72}")
    if n_zero > 0:
        # Estimate inflated calibration miss
        impact_pct = n_zero / n_total
        print(f"  {n_zero}/{n_total} races ({impact_pct:.1%}) have NO winner credited")
        print(f"  Even random guessing should expect ~{n_zero} winners in these races")
        print(f"  This artificially LOWERS observed win rate by ~{impact_pct:.1%}")
        print(f"  Likely explains why high-probability bins look miscalibrated")

    if n_zero == 0 and n_multi == 0:
        print(f"  ✓ No winner attribution bugs found.")
        print(f"    The miscalibration is real — model genuinely over-predicts.")


if __name__ == '__main__':
    main()