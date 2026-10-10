"""
scan_high_payouts.py

Scan entries CSVs for unusually high win/place/show payouts. On a $2 base
in a 17%-takeout pool, a 75-1 winner pays roughly:
  $2 × 76 × 0.83 ≈ $126.16

So anything above ~$150 on win_paid is implied at ~90-1 or higher, which
for a ML-favorite market is extremely rare and often a data-entry error
(decimal misplacement, exotic payout tagged as win_paid, etc.).

Thresholds (configurable):
  WIN_MAX   = 150     # above this, flag
  PLACE_MAX = 75
  SHOW_MAX  = 50

Reports the flagged rows with date/track/race/horse so you can check
them against public charts.

Usage:
    python scan_high_payouts.py
    python scan_high_payouts.py --win-max 100   # stricter
    python scan_high_payouts.py --csv path/to/specific.csv
"""

import os, argparse
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
ENTRIES_DIR = DATA_PATH / 'entries'


def _parse_payout(val):
    """Parse '$8.40' → 8.4, invalid → None. Different from the production
    parser which returns 0 for invalid — we want to distinguish 'no payout'
    from '$0 payout' in the diagnostic."""
    if pd.isna(val):
        return None
    s = str(val).strip().replace('$', '').replace(',', '')
    if s == '' or s.lower() in ('nan', 'none', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def scan_file(path, win_max=150, place_max=75, show_max=50):
    """Return list of flagged rows from one CSV."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  skipping {path.name}: {e}")
        return []

    flagged = []
    cols = {
        'win_paid': win_max,
        'place_paid': place_max,
        'show_paid': show_max,
    }
    have_cols = [c for c in cols if c in df.columns]
    if not have_cols:
        return []

    for idx, row in df.iterrows():
        for col, threshold in cols.items():
            if col not in df.columns:
                continue
            val = _parse_payout(row[col])
            if val is None:
                continue
            if val >= threshold:
                # Implied fair odds from the payout, pre-takeout adjustment.
                # A $150 win_paid on $2 base implies ~ (150/2)/0.83 - 1 ≈ 89-1.
                implied_odds = (val / 2.0) / 0.83 - 1
                flagged.append({
                    'file': path.name,
                    'date': row.get('date', '—'),
                    'track': row.get('track', '—'),
                    'race': row.get('race', '—'),
                    'horse_number': row.get('horse_number', row.get('horse', '—')),
                    'column': col,
                    'value': val,
                    'threshold': threshold,
                    'implied_odds': f"{implied_odds:.1f}-1",
                    'ml_odds': row.get('ml_odds', '—'),
                    'finish': row.get('finish', '—'),
                })

    return flagged


def main():
    parser = argparse.ArgumentParser(
        description='Scan entries CSVs for suspiciously high payouts')
    parser.add_argument('--win-max', type=float, default=150.0,
                       help='win_paid threshold (default $150 = ~89-1 implied)')
    parser.add_argument('--place-max', type=float, default=75.0,
                       help='place_paid threshold (default $75)')
    parser.add_argument('--show-max', type=float, default=50.0,
                       help='show_paid threshold (default $50)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Scan a single CSV instead of whole entries dir')
    args = parser.parse_args()

    if args.csv:
        paths = [Path(args.csv)]
    else:
        if not ENTRIES_DIR.exists():
            print(f"{ENTRIES_DIR} does not exist")
            return
        paths = sorted(ENTRIES_DIR.glob('*.csv'))

    print(f"Scanning {len(paths)} file(s) for payouts ≥ "
          f"${args.win_max:.0f}/win, ${args.place_max:.0f}/place, "
          f"${args.show_max:.0f}/show")
    print()

    all_flagged = []
    for p in paths:
        flagged = scan_file(p, args.win_max, args.place_max, args.show_max)
        all_flagged.extend(flagged)

    if not all_flagged:
        print(f"✓ No flagged payouts found.")
        return

    print(f"Found {len(all_flagged)} suspicious payout(s):\n")
    # Group by file for readability
    by_file = {}
    for f in all_flagged:
        by_file.setdefault(f['file'], []).append(f)

    for fname, rows in sorted(by_file.items()):
        print(f"=== {fname} ({len(rows)} flag{'s' if len(rows) > 1 else ''}) ===")
        for r in rows:
            print(f"  {r['date']} {r['track']} R{r['race']} #{r['horse_number']}: "
                  f"{r['column']}=${r['value']:.2f}  "
                  f"(implied {r['implied_odds']}, ML {r['ml_odds']}, "
                  f"finish={r['finish']})")
        print()

    # Summary stats to help you decide if these are legit longshots or errors
    print(f"Summary:")
    for col in ('win_paid', 'place_paid', 'show_paid'):
        col_flags = [f for f in all_flagged if f['column'] == col]
        if col_flags:
            vals = [f['value'] for f in col_flags]
            print(f"  {col}: {len(col_flags)} flags, "
                  f"max ${max(vals):.2f}, median ${sorted(vals)[len(vals)//2]:.2f}")

    print(f"\nNext steps:")
    print(f"  - Spot-check a few against public charts (Equibase, etc.)")
    print(f"  - If legit: lower thresholds or accept these as real longshots")
    print(f"  - If data errors: fix the source CSVs, then re-run "
          f"regression_analysis.py")


if __name__ == '__main__':
    main()