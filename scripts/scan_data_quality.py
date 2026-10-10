"""
scan_data_quality.py

Two diagnostics:

  1. MISSING PAYOUTS: horses with finish=1 (i.e. winners) that have $0
     or blank win_paid. On a $2 base with takeout, even a strong favorite
     pays ≥ $2.10 to win, so a winner with $0 payout is a data error.
     Same check for finish=2 (place_paid) and finish=3 (show_paid).

  2. ML-ODDS DISTRIBUTION: histogram of ml_odds across all entries,
     so you can see if the historical data skews to longshots (explaining
     why backtest predicts longshots but live predicts favorites).
     Compares backtest ML distribution against typical live distributions.

Usage:
    python scan_data_quality.py
    python scan_data_quality.py --csv data/entries/specific.csv
"""

import os, argparse
from collections import Counter
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
ENTRIES_DIR = DATA_PATH / 'entries'


def _parse_payout(val):
    """Returns None for blank/invalid, float for parseable ($, commas stripped)."""
    if pd.isna(val):
        return None
    s = str(val).strip().replace('$', '').replace(',', '')
    if s == '' or s.lower() in ('nan', 'none', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_ml_decimal(odds_str):
    """Convert '5-2' or '2/1' to decimal odds (3.5, 3.0). Returns None on failure."""
    if pd.isna(odds_str):
        return None
    s = str(odds_str).strip()
    for sep in ('-', '/'):
        if sep in s:
            try:
                num, den = s.split(sep, 1)
                num, den = float(num), float(den)
                if den <= 0:
                    return None
                return num / den + 1  # decimal odds: 5-2 → 3.5
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def scan_missing_payouts(df, source_name):
    """Find winners/placers/showers with missing payouts."""
    issues = []
    if 'finish' not in df.columns:
        return issues

    df = df.copy()
    df['_finish'] = pd.to_numeric(df['finish'], errors='coerce')

    # For each finish position, the corresponding payout column must be >$0
    checks = [
        (1, 'win_paid'),
        (2, 'place_paid'),
        (3, 'show_paid'),
    ]

    for finish_pos, col in checks:
        if col not in df.columns:
            continue
        mask = df['_finish'] == finish_pos
        if not mask.any():
            continue

        for _, row in df[mask].iterrows():
            val = _parse_payout(row[col])
            # Null, blank, or $0 — all suspicious
            if val is None or val == 0:
                issues.append({
                    'file': source_name,
                    'date': row.get('date', '—'),
                    'track': row.get('track', '—'),
                    'race': row.get('race', '—'),
                    'horse_number': row.get('horse_number', row.get('horse', '—')),
                    'finish': finish_pos,
                    'column': col,
                    'raw_value': repr(row[col]),
                    'ml_odds': row.get('ml_odds', '—'),
                })

    return issues


def ml_odds_distribution(df):
    """Bucket ml_odds values by decimal-odds range."""
    if 'ml_odds' not in df.columns:
        return None

    buckets = {
        '≤2.0 (1-1 to even)': 0,
        '2.0-3.5 (even to 5-2)': 0,
        '3.5-4.5 (5-2 to 7-2)': 0,
        '4.5-6.0 (7-2 to 5-1)': 0,
        '6.0-9.0 (5-1 to 8-1)': 0,
        '9.0-15 (8-1 to 14-1)': 0,
        '15-30 (14-1 to 29-1)': 0,
        '>30 (29-1+)': 0,
        'unparseable': 0,
    }

    for val in df['ml_odds']:
        d = _parse_ml_decimal(val)
        if d is None:
            buckets['unparseable'] += 1
        elif d <= 2.0:
            buckets['≤2.0 (1-1 to even)'] += 1
        elif d <= 3.5:
            buckets['2.0-3.5 (even to 5-2)'] += 1
        elif d <= 4.5:
            buckets['3.5-4.5 (5-2 to 7-2)'] += 1
        elif d <= 6.0:
            buckets['4.5-6.0 (7-2 to 5-1)'] += 1
        elif d <= 9.0:
            buckets['6.0-9.0 (5-1 to 8-1)'] += 1
        elif d <= 15.0:
            buckets['9.0-15 (8-1 to 14-1)'] += 1
        elif d <= 30.0:
            buckets['15-30 (14-1 to 29-1)'] += 1
        else:
            buckets['>30 (29-1+)'] += 1

    return buckets


def main():
    parser = argparse.ArgumentParser(description='Data quality diagnostics')
    parser.add_argument('--csv', type=str, default=None,
                       help='Scan a single CSV instead of entries directory')
    args = parser.parse_args()

    if args.csv:
        paths = [Path(args.csv)]
    else:
        if not ENTRIES_DIR.exists():
            print(f"{ENTRIES_DIR} does not exist")
            return
        paths = sorted(ENTRIES_DIR.glob('*.csv'))

    print(f"Scanning {len(paths)} file(s)\n")

    # ── Missing payouts ──
    all_issues = []
    combined_frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  skipping {p.name}: {e}")
            continue
        combined_frames.append(df)
        all_issues.extend(scan_missing_payouts(df, p.name))

    print(f"{'='*70}")
    print(f"MISSING PAYOUTS (winners/placers/showers with $0 or blank)")
    print(f"{'='*70}")
    if not all_issues:
        print("✓ No missing payouts found.")
    else:
        print(f"Found {len(all_issues)} missing-payout row(s):")
        # Group by column to show pattern
        by_col = {}
        for iss in all_issues:
            by_col.setdefault(iss['column'], []).append(iss)
        for col, rows in sorted(by_col.items()):
            print(f"\n  {col}: {len(rows)} missing "
                  f"(out of expected {len(rows)} {'winners' if col == 'win_paid' else 'placers' if col == 'place_paid' else 'showers'})")
            for r in rows[:20]:  # cap at 20 per column
                print(f"    {r['date']} {r['track']} R{r['race']} "
                      f"#{r['horse_number']} finish={r['finish']} "
                      f"ml={r['ml_odds']} raw={r['raw_value']}")
            if len(rows) > 20:
                print(f"    ... and {len(rows) - 20} more")

    # ── ML-odds distribution ──
    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        dist = ml_odds_distribution(combined)

        print(f"\n{'='*70}")
        print(f"ML-ODDS DISTRIBUTION (all entries rows, all dates)")
        print(f"{'='*70}")
        total = sum(dist.values()) if dist else 0
        if total == 0:
            print("No ml_odds column or no parseable values.")
        else:
            max_count = max(dist.values())
            for bucket, n in dist.items():
                pct = n / total * 100 if total else 0
                bar = '█' * int(40 * n / max_count) if max_count else ''
                print(f"  {bucket:<24s} {n:>5d}  ({pct:>5.1f}%)  {bar}")
            print(f"\n  Total: {total} entries")

            # Sanity flag: if >40% of horses are >15-1 ML, that's a red flag.
            # Typical big-track cards have maybe 25-35% at that range.
            longshot_n = dist['15-30 (14-1 to 29-1)'] + dist['>30 (29-1+)']
            if total > 0 and longshot_n / total > 0.40:
                print(f"\n  ⚠ {longshot_n}/{total} ({longshot_n/total:.0%}) of entries "
                      f"are at 15-1 or higher — unusually many longshots.")
                print(f"    Could be ML-odds parsing issue (fractional misinterpreted)")
                print(f"    or the cards are genuinely longshot-heavy.")


if __name__ == '__main__':
    main()