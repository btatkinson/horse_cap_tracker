"""
backfill_ai.py

Run AI providers for historical dates, skipping dates that already have data.
Also adds closing odds predictions to existing Claude/Gemini CSVs.

Usage:
    # Run OpenAI for all historical dates that have chopped forms
    python backfill_ai.py --provider openai

    # Add closing odds to Claude for all dates
    python backfill_ai.py --provider claude --closing-only

    # Run all providers for missing dates
    python backfill_ai.py --provider all

    # Find AI pick files that are missing horses (per entries) and re-run them
    python backfill_ai.py --provider all --repair

    # Dry run — show what would be sent
    python backfill_ai.py --provider openai --dry-run
"""

import os, sys, subprocess, argparse
import pandas as pd
from pathlib import Path
from datetime import date

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = Path(os.environ['DATA_PATH'])
SCRIPT_DIR = Path(__file__).parent

PROVIDERS = {
    'claude': {
        'script': 'send_claude.py',
        'csv_name': 'claude.csv',
        'closing_col': 'claude_closing_prob',
    },
    'gemini': {
        'script': 'send_gemini.py',
        'csv_name': 'gemini.csv',
        'closing_col': 'gemini_closing_prob',
    },
    'openai': {
        'script': 'send_openai.py',
        'csv_name': 'openai.csv',
        'closing_col': 'openai_closing_prob',
    },
}


def find_historical_dates():
    """Find all date/track combos that have chopped forms."""
    chopped_dir = DATA_PATH / 'chopped_forms'
    if not chopped_dir.exists():
        return []

    dates = []
    for date_dir in sorted(chopped_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for track_dir in sorted(date_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            race_pdfs = list(track_dir.glob('race_*.pdf'))
            if race_pdfs:
                dates.append({
                    'date': date_dir.name,
                    'track': track_dir.name,
                    'n_races': len(race_pdfs),
                })
    return dates


def has_csv(date_str, track, provider):
    """Check if the base CSV exists with beyer+prob data."""
    info = PROVIDERS[provider]
    csv_path = DATA_PATH / 'ai_picks' / date_str / track / info['csv_name']
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
        return len(df) > 0
    except Exception:
        return False


def has_closing(date_str, track, provider):
    """Check if the CSV already has closing odds data."""
    info = PROVIDERS[provider]
    csv_path = DATA_PATH / 'ai_picks' / date_str / track / info['csv_name']
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
        col = info['closing_col']
        return col in df.columns and df[col].notna().any()
    except Exception:
        return False


def _normalize_track(t):
    if pd.isna(t):
        return t
    return str(t).strip().lower()


def _running_horses_per_race(date_str, track):
    """
    Look up the set of running horses per race for (date, track) from
    entries data. Returns dict {race_num: set(horse_numbers)}, or None
    if entries data isn't available for this card.

    Tries entries/*.csv first (historical), then upc_entries/{track}.csv
    (live). Both are valid sources of truth for "what horses are running."
    """
    track_norm = _normalize_track(track)
    entries_dir = DATA_PATH / 'entries'
    upc_dir = DATA_PATH / 'upc_entries'

    # Try historical entries first
    matched_df = None
    if entries_dir.exists():
        for f in entries_dir.glob('*.csv'):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if 'date' not in df.columns or 'track' not in df.columns:
                continue
            df['date'] = df['date'].astype(str).str.strip()
            df['track'] = df['track'].apply(_normalize_track)
            match = df[(df['date'] == date_str) & (df['track'] == track_norm)]
            if len(match) > 0:
                matched_df = match
                break

    # Fall back to live upc_entries
    if matched_df is None:
        upc_path = upc_dir / f"{track}.csv"
        if upc_path.exists():
            try:
                matched_df = pd.read_csv(upc_path)
            except Exception:
                matched_df = None

    if matched_df is None or matched_df.empty:
        return None

    df = matched_df.copy()

    # Some entries files use 'horse' instead of 'horse_number'
    if 'horse_number' not in df.columns and 'horse' in df.columns:
        df['horse_number'] = df['horse']

    if 'race' not in df.columns or 'horse_number' not in df.columns:
        return None

    df['race'] = pd.to_numeric(df['race'], errors='coerce')
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')

    # Filter to running horses only — scratched horses shouldn't be in AI picks
    if 'running' in df.columns:
        df = df[df['running'] == 1]

    df = df.dropna(subset=['race', 'horse_number'])

    result = {}
    for race_num, race_df in df.groupby('race'):
        result[int(race_num)] = set(int(h) for h in race_df['horse_number'])

    return result if result else None


def is_complete(date_str, track, provider):
    """
    Check if the AI pick file has a row for every running horse in entries.
    Returns (is_complete: bool, missing: dict of {race_num: set(missing horse_numbers)}).

    Used by --repair to find files where the AI pipeline silently dropped
    horses (truncated LLM response, malformed parse, etc.). If entries
    data isn't available, returns (True, {}) — can't repair what we can't
    verify, so don't claim it's broken.
    """
    info = PROVIDERS[provider]
    csv_path = DATA_PATH / 'ai_picks' / date_str / track / info['csv_name']
    if not csv_path.exists():
        # File doesn't exist at all — handled by has_csv, not a "repair" case
        return True, {}

    expected = _running_horses_per_race(date_str, track)
    if expected is None:
        # No entries data to compare against; can't determine completeness
        return True, {}

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False, {}

    if 'race' not in df.columns or 'horse_number' not in df.columns:
        return False, {}

    df = df.copy()
    df['race'] = pd.to_numeric(df['race'], errors='coerce')
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')
    df = df.dropna(subset=['race', 'horse_number'])

    actual = {}
    for race_num, race_df in df.groupby('race'):
        actual[int(race_num)] = set(int(h) for h in race_df['horse_number'])

    missing = {}
    for race_num, expected_set in expected.items():
        got_set = actual.get(race_num, set())
        diff = expected_set - got_set
        if diff:
            missing[race_num] = diff

    return (len(missing) == 0), missing


def run_provider(date_str, track, provider, closing_only=False, model=None, races=None):
    """Run a provider script for one date/track.

    races: optional iterable of race numbers. When provided, passes
           --races to the underlying send script so only those races are
           re-run (preserving existing data for other races on the card).
           Used by --repair to avoid redundant API calls.
    """
    info = PROVIDERS[provider]
    script = info['script']

    cmd = [sys.executable, str(SCRIPT_DIR / script),
           '--date', date_str, '--track', track, '-y']

    if model:
        cmd.extend(['--model', model])

    if closing_only:
        cmd.append('--closing-only')

    if races:
        cmd.extend(['--races', ','.join(str(r) for r in sorted(races))])

    races_str = f" (races {sorted(races)})" if races else ""
    print(f"\n{'─'*50}")
    print(f"Running {provider} for {date_str}/{track}"
          f"{' (closing only)' if closing_only else ''}{races_str}")
    print(f"{'─'*50}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Backfill AI predictions for historical dates')
    parser.add_argument('--provider', type=str, required=True,
                        choices=['claude', 'gemini', 'openai', 'all'],
                        help='Which provider to run')
    parser.add_argument('--closing-only', action='store_true',
                        help='Only add closing odds (skip beyer/prob if they exist)')
    parser.add_argument('--repair', action='store_true',
                        help='Find AI pick files that are missing horses '
                             '(fewer rows than entries says are running) and '
                             're-run them. Catches cases where the LLM '
                             'silently dropped horses from the response.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be sent without sending')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model name')
    parser.add_argument('--date', type=str, default=None,
                        help='Run for a specific date only')
    parser.add_argument('--track', type=str, default=None,
                        help='Run for a specific track only')
    args = parser.parse_args()

    if args.repair and args.closing_only:
        print("ERROR: --repair and --closing-only are mutually exclusive. "
              "Repair re-runs the full provider; closing-only just adds odds.")
        return

    providers = list(PROVIDERS.keys()) if args.provider == 'all' else [args.provider]

    dates = find_historical_dates()
    if not dates:
        print("No chopped forms found.")
        return

    if args.date:
        dates = [d for d in dates if d['date'] == args.date]
    if args.track:
        dates = [d for d in dates if d['track'] == args.track.lower()]

    print(f"Found {len(dates)} date/track combos with chopped forms")
    print(f"Providers: {', '.join(providers)}")
    if args.repair:
        print(f"Mode: REPAIR (compare AI picks to entries, re-run if short)")

    # Check what needs to be done
    work = []
    for provider in providers:
        for d in dates:
            date_str, track = d['date'], d['track']

            races_to_run = None  # None = all races on the card

            if args.repair:
                # Repair mode: re-run files that are missing horses per entries.
                # Skip files that don't exist (those are has_csv's job, not ours).
                if not has_csv(date_str, track, provider):
                    continue  # nothing to repair if there's no file
                complete, missing = is_complete(date_str, track, provider)
                if complete:
                    continue  # file is complete, leave it alone
                # Capture the specific short races so we can pass --races
                # to the send script and avoid re-sending the whole card
                races_to_run = sorted(missing.keys())
                short_races = races_to_run
                n_missing_total = sum(len(s) for s in missing.values())
                if len(short_races) <= 4:
                    races_str = ', '.join(f"R{r}" for r in short_races)
                else:
                    races_str = (f"R{short_races[0]}..R{short_races[-1]} "
                                f"({len(short_races)} races)")
                status = f"missing {n_missing_total}h in {races_str}"
            elif args.closing_only:
                # Only need to send closing odds
                if has_closing(date_str, track, provider):
                    continue  # already has closing data, skip
                if not has_csv(date_str, track, provider):
                    print(f"  SKIP {provider} {date_str}/{track}: "
                          f"no base CSV (run without --closing-only first)")
                    continue
                status = 'needs closing'
            else:
                # Full run: beyer + prob + closing
                if has_csv(date_str, track, provider):
                    continue  # already has base data, skip
                status = 'missing'

            work.append({
                'date': date_str,
                'track': track,
                'provider': provider,
                'status': status,
                # n_races shows the COUNT we'll actually request, not the card size.
                # For repair this is len(races_to_run); for full runs it's card size.
                'n_races': len(races_to_run) if races_to_run else d['n_races'],
                'races': races_to_run,  # specific races, or None for "all"
            })

    if not work:
        if args.repair:
            print("\nNothing to repair — every AI pick file matches its entries.")
        else:
            print("\nNothing to do — all data exists.")
        return

    print(f"\nWork to do: {len(work)} provider/date/track combos")
    # Wider Status column when in repair mode since the status strings carry detail
    status_w = 50 if args.repair else 20
    print(f"\n  {'Provider':<10s} {'Date':<12s} {'Track':<12s} "
          f"{'Races':>5s} {'Status':<{status_w}s}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*5} {'-'*status_w}")
    for w in work:
        print(f"  {w['provider']:<10s} {w['date']:<12s} {w['track']:<12s} "
              f"{w['n_races']:>5d} {w['status']:<{status_w}s}")

    if args.dry_run:
        print(f"\n  (dry run — nothing sent)")
        return

    # Confirm
    if args.closing_only:
        total_requests = sum(w['n_races'] for w in work)
    else:
        # Repair re-runs the same way a full run does — same request count
        total_requests = sum(w['n_races'] * (3 if w['provider'] == 'openai' else 2)
                            for w in work)
    print(f"\n  Total API requests: ~{total_requests}")
    confirm = input("  Proceed? [y/N] ")
    if confirm.lower() != 'y':
        print("  Aborted.")
        return

    # Run
    success = 0
    failed = 0
    for w in work:
        ok = run_provider(w['date'], w['track'], w['provider'],
                          closing_only=args.closing_only,
                          model=args.model,
                          races=w.get('races'))
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Backfill complete: {success} succeeded, {failed} failed")


if __name__ == '__main__':
    main()