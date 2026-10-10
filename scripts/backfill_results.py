"""
backfill_results.py

Walk every (date, track) pair in {DATA_PATH}/predictions/ and call
download_results.py to fetch the entries + meta CSV (with payouts and
horizontals data) for each one.

Use cases:
    1. Initial fill: ran predictions for a stretch of dates, never
       downloaded results. Run this once to backfill everything.
    2. Refresh: a parser bug got fixed in download_results.py, you
       want to re-download with the corrected parsing. Use --force.
    3. Periodic catch-up: cards were predicted but you forgot to
       grab results that day. Run this; it skips done pairs.

Skip behavior:
    By default, a (date, track) pair is skipped if BOTH of these exist:
      {DATA_PATH}/entries/{track}_{date}.csv
      {DATA_PATH}/meta/{track}_{date}.csv
    A pair where only one is present gets re-fetched (something went
    wrong last time and we want to retry).

Rate limiting:
    Default: 2 seconds between requests. Configurable with --sleep.
    Skipped pairs do NOT count toward rate-limit sleeps — only actual
    network requests trigger the wait. So an all-skipped run is fast.

Usage:
    python backfill_results.py
    python backfill_results.py --sleep 5            # slower
    python backfill_results.py --force              # re-download all
    python backfill_results.py --tracks keeneland,churchill-downs
    python backfill_results.py --dry-run            # show plan, do nothing
    python backfill_results.py --start-date 2026-04-01
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Match the pattern in download_results.py and other scripts: load .env
# from the parent of scripts/ so we can run from any directory.
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
SCRIPTS_DIR = Path(__file__).parent
DOWNLOAD_SCRIPT = SCRIPTS_DIR / 'download_results.py'


def discover_pairs():
    """
    Walk {DATA_PATH}/predictions/ to find every (date, track) pair with
    a predictions.csv file.

    Layout assumed:
        {DATA_PATH}/predictions/{date}/{track}/predictions.csv

    Returns a sorted list of (date_str, track) tuples. Sorting by date
    keeps the run order chronological — useful when watching tqdm-style
    output and helps debugging if something fails midway.
    """
    pred_root = DATA_PATH / 'predictions'
    if not pred_root.exists():
        return []

    pairs = []
    for date_dir in pred_root.iterdir():
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        # Sanity check: name should look like YYYY-MM-DD. Skip files
        # like walk_forward_summary.csv that live in this directory.
        if not _looks_like_date(date_str):
            continue
        for track_dir in date_dir.iterdir():
            if not track_dir.is_dir():
                continue
            if (track_dir / 'predictions.csv').exists():
                pairs.append((date_str, track_dir.name))

    return sorted(pairs)


def _looks_like_date(s):
    try:
        datetime.strptime(s, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def is_pair_done(date_str, track):
    """
    A pair counts as 'done' only if BOTH expected output files exist.
    If just one is present (rare — partial failure), we re-fetch.
    """
    entries = DATA_PATH / 'entries' / f'{track}_{date_str}.csv'
    meta = DATA_PATH / 'meta' / f'{track}_{date_str}.csv'
    return entries.exists() and meta.exists()


def run_download(date_str, track, verbose=False):
    """
    Shell out to download_results.py. Returns True on success.

    On failure, prints stderr tail so the caller can see why without
    rerunning. Does NOT raise — failures of one pair shouldn't kill
    the whole backfill.
    """
    if not DOWNLOAD_SCRIPT.exists():
        print(f"ERROR: download_results.py not found at {DOWNLOAD_SCRIPT}")
        return False

    cmd = [sys.executable, str(DOWNLOAD_SCRIPT),
           '--track', track, '--date', date_str]

    # Force utf-8 in the child process — same Windows cp1252 problem
    # backtest_walk_forward.py hits.
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    if verbose:
        result = subprocess.run(cmd, env=env, check=False)
        return result.returncode == 0

    result = subprocess.run(
        cmd, env=env, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='replace',
    )

    if result.returncode != 0:
        # Surface enough info to debug without forcing a verbose rerun.
        # Last ~20 lines is usually enough to see the traceback origin.
        print(f"  FAILED (exit {result.returncode}):")
        if result.stderr:
            tail = '\n'.join(result.stderr.rstrip().split('\n')[-20:])
            print(f"  stderr tail:")
            for line in tail.split('\n'):
                print(f"    {line}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Backfill historical entries+meta CSVs for every '
                    '(date, track) pair with predictions.'
    )
    parser.add_argument('--sleep', type=float, default=2.0,
                        help='Seconds to sleep between network requests '
                             '(default 2.0, only applies to actual fetches '
                             'not to skipped pairs)')
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if entries+meta already exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the plan and exit without any requests')
    parser.add_argument('--tracks', type=str, default=None,
                        help='Comma-separated track filter '
                             '(e.g. "keeneland,churchill-downs")')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Only process dates >= this (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Only process dates <= this (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true',
                        help='Stream child stdout/stderr live (noisy)')
    args = parser.parse_args()

    pairs = discover_pairs()
    if not pairs:
        print(f"No predictions found under {DATA_PATH / 'predictions'}.")
        return

    # Apply filters
    if args.tracks:
        wanted_tracks = set(t.strip() for t in args.tracks.split(','))
        pairs = [(d, t) for d, t in pairs if t in wanted_tracks]
    if args.start_date:
        pairs = [(d, t) for d, t in pairs if d >= args.start_date]
    if args.end_date:
        pairs = [(d, t) for d, t in pairs if d <= args.end_date]

    if not pairs:
        print("No pairs match the given filters.")
        return

    # Partition into to-do vs already-done
    todo = []
    skipped = []
    for date_str, track in pairs:
        if not args.force and is_pair_done(date_str, track):
            skipped.append((date_str, track))
        else:
            todo.append((date_str, track))

    print(f"Backfill plan:")
    print(f"  Discovered:  {len(pairs)} (date, track) pairs")
    print(f"  Already done: {len(skipped)} (skipping; pass --force to redo)")
    print(f"  To download:  {len(todo)}")
    if todo:
        print(f"  Sleep between requests: {args.sleep}s")
        # ETA: each request is ~5-10s of network + parsing + sleep.
        # Be conservative and assume 8s per request including sleep.
        est_seconds = len(todo) * (8 + args.sleep)
        if est_seconds < 90:
            eta_str = f"~{int(est_seconds)}s"
        else:
            eta_str = f"~{est_seconds/60:.1f}m"
        print(f"  Rough ETA:   {eta_str}")

    if args.dry_run:
        print("\nDry run, exiting.")
        if todo:
            print("\nWould download (first 10):")
            for d, t in todo[:10]:
                print(f"  {d} / {t}")
            if len(todo) > 10:
                print(f"  ... and {len(todo) - 10} more")
        return

    if not todo:
        print("\nNothing to do.")
        return

    print()
    succeeded = []
    failed = []
    for i, (date_str, track) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {date_str} / {track} ...", flush=True)
        ok = run_download(date_str, track, verbose=args.verbose)
        if ok:
            succeeded.append((date_str, track))
        else:
            failed.append((date_str, track))

        # Sleep between requests, but not after the last one
        if i < len(todo):
            time.sleep(args.sleep)

    print()
    print("=" * 60)
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed, "
          f"{len(skipped)} skipped.")
    if failed:
        print(f"\nFailed pairs:")
        for d, t in failed:
            print(f"  {d} / {t}")
        print(f"\nRerun with --verbose on a single pair to see full error:")
        d, t = failed[0]
        print(f"  python download_results.py --date {d} --track {t}")


if __name__ == '__main__':
    main()
