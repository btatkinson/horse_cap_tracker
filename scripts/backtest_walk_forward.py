"""
backtest_walk_forward.py

Walk-forward backtest: for each holdout window of N consecutive days,
retrain the 3 regression models on all OTHER dates, then run the full
predict_v2 pipeline (LLM ensemble_prob + regression models + bet logic)
on the holdout date(s). The predictions.csv files predict_v2 produces
are the true out-of-sample bet candidates — no data leakage, same code
path as production.

This is the "gold standard" backtest. The simpler backtest_kfold.py uses
the CV predictions from regression_analysis.py directly, but those don't
include ensemble_prob (LLM+Beyer blend) or the real ev_per_2 formula
that predict_v2 uses live. This script fixes that.

Flow per holdout:
    1. Move holdout date's data out of training (via regression_analysis
       --skip-dates flag). Refit all 3 models; new JSONs written to disk.
    2. Synthesize upc_entries/{track}.csv from the post-race entries
       data for the holdout date. This is critical: upc_entries is
       per-track (no date in filename), so without this step every
       holdout would read whatever upc_entries scrape happens to be on
       disk — which is some recent live card, not the holdout date.
       That made every walk-forward prediction silently use the wrong
       horses + ML odds. Synthesizing per-holdout from entries gives
       predict_v2 a clean view of just the date being predicted.
       (We back up the original upc_entries first and restore at the
       end so live predict_v2 runs aren't affected.)
    3. Run predict_v2 against each track on the holdout date. This writes
       data/predictions/{date}/{track}/predictions.csv with ensemble_prob,
       ev_per_2, ridge_value, raw_value — everything backtest_kfold
       would want but can't reconstruct.
    4. After all holdouts processed, re-run regression_analysis.py on the
       FULL dataset so production JSONs are back to normal. (Without this
       step, your live predict_v2 runs would use the last holdout's
       leave-one-out model, which is missing data.)

Output:
    - data/predictions/{date}/{track}/predictions.csv  (per-date, as predict_v2 writes)
    - data/predictions/walk_forward_summary.csv        (concatenated, with is_holdout=True marker)

Usage:
    python backtest_walk_forward.py
    python backtest_walk_forward.py --holdout-days 2   # pair-of-days holdouts
    python backtest_walk_forward.py --dates 2025-03-01,2025-03-05   # only these
    python backtest_walk_forward.py --skip-final-retrain  # if you'll retrain manually

After running, feed walk_forward_summary.csv into backtest_kfold.py
(with a flag to prefer it over regression_predictions.csv) for honest
ROI numbers.
"""

import os, sys, argparse, subprocess, shutil, warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
SCRIPTS_DIR = Path(__file__).parent


def discover_dates_and_tracks():
    """
    Walk the ai_picks directory to find every (date, track) pair with
    saved AI picks. These are the dates that can be held out.

    Returns: {date_str: [track, ...]} dict sorted by date.
    """
    picks_dir = DATA_PATH / 'ai_picks'
    if not picks_dir.exists():
        raise FileNotFoundError(f"{picks_dir} does not exist")

    result = {}
    for date_dir in sorted(picks_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        tracks = []
        for track_dir in sorted(date_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            # Must have at least one AI pick CSV to count
            if any(track_dir.glob('*.csv')):
                tracks.append(track_dir.name)
        if tracks:
            result[date_str] = tracks

    return result


def chunk_holdouts(sorted_dates, window):
    """
    Yield consecutive holdout windows of size `window`.

    For window=1: [[d1], [d2], [d3], ...]    (pure leave-one-out)
    For window=2: [[d1, d2], [d3, d4], ...]  (paired holdouts)
    For window=7: roughly weekly chunks.

    The last chunk may be shorter than `window` if the date count isn't
    divisible — that's intentional. Skipping the partial chunk would
    leave late-period dates untested.
    """
    for i in range(0, len(sorted_dates), window):
        yield sorted_dates[i:i + window]


def _run_subprocess(cmd, label, verbose=False, ignore_failure=False):
    """
    Run a subprocess with proper output handling for Windows + unicode.

    Two problems the raw-subprocess approach has:

    1. When verbose=False, the previous code sent stdout/stderr to DEVNULL.
       A failure in the child script (traceback, missing file, encoding
       error) was therefore invisible — caller saw only the exit code.
       Fix: capture stdout+stderr to buffers, print them on failure so
       the first failing iteration shows the actual error.

    2. Windows terminals default to cp1252, which can't encode some unicode
       characters produced by regression_analysis/predict_v2 (σ, →, ✓,
       arrow glyphs, etc.). When stdout is inherited from the parent, a
       print() with such a character raises UnicodeEncodeError inside the
       child process — non-verbose mode masks this because DEVNULL accepts
       any bytes. Fix: pass PYTHONIOENCODING=utf-8 in the child env so
       prints can always encode, independent of terminal locale.

    Returns True on success, False on failure (unless ignore_failure=False,
    in which case raises RuntimeError).
    """
    # Force utf-8 in the child process — avoids Windows cp1252 crashes on
    # unicode print statements. PYTHONUTF8=1 is Python 3.7+ and makes the
    # interpreter use utf-8 for stdin/stdout/stderr regardless of locale.
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    if verbose:
        # Live output to the terminal — use the parent's stdout/stderr.
        result = subprocess.run(cmd, check=False, env=env)
        if result.returncode != 0 and not ignore_failure:
            raise RuntimeError(
                f"{label} failed (exit {result.returncode}). "
                f"See output above."
            )
        return result.returncode == 0

    # Quiet mode: capture output so we can surface it on failure.
    # text=True + encoding='utf-8' decodes the captured bytes correctly.
    result = subprocess.run(
        cmd, check=False, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='replace',
    )

    if result.returncode != 0:
        # Print what the child said so the user can diagnose without
        # rerunning in --verbose mode.
        print(f"\n  === {label} output (exit {result.returncode}) ===")
        if result.stdout:
            # Just the last ~30 lines — full output is usually noise
            # from before the error
            stdout_tail = '\n'.join(result.stdout.rstrip().split('\n')[-30:])
            print(f"  stdout (tail):\n{stdout_tail}")
        if result.stderr:
            stderr_tail = '\n'.join(result.stderr.rstrip().split('\n')[-30:])
            print(f"  stderr (tail):\n{stderr_tail}")
        print(f"  === end output ===\n")

        if not ignore_failure:
            raise RuntimeError(
                f"{label} failed (exit {result.returncode}). "
                f"See output above."
            )
        return False

    return True


def run_regression_training(skip_dates=None, verbose=False):
    """
    Shell out to regression_analysis.py with --skip-dates.

    Using subprocess rather than importing because regression_analysis is
    a heavyweight script with lots of module-level state (the dotenv load,
    DATA_PATH binding, etc.). Subprocess gives a clean process each time
    and matches what the user runs manually.
    """
    script = SCRIPTS_DIR / 'regression_analysis.py'
    if not script.exists():
        raise FileNotFoundError(f"{script} not found alongside this script")

    cmd = [sys.executable, str(script)]
    if skip_dates:
        cmd.extend(['--skip-dates', ','.join(sorted(skip_dates))])

    _run_subprocess(cmd, 'regression_analysis.py', verbose=verbose)


def run_predict_for_date(date_str, track, verbose=False, include_tipmeerkat=False):
    """
    Shell out to predict_v2.py for one date/track.

    predict_v2 writes data/predictions/{date}/{track}/predictions.csv.
    The newly-trained JSONs (from run_regression_training) are read at
    prediction time, so this produces genuinely out-of-sample predictions.

    Returns True on success. On failure, prints the error and returns
    False rather than aborting — one bad date shouldn't kill the whole
    walk-forward run.

    include_tipmeerkat: passes --include-tipmeerkat through to predict_v2.
        Default False because predict_v2 itself defaults to skipping
        tipmeerkat (its data is currently bugged).
    """
    script = SCRIPTS_DIR / 'predict_v2.py'
    if not script.exists():
        raise FileNotFoundError(f"{script} not found alongside this script")

    cmd = [sys.executable, str(script), '--date', date_str, '--track', track]
    if include_tipmeerkat:
        cmd.append('--include-tipmeerkat')
    ok = _run_subprocess(
        cmd, f'predict_v2.py for {date_str}/{track}',
        verbose=verbose, ignore_failure=True,
    )
    if not ok:
        print(f"  (continuing with next date)")
    return ok


def synthesize_upc_from_entries(date_str, track):
    """
    Build a synthetic upc_entries/{track}.csv for one specific holdout date
    using the post-race entries/*.csv files as ground truth.

    Why this exists
    ---------------
    upc_entries/{track}.csv is the morning-of snapshot predict_v2 reads
    for scratches and ML odds. Its filename is per-track (no date), so
    when backtest_walk_forward iterates through historical dates, every
    iteration of predict_v2 reads whatever upc_entries was on disk at
    backtest time — which is whatever was scraped most recently, NOT the
    state on the historical race day. The result: predict_v2 sees a
    completely different field of horses than actually ran, with mismatched
    ML odds and bogus scratch flags. Every walk-forward prediction is
    evaluated against a phantom card.

    The fix here only applies during backtest. Live runs of predict_v2
    still use the real upc_entries (which IS correct on race morning).
    By synthesizing a fake upc_entries from the post-race entries before
    each holdout, predict_v2 sees the right horses with the right ML odds
    for the date being predicted.

    Returns the path of the temporary upc_entries snapshot. Caller must
    restore the original file afterward — see backup_upc / restore_upc.
    """
    entries_dir = DATA_PATH / 'entries'
    if not entries_dir.exists():
        return None

    # Track aliases — entries CSVs may use 'gulfstream-park' while upc
    # uses 'gulfstream'. predict_v2 reads upc_entries/{track}.csv with
    # exact name match, so we write a file named after the track key
    # we're being called with.
    aliases = {
        'gulfstream-park': 'gulfstream', 'gulfstream park': 'gulfstream',
        'churchill-downs': 'churchill', 'churchill downs': 'churchill',
    }
    track_normalized = aliases.get(track.lower(), track.lower())

    # Pull all entries CSVs and filter to (date, track). Same merge
    # pattern regression_analysis uses.
    frames = []
    for f in entries_dir.glob('*.csv'):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return None
    entries = pd.concat(frames, ignore_index=True)

    if 'track' in entries.columns:
        entries['track'] = entries['track'].astype(str).str.strip().str.lower()
        entries['track'] = entries['track'].replace(aliases)
    if 'horse' in entries.columns and 'horse_number' not in entries.columns:
        entries['horse_number'] = pd.to_numeric(
            entries['horse'], errors='coerce'
        ).astype('Int64')

    # Some entries CSVs have date columns in different formats; normalize.
    if 'date' in entries.columns:
        entries['date'] = entries['date'].astype(str).str.strip()

    card = entries[
        (entries.get('date') == date_str)
        & (entries.get('track') == track_normalized)
    ].copy()
    if card.empty:
        # Try the un-aliased track name in case the entries CSV uses it
        card = entries[
            (entries.get('date') == date_str)
            & (entries.get('track') == track.lower())
        ].copy()
    if card.empty:
        return None

    # Build the synthetic upc_entries with exactly the columns predict_v2
    # reads (load_upc_entries + the ml_odds merge inside run_predictions).
    # predict_v2 needs: race, horse_number, running, ml_odds. It also
    # accepts 'horse' as an alias for horse_number.
    out = pd.DataFrame()
    out['race'] = pd.to_numeric(card['race'], errors='coerce').astype('Int64')
    out['horse_number'] = pd.to_numeric(
        card['horse_number'], errors='coerce'
    ).astype('Int64')
    if 'horse' in card.columns:
        out['horse'] = card['horse'].values
    else:
        out['horse'] = out['horse_number']

    # Running flag: post-race entries contain only horses that actually
    # entered, with running=1 for those who started and running=0 for
    # late scratches. We trust the entries flag — it's ground truth.
    if 'running' in card.columns:
        out['running'] = pd.to_numeric(card['running'], errors='coerce').fillna(1).astype(int)
    else:
        # Older entries CSVs without a running column: assume everyone ran.
        out['running'] = 1

    # ML odds: copy through. predict_v2's _parse_ml_odds handles strings
    # like '6/1' or decimals; we don't need to convert.
    if 'ml_odds' in card.columns:
        out['ml_odds'] = card['ml_odds'].values
    else:
        out['ml_odds'] = ''

    # date/track columns are optional but useful for debugging if you
    # cat the temp file later.
    out['date'] = date_str
    out['track'] = track_normalized

    # Drop duplicate (race, horse_number) — entries occasionally has
    # multiple rows per horse from joined sources. Keep first.
    out = out.drop_duplicates(['race', 'horse_number'])

    # Write to the path predict_v2 reads. Sort for human readability.
    out = out.sort_values(['race', 'horse_number'])
    upc_dir = DATA_PATH / 'upc_entries'
    upc_dir.mkdir(parents=True, exist_ok=True)
    out_path = upc_dir / f'{track}.csv'
    out.to_csv(out_path, index=False)
    return out_path


def backup_upc(track):
    """Move the existing upc_entries/{track}.csv to a sibling .bak file
    so we can restore it in finally:. Returns the path of the backup
    if one was made, else None."""
    upc_dir = DATA_PATH / 'upc_entries'
    candidates = [
        upc_dir / f'{track}.csv',
        upc_dir / f'{track.replace("-", "_")}.csv',
    ]
    for p in candidates:
        if p.exists():
            backup_path = p.with_suffix('.csv.bak_walkforward')
            shutil.copy2(p, backup_path)
            return p, backup_path
    return None, None


def restore_upc(original_path, backup_path):
    """Restore the backed-up upc_entries file. No-op if no backup exists."""
    if backup_path is None or not backup_path.exists():
        return
    shutil.copy2(backup_path, original_path)
    backup_path.unlink()


def collect_predictions(dates_run):
    """
    Concatenate all per-date predictions.csv files into one summary.

    Adds `is_holdout=True` so backtest_kfold can distinguish walk-forward
    rows from any stale in-sample predictions that might linger in the
    predictions directory.
    """
    pred_root = DATA_PATH / 'predictions'
    frames = []

    for date_str in sorted(dates_run):
        date_dir = pred_root / date_str
        if not date_dir.exists():
            continue
        for track_dir in date_dir.iterdir():
            if not track_dir.is_dir():
                continue
            pred_csv = track_dir / 'predictions.csv'
            if not pred_csv.exists():
                continue
            try:
                df = pd.read_csv(pred_csv)
                # predict_v2 writes date/track into the file, but be defensive
                if 'date' not in df.columns:
                    df['date'] = date_str
                if 'track' not in df.columns:
                    df['track'] = track_dir.name
                df['is_holdout'] = True
                frames.append(df)
            except Exception as e:
                print(f"  WARNING: could not read {pred_csv}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward backtest: holdout retrain + full predict_v2')
    parser.add_argument('--holdout-days', type=int, default=1,
                       help='Number of consecutive days per holdout window '
                            '(default 1 = leave-one-out). Larger windows '
                            'trade fewer retrains for higher per-iteration '
                            'data loss.')
    parser.add_argument('--dates', type=str, default=None,
                       help='Comma-separated YYYY-MM-DD list to use as '
                            'holdouts (default: every date in ai_picks/).')
    parser.add_argument('--skip-final-retrain', action='store_true',
                       help="Don't re-fit on full data after loop. Useful if "
                            "you want to inspect the last holdout's JSON, or "
                            "if you'll manually run regression_analysis.py.")
    parser.add_argument('--verbose', action='store_true',
                       help='Show stdout from regression_analysis + predict_v2 '
                            '(noisy but useful for debugging)')
    parser.add_argument('--include-tipmeerkat', action='store_true',
                       help='Pass --include-tipmeerkat through to predict_v2 '
                            'for each holdout. Default behavior matches '
                            "predict_v2's default (skip tipmeerkat — bugged data).")
    args = parser.parse_args()

    # Discover what's available
    all_dates_tracks = discover_dates_and_tracks()
    if not all_dates_tracks:
        print(f"No dates found under {DATA_PATH / 'ai_picks'}")
        return

    if args.dates:
        requested = set(d.strip() for d in args.dates.split(','))
        sorted_dates = sorted(requested & set(all_dates_tracks.keys()))
        missing = requested - set(all_dates_tracks.keys())
        if missing:
            print(f"  WARNING: requested dates missing AI picks: {sorted(missing)}")
    else:
        sorted_dates = sorted(all_dates_tracks.keys())

    if not sorted_dates:
        print("No dates to process")
        return

    total_tracks = sum(len(all_dates_tracks[d]) for d in sorted_dates)
    print(f"Walk-forward backtest")
    print(f"  Dates to hold out: {len(sorted_dates)} ({sorted_dates[0]} → {sorted_dates[-1]})")
    print(f"  Tracks across holdouts: {total_tracks}")
    print(f"  Holdout window: {args.holdout_days} day(s)")
    print(f"  Iterations: {-(-len(sorted_dates) // args.holdout_days)} "
          f"(≈ {-(-len(sorted_dates) // args.holdout_days)} retrains)")
    print()

    chunks = list(chunk_holdouts(sorted_dates, args.holdout_days))
    all_dates_run = []

    # Back up every track's upc_entries up front. Each holdout iteration
    # overwrites upc_entries/{track}.csv with a synthetic snapshot built
    # from the post-race entries, then we restore at the end. This is
    # what makes the backtest honest: predict_v2 reads upc_entries fresh
    # for each holdout date instead of getting whatever the most recent
    # live scrape left there. See synthesize_upc_from_entries for the
    # rationale.
    all_tracks = {t for tracks in all_dates_tracks.values() for t in tracks}
    backups = {}  # track → (original_path, backup_path)
    for t in all_tracks:
        orig, bak = backup_upc(t)
        if orig is not None:
            backups[t] = (orig, bak)
    if backups:
        print(f"Backed up upc_entries for {len(backups)} track(s); will "
              f"restore after walk-forward completes.")
    print()

    try:
        for i, chunk in enumerate(chunks, 1):
            print(f"[{i}/{len(chunks)}] Holdout: {chunk}")
            print(f"  Retraining on all other dates...", flush=True)

            try:
                run_regression_training(skip_dates=chunk, verbose=args.verbose)
            except Exception as e:
                print(f"  FAILED to retrain: {e}")
                continue

            for date_str in chunk:
                tracks = all_dates_tracks.get(date_str, [])
                for track in tracks:
                    # Synthesize a per-date upc_entries before each
                    # predict_v2 call. If entries data isn't available
                    # for this date (e.g. results not yet backfilled),
                    # skip — running predict_v2 against stale upc_entries
                    # would silently corrupt the prediction.
                    synth_path = synthesize_upc_from_entries(date_str, track)
                    if synth_path is None:
                        print(f"  Skipping {date_str} / {track}: no entries "
                              f"data found, can't synthesize upc_entries. "
                              f"Backfill entries CSVs and re-run.")
                        continue

                    print(f"  Running predict_v2: {date_str} / {track} ...",
                          flush=True)
                    ok = run_predict_for_date(
                        date_str, track,
                        verbose=args.verbose,
                        include_tipmeerkat=args.include_tipmeerkat,
                    )
                    if ok:
                        all_dates_run.append(date_str)

        # Optional: refit on full data so production JSONs are clean
        if not args.skip_final_retrain:
            print()
            print(f"Final step: retraining on FULL dataset so production JSONs")
            print(f"reflect all data (not just last-holdout subset)...")
            run_regression_training(skip_dates=None, verbose=args.verbose)
            print(f"  Done. JSONs in {DATA_PATH} are back to production state.")
        else:
            print()
            print(f"  ⚠ --skip-final-retrain set. Production JSONs currently reflect")
            print(f"    the LAST holdout's training set (missing {chunks[-1]}).")
            print(f"    Run `python regression_analysis.py` before using predict_v2 live.")
    finally:
        # Always restore the original upc_entries snapshots, even if
        # walk-forward crashed mid-loop. Otherwise the user's live
        # predict_v2 runs would read whatever synthetic snapshot we
        # last wrote.
        if backups:
            for track, (orig, bak) in backups.items():
                restore_upc(orig, bak)
            print(f"\nRestored upc_entries for {len(backups)} track(s) "
                  f"to their pre-backtest state.")

    # Collect all out-of-sample predictions into one summary
    print()
    print(f"Collecting holdout predictions...")
    summary = collect_predictions(set(all_dates_run))
    if summary.empty:
        print(f"  No predictions.csv files found. Check data/predictions/.")
        return

    out_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"  Wrote {len(summary)} rows across "
          f"{summary[['date', 'track', 'race']].drop_duplicates().shape[0]} races "
          f"to {out_path}")
    print()
    print(f"Next: run backtest_kfold.py on walk_forward_summary.csv for")
    print(f"honest ROI numbers using the real predict_v2 bet logic.")


if __name__ == '__main__':
    main()
    