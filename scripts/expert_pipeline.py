"""
run_expert_pipeline.py

Combined pipeline: scrape experts → send to LLM → generate predictions.
Skips already-processed experts automatically.

Usage:
    python run_expert_pipeline.py --track keeneland --date 2026-04-09
    python run_expert_pipeline.py --track keeneland    # defaults to today
    python run_expert_pipeline.py --track keeneland --skip-predict

Steps:
    1. Scrape expert picks from track website
    2. Send raw text to Claude for logit delta extraction
    3. Generate predictions (if AI picks exist)
"""

import os, sys, subprocess, argparse
from pathlib import Path
from datetime import date

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = Path(os.environ['DATA_PATH'])
SCRIPT_DIR = Path(__file__).parent


def run_script(name, args_list):
    """Run a Python script with args, return True if successful."""
    cmd = [sys.executable, str(SCRIPT_DIR / name)] + args_list
    print(f"\n{'─'*60}")
    print(f"Running: {name} {' '.join(args_list)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Expert picks pipeline: scrape → extract → predict')
    parser.add_argument('--track', type=str, required=True, help='Track name (e.g. keeneland)')
    parser.add_argument('--date', type=str, default=None, help='Date YYYY-MM-DD (default: today)')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514',
                        help='Claude model for extraction')
    parser.add_argument('--skip-predict', action='store_true', help='Skip prediction step')
    args = parser.parse_args()

    date_str = args.date or date.today().strftime('%Y-%m-%d')

    print(f"{'='*60}")
    print(f"EXPERT PICKS PIPELINE")
    print(f"{'='*60}")
    print(f"  Track: {args.track}")
    print(f"  Date:  {date_str}")

    # Status check
    raw_dir = DATA_PATH / 'raw_expert_picks' / date_str / args.track
    sent_dir = DATA_PATH / 'already_sent_eps' / date_str / args.track
    expert_dir = DATA_PATH / 'expert_picks' / date_str / args.track
    ai_dir = DATA_PATH / 'ai_picks' / date_str / args.track

    n_raw = len(list(raw_dir.glob('*.txt'))) if raw_dir.exists() else 0
    n_sent = len(list(sent_dir.glob('*.txt'))) if sent_dir.exists() else 0
    n_extracted = len(list(expert_dir.glob('*.csv'))) if expert_dir.exists() else 0
    has_ai = ai_dir.exists() and (
        (ai_dir / 'claude.csv').exists() or (ai_dir / 'gemini.csv').exists()
    )

    print(f"\n  Status:")
    print(f"    Raw expert text:    {n_raw} files")
    print(f"    Already sent:       {n_sent} files")
    print(f"    Extracted CSVs:     {n_extracted} files")
    print(f"    AI picks:           {'✓' if has_ai else '✗ (run run_ai_pipeline.py first)'}")

    # ── Step 0: Download today's entries (needed for name→number reference) ──
    run_script('download_upc.py', [
        '--track', args.track,
        '--date', date_str,
    ])

    # ── Step 1: Scrape ───────────────────────────────────────────────
    run_script('scrape_experts.py', [
        '--track', args.track,
        '--date', date_str,
    ])

    # ── Step 2: Send to LLM ─────────────────────────────────────────
    # Check if there are new files to send
    n_raw_after = len(list(raw_dir.glob('*.txt'))) if raw_dir.exists() else 0
    if n_raw_after > 0:
        run_script('send_experts.py', [
            '--date', date_str,
            '--track', args.track,
            '--model', args.model,
        ])
    else:
        print(f"\n  No new expert text to process (all already sent)")

    # ── Step 3: Predict ──────────────────────────────────────────────
    if not args.skip_predict and has_ai:
        run_script('predict_races.py', [
            '--date', date_str,
            '--track', args.track,
        ])
    elif not has_ai:
        print(f"\n  Skipping predictions — no AI picks yet")
        print(f"  Run: python run_ai_pipeline.py --track {args.track} --date {date_str}")

    # ── Done ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EXPERT PIPELINE COMPLETE")
    print(f"{'='*60}")
    n_final = len(list(expert_dir.glob('*.csv'))) if expert_dir.exists() else 0
    print(f"  {n_final} expert pick files in {expert_dir}/")
    if has_ai and not args.skip_predict:
        print(f"  Predictions: {DATA_PATH / 'predictions' / date_str / args.track}/")


if __name__ == '__main__':
    main()