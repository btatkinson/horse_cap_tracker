"""
run_ai_pipeline.py

Combined pipeline: chop PDF → fetch track conditions → send to Claude + Gemini + OpenAI.
Prompts for confirmation at each step to avoid wasting API calls.

Usage:
    python run_ai_pipeline.py --track keeneland
    python run_ai_pipeline.py --track keeneland --date 2026-04-09
    python run_ai_pipeline.py --track keeneland --pdf GP--04-09-2026.pdf
    python run_ai_pipeline.py --track keeneland --skip-gemini
    python run_ai_pipeline.py --track keeneland --skip-openai

Steps:
    1. Locate PDF in data/forms/ (auto or --pdf)
    2. Detect date + track (confirm with user)
    3. Chop into per-race PDFs
    4. Fetch track conditions
    5. Send to Claude for Beyer + win prob
    6. Send to Gemini for Beyer + win prob
    7. Send to OpenAI for Beyer + win prob
"""

import os, sys, subprocess, argparse
from pathlib import Path
from datetime import date, datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = Path(os.environ['DATA_PATH'])
SCRIPT_DIR = Path(__file__).parent

# Provider config — matches backfill_ai.py
PROVIDERS = {
    'claude': {
        'script': 'send_claude.py',
        'csv_name': 'claude.csv',
        'default_model': 'claude-opus-4-7',
    },
    'gemini': {
        'script': 'send_gemini.py',
        'csv_name': 'gemini.csv',
        'default_model': 'gemini-3.1-pro-preview',
    },
    'openai': {
        'script': 'send_openai.py',
        'csv_name': 'openai.csv',
        'default_model': 'gpt-5.4',
    },
}


def confirm(prompt, default='y'):
    """Prompt user for y/n confirmation."""
    suffix = ' [Y/n] ' if default == 'y' else ' [y/N] '
    resp = input(prompt + suffix).strip().lower()
    if not resp:
        return default == 'y'
    return resp in ('y', 'yes')


def run_script(name, args_list):
    """Run a Python script with args, return True if successful."""
    cmd = [sys.executable, str(SCRIPT_DIR / name)] + args_list
    print(f"\n{'─'*60}")
    print(f"Running: {name} {' '.join(args_list)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def find_latest_pdf():
    """Find the most recent PDF in data/forms/."""
    forms_dir = DATA_PATH / 'forms'
    if not forms_dir.exists():
        return None
    pdfs = sorted(forms_dir.glob('*.pdf'), key=lambda p: p.stat().st_mtime, reverse=True)
    return pdfs[0] if pdfs else None


def main():
    parser = argparse.ArgumentParser(description='AI handicapping pipeline: chop → conditions → Claude + Gemini + OpenAI')
    parser.add_argument('--track', type=str, required=True, help='Track name (e.g. keeneland)')
    parser.add_argument('--date', type=str, default=None, help='Date YYYY-MM-DD (auto-detected from PDF)')
    parser.add_argument('--pdf', type=str, default=None, help='PDF filename in data/forms/')
    parser.add_argument('--claude-model', type=str, default=PROVIDERS['claude']['default_model'], help='Claude model')
    parser.add_argument('--gemini-model', type=str, default=PROVIDERS['gemini']['default_model'], help='Gemini model')
    parser.add_argument('--openai-model', type=str, default=PROVIDERS['openai']['default_model'], help='OpenAI model')
    parser.add_argument('--skip-claude', action='store_true', help='Skip Claude API call')
    parser.add_argument('--skip-gemini', action='store_true', help='Skip Gemini API call')
    parser.add_argument('--skip-openai', action='store_true', help='Skip OpenAI API call')
    parser.add_argument('--skip-combine', action='store_true',
                        help='Skip the combine_ai_picks step (you can run it later manually)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"AI HANDICAPPING PIPELINE")
    print(f"{'='*60}")

    # ── Step 1: Locate PDF ───────────────────────────────────────────
    if args.pdf:
        pdf_path = DATA_PATH / 'forms' / args.pdf
        if not pdf_path.exists():
            pdf_path = Path(args.pdf)
    else:
        pdf_path = find_latest_pdf()

    if not pdf_path or not pdf_path.exists():
        print(f"\nNo PDF found. Place DRF PP PDF in {DATA_PATH / 'forms'}/")
        print(f"Or specify: --pdf filename.pdf")
        return

    print(f"\n  PDF:   {pdf_path.name}")
    print(f"  Track: {args.track}")

    # ── Step 2: Detect/confirm date ──────────────────────────────────
    if args.date:
        date_str = args.date
    else:
        # Auto-detect from filename using chop_form logic
        import re
        fn = pdf_path.name
        m = re.search(r'(\d{2})-(\d{2})-(\d{4})', fn)
        if m:
            date_str = f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
        else:
            m = re.search(r'(\d{1,2})-(\d{1,2})-(\d{2})(?=-|$)', fn)
            if m:
                yr = int(m.group(3))
                yr = yr + 2000 if yr < 100 else yr
                date_str = f"{yr}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
            else:
                date_str = date.today().strftime('%Y-%m-%d')

    print(f"  Date:  {date_str}")

    # Check if chopped forms already exist
    chopped_dir = DATA_PATH / 'chopped_forms' / date_str / args.track
    already_chopped = chopped_dir.exists() and list(chopped_dir.glob('race_*.pdf'))

    # Check if AI picks already exist
    ai_dir = DATA_PATH / 'ai_picks' / date_str / args.track
    claude_exists = (ai_dir / PROVIDERS['claude']['csv_name']).exists()
    gemini_exists = (ai_dir / PROVIDERS['gemini']['csv_name']).exists()
    openai_exists = (ai_dir / PROVIDERS['openai']['csv_name']).exists()

    print(f"\n  Status:")
    print(f"    Chopped forms: {'✓ exists' if already_chopped else '✗ not yet'}")
    print(f"    Claude picks:  {'✓ exists' if claude_exists else '✗ not yet'}")
    print(f"    Gemini picks:  {'✓ exists' if gemini_exists else '✗ not yet'}")
    print(f"    OpenAI picks:  {'✓ exists' if openai_exists else '✗ not yet'}")

    # ── Confirm before proceeding ────────────────────────────────────
    print(f"\n  This will:")
    steps = []
    if not already_chopped:
        steps.append(f"    1. Chop {pdf_path.name} into per-race PDFs")
    else:
        steps.append(f"    1. (skip chop — already done)")
    steps.append(f"    2. Fetch track conditions for {args.track}")
    if not args.skip_claude:
        if claude_exists:
            steps.append(f"    3. Send to Claude ({args.claude_model}) — ⚠ OVERWRITING existing")
        else:
            steps.append(f"    3. Send to Claude ({args.claude_model})")
    else:
        steps.append(f"    3. (skip Claude)")
    if not args.skip_gemini:
        if gemini_exists:
            steps.append(f"    4. Send to Gemini ({args.gemini_model}) — ⚠ OVERWRITING existing")
        else:
            steps.append(f"    4. Send to Gemini ({args.gemini_model})")
    else:
        steps.append(f"    4. (skip Gemini)")
    if not args.skip_openai:
        if openai_exists:
            steps.append(f"    5. Send to OpenAI ({args.openai_model}) — ⚠ OVERWRITING existing")
        else:
            steps.append(f"    5. Send to OpenAI ({args.openai_model})")
    else:
        steps.append(f"    5. (skip OpenAI)")
    if not args.skip_combine:
        combined_exists = (ai_dir / 'ai_combined.csv').exists()
        if combined_exists:
            steps.append(f"    6. Combine providers + impute → ai_combined.csv (⚠ overwriting existing)")
        else:
            steps.append(f"    6. Combine providers + impute → ai_combined.csv")
    else:
        steps.append(f"    6. (skip combine)")

    for s in steps:
        print(s)

    if not confirm(f"\n  Proceed?"):
        print("  Aborted.")
        return

    # ── Step 3: Chop PDF ─────────────────────────────────────────────
    if not already_chopped:
        ok = run_script('chop_form.py', [
            '--pdf', pdf_path.name,
            '--date', date_str,
            '--track', args.track,
        ])
        if not ok:
            print("\n  ✗ Chop failed. Aborting.")
            return
    else:
        n = len(list(chopped_dir.glob('race_*.pdf')))
        print(f"\n  Skipping chop — {n} race PDFs already in {chopped_dir}/")

    # ── Step 4: Track conditions ─────────────────────────────────────
    run_script('track_condition.py', [
        '--track', args.track,
        '--date', date_str,
    ])
    # Don't abort on failure — conditions are optional

    # ── Step 5: Send to Claude ───────────────────────────────────────
    if not args.skip_claude:
        ok = run_script(PROVIDERS['claude']['script'], [
            '--date', date_str,
            '--track', args.track,
            '--model', args.claude_model,
            '-y',
        ])
        if not ok:
            print("\n  ✗ Claude failed.")
            remaining = []
            if not args.skip_gemini:
                remaining.append('Gemini')
            if not args.skip_openai:
                remaining.append('OpenAI')
            if remaining:
                if not confirm(f"  Continue to {' + '.join(remaining)}?"):
                    return

    # ── Step 6: Send to Gemini ───────────────────────────────────────
    if not args.skip_gemini:
        ok = run_script(PROVIDERS['gemini']['script'], [
            '--date', date_str,
            '--track', args.track,
            '--model', args.gemini_model,
            '-y',
        ])
        if not ok:
            print("\n  ✗ Gemini failed.")
            if not args.skip_openai:
                if not confirm("  Continue to OpenAI?"):
                    return

    # ── Step 7: Send to OpenAI ───────────────────────────────────────
    if not args.skip_openai:
        ok = run_script(PROVIDERS['openai']['script'], [
            '--date', date_str,
            '--track', args.track,
            '--model', args.openai_model,
            '-y',
        ])
        if not ok:
            print("\n  ✗ OpenAI failed.")

    # ── Step 8: Combine providers + impute ────────────────────────────
    # combine_ai_picks merges claude/gemini/openai CSVs and runs sklearn
    # IterativeImputer to fill any per-provider gaps (e.g. gemini missing
    # a beyer for one horse), then writes ai_combined.csv. predict_v2 and
    # regression_analysis read ai_combined.csv directly — running this
    # step here means downstream commands won't fail with "ai_combined.csv
    # not found." --force overwrites any stale file from a previous run.
    if not args.skip_combine:
        # Verify at least one provider succeeded — combine needs at least
        # one input CSV. If none exist, skip and warn.
        any_provider_csv = (
            (ai_dir / PROVIDERS['claude']['csv_name']).exists()
            or (ai_dir / PROVIDERS['gemini']['csv_name']).exists()
            or (ai_dir / PROVIDERS['openai']['csv_name']).exists()
        )
        if not any_provider_csv:
            print(f"\n  ✗ Skipping combine — no provider CSVs exist in {ai_dir}/")
        else:
            ok = run_script('combine_ai_picks.py', [
                '--date', date_str,
                '--track', args.track,
                '--force',
            ])
            if not ok:
                print(f"\n  ✗ combine_ai_picks failed. You can re-run it later:")
                print(f"      python combine_ai_picks.py --date {date_str} "
                      f"--track {args.track} --force")

    # ── Done ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  AI picks: {DATA_PATH / 'ai_picks' / date_str / args.track}/")
    print(f"\n  Next steps:")
    if args.skip_combine:
        print(f"    python combine_ai_picks.py --date {date_str} --track {args.track} --force")
    print(f"    python scrape_experts.py --track {args.track} --date {date_str}")
    print(f"    python send_experts.py --date {date_str} --track {args.track}")
    print(f"    python predict_races.py --date {date_str} --track {args.track}")


if __name__ == '__main__':
    main()