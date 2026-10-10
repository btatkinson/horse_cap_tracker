"""
send_openai.py

Send chopped race PDFs to OpenAI GPT. Each race gets THREE prompts:
  1. Estimate mean Beyer speed figure per horse
  2. Estimate win probability per horse
  3. Predict closing tote odds per horse

Usage:
    python send_openai.py --date 2026-04-12 --track keeneland
    python send_openai.py --date 2026-04-12 --track keeneland --model gpt-5.4

Reads from:  {DATA_PATH}/chopped_forms/{date}/{track}/race_*.pdf
Writes to:   {DATA_PATH}/ai_picks/{date}/{track}/openai.csv

Requires: .env with OPENAI_API_KEY and DATA_PATH
"""

import asyncio
import argparse
import base64
import os
import random
import re
import csv
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = os.environ['DATA_PATH']

# Default max concurrent requests to OpenAI API.
# Tuned conservatively: large PDFs on Derby Day will trip TPM limits when
# many requests are in flight. Bump up if your tier allows it.
DEFAULT_CONCURRENCY = 2


BEYER_SYSTEM = """You are an expert horse racing handicapper estimating Beyer Speed Figures.

Given DRF past performance data for a single race, estimate a Beyer speed figure for each horse.

RESPOND WITH ONLY CSV DATA. No headers, no markdown, no explanations, no commentary, no asterisks.
Each line must be exactly: horse_number,expected_beyer
where horse_number is the program number (integer) and expected_beyer is your estimate (integer).

Example of a CORRECT response (the entire response should look like this):
1,82
2,75
3,88
4,70

DO NOT include horse names, analysis, bold text, or any text besides the CSV rows.

Rules:
- Use integer Beyer estimates
- Include ALL horses in the race (main body + Also Eligibles)
- For first-time starters, estimate based on pedigree, trainer debut stats, sale price, workouts
- Estimate what each horse will ACTUALLY RUN today, not their best past figure
- Consider track/surface/distance changes, form cycle, layoffs, equipment changes
"""

WINPROB_SYSTEM = """You are an expert horse racing handicapper estimating win probabilities.

Given DRF past performance data for a single race, estimate the win probability for each horse.

RESPOND WITH ONLY CSV DATA. No headers, no markdown, no explanations, no commentary, no asterisks.
Each line must be exactly: horse_number,win_prob
where horse_number is the program number (integer) and win_prob is a decimal between 0 and 1.
Probabilities MUST sum to approximately 1.0.

Example of a CORRECT response:
1,0.25
2,0.10
3,0.35
4,0.30

DO NOT include horse names, analysis, bold text, or any text besides the CSV rows.

Rules:
- Probabilities must sum to ~1.0
- Include ALL horses in the race
- For first-time starters, consider pedigree, trainer debut stats, sale price, workouts
- Factor in pace scenario: who benefits from the expected shape of the race?
- Consider jockey/trainer win rates at this track, distance, surface
"""

CLOSING_ODDS_SYSTEM = """You are an expert horse racing analyst predicting what the FINAL TOTE BOARD ODDS will be at post time.

This is NOT about who will win. This is about predicting how the BETTING PUBLIC will wager.
Consider: public biases toward favorites, trainers with big followings, popular jockeys,
flashy workouts, hot barns, Wesley Ward 2-year-olds, well-known sires, recent hype,
and which horses the casual bettor will overlook.

Given DRF past performance data for a single race, predict the closing tote odds for each horse.

RESPOND WITH ONLY CSV DATA. No headers, no markdown, no explanations.
Each line must be exactly: horse_number,predicted_odds
where horse_number is the program number (integer) and predicted_odds is the decimal odds
(e.g. 3.0 for 3/1, 0.5 for 1/2, 15.0 for 15/1).

Example of a CORRECT response:
1,3.0
2,8.0
3,1.5
4,12.0

Rules:
- These are YOUR PREDICTIONS of what the public will bet, not what the horse deserves
- Morning line is a starting point but the public often disagrees significantly
- Horses with big-name trainers/jockeys tend to be overbet (lower odds than ML)
- First-time starters from expensive sires with fast workouts attract action
- Horses from obscure barns with poor jockeys tend to drift higher than ML
- Consider the "herd mentality" — if one horse is obviously the best, it gets pounded
"""


def load_race_pdfs(chopped_dir: str) -> dict[int, bytes]:
    """Load race_*.pdf files, return {race_number: pdf_bytes}."""
    races = {}
    for f in sorted(Path(chopped_dir).glob("race_*.pdf")):
        match = re.search(r'race_(\d+)', f.name)
        if match:
            rn = int(match.group(1))
            races[rn] = f.read_bytes()
    return races


async def call_openai(race_num: int, pdf_bytes: bytes,
                      system: str, model: str, label: str,
                      semaphore: asyncio.Semaphore,
                      max_retries: int = 5) -> tuple[int, str, str]:
    """Send one race PDF to OpenAI, return (race_num, label, csv_text).

    Uses a semaphore to limit concurrent requests. On a 429, prefers the
    server's retry-after header; otherwise waits ~60s (since TPM is a
    per-minute bucket). Adds jitter so workers don't wake in lockstep.
    """
    async with semaphore:
        # max_retries=0 disables the SDK's built-in retries so they don't
        # silently fight ours and burn the token budget.
        from openai import AsyncOpenAI, RateLimitError
        client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'), max_retries=0)

        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode('utf-8')

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": f"race_{race_num}.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": f"Here is the DRF past performance PDF for Race {race_num}. "
                            f"Provide your estimates for each horse.",
                },
            ]},
        ]

        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return (race_num, label, resp.choices[0].message.content.strip())
            except RateLimitError as e:
                retry_after = None
                try:
                    retry_after = e.response.headers.get("retry-after")
                except AttributeError:
                    pass
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 60.0
                else:
                    wait = 60.0
                wait += random.uniform(0, 5)
                print(f"  R{race_num} ({label}) rate limited "
                      f"(attempt {attempt+1}/{max_retries}). Waiting {wait:.1f}s...")
                await asyncio.sleep(wait)
            except Exception as e:
                wait = min(2 ** attempt, 30) + random.uniform(0, 1)
                print(f"  R{race_num} ({label}) attempt {attempt+1} failed: {e}. "
                      f"Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)

        print(f"  R{race_num} ({label}) FAILED after {max_retries} attempts")
        return (race_num, label, "")


def parse_csv_response(raw: str) -> list[tuple[str, float]]:
    """Parse LLM CSV response into [(horse_number, value), ...]."""
    text = re.sub(r'```(?:csv)?\n?', '', raw).strip()
    lines = text.strip().split('\n')

    if lines and any(kw in lines[0].lower() for kw in ['horse', 'beyer', 'prob', 'number', 'win', 'odds']):
        lines = lines[1:]

    results = []
    for line in lines:
        line = line.strip()
        if not line or '**' in line or ':' in line:
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            horse = parts[0].strip()
            if not re.match(r'^\d+[A-Za-z]?$', horse):
                continue
            try:
                val = float(parts[1].strip())
                results.append((horse, val))
            except ValueError:
                continue
    return results


async def main():
    parser = argparse.ArgumentParser(description='Send race PDFs to OpenAI for estimates')
    parser.add_argument('--date', type=str, required=True, help='Date folder (e.g. 2026-04-12)')
    parser.add_argument('--track', type=str, required=True, help='Track folder (e.g. keeneland)')
    parser.add_argument('--model', type=str, default='gpt-5.4', help='OpenAI model')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--skip-odds', action='store_true', help='Skip closing odds prediction')
    parser.add_argument('--closing-only', action='store_true',
                        help='Only send closing odds prompt, merge into existing CSV')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Max concurrent API requests (default {DEFAULT_CONCURRENCY})')
    args = parser.parse_args()

    chopped_dir = os.path.join(DATA_PATH, 'chopped_forms', args.date, args.track)
    out_dir = os.path.join(DATA_PATH, 'ai_picks', args.date, args.track)
    out_path = os.path.join(out_dir, 'openai.csv')

    races = load_race_pdfs(chopped_dir)
    if not races:
        print(f"No race PDFs found in {chopped_dir}")
        return

    n = len(races)
    total_mb = sum(len(b) for b in races.values()) / 1024 / 1024
    print(f"{n} races loaded from {chopped_dir} ({total_mb:.1f} MB)")
    print(f"Model: {args.model}")

    # Load track conditions
    meta_path = os.path.join(DATA_PATH, 'historical_meta.pkl')
    track_prompt = ""
    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        track_meta = meta.get(args.date, {}).get(args.track)
        if track_meta:
            track_prompt = track_meta.get('prompt', '')
            print(f"\nTrack conditions ({track_meta.get('source', '?')}):")
            print(f"  Dirt: {track_meta.get('dirt', '?')}")
            print(f"  Turf: {track_meta.get('turf', '?')}")
        else:
            print(f"\n⚠  No track conditions found for {args.date} / {args.track}")
            if not args.yes:
                confirm = input("   Continue without track conditions? [y/N] ")
                if confirm.lower() != 'y':
                    return

    # Build system prompts
    beyer_system = BEYER_SYSTEM
    winprob_system = WINPROB_SYSTEM
    closing_system = CLOSING_ODDS_SYSTEM
    if track_prompt:
        beyer_system += "\n\n" + track_prompt
        winprob_system += "\n\n" + track_prompt
        closing_system += "\n\n" + track_prompt

    n_queries = n * 3 if not args.skip_odds else n * 2
    if args.closing_only:
        n_queries = n

    if args.closing_only:
        print(f"Sending {n} requests (closing odds only, max {args.concurrency} at a time)...")
    else:
        print(f"Sending {n_queries} requests ({n} beyer + {n} win prob"
              f"{f' + {n} closing odds' if not args.skip_odds else ''}, "
              f"max {args.concurrency} at a time)...")

    if not args.yes:
        confirm = input("Proceed? [Y/n] ")
        if confirm.lower() == 'n':
            print("Aborted.")
            return

    # Build tasks
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for rn, pdf_bytes in sorted(races.items()):
        if args.closing_only:
            tasks.append(call_openai(rn, pdf_bytes, closing_system, args.model, 'closing', semaphore))
        else:
            tasks.append(call_openai(rn, pdf_bytes, beyer_system, args.model, 'beyer', semaphore))
            tasks.append(call_openai(rn, pdf_bytes, winprob_system, args.model, 'winprob', semaphore))
            if not args.skip_odds:
                tasks.append(call_openai(rn, pdf_bytes, closing_system, args.model, 'closing', semaphore))

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"All {len(results)} responses in {elapsed:.1f}s")

    # Parse results
    beyer_data = {}
    winprob_data = {}
    closing_data = {}

    for race_num, label, raw_text in results:
        parsed = parse_csv_response(raw_text)
        if parsed:
            print(f"  R{race_num:>2} {label:>7}: {len(parsed)} horses")
            d = {horse: val for horse, val in parsed}
            if label == 'beyer':
                beyer_data[race_num] = d
            elif label == 'winprob':
                winprob_data[race_num] = d
            elif label == 'closing':
                closing_data[race_num] = d
        else:
            print(f"  R{race_num:>2} {label:>7}: *** PARSE FAILED ***")
            if raw_text:
                print(f"    First 200 chars: {raw_text[:200]}")

    if args.closing_only:
        # Merge closing odds into existing CSV
        import pandas as pd
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
        else:
            print(f"  No existing openai.csv — run without --closing-only first")
            return

        closing_rows = []
        for race_num, horses in closing_data.items():
            for horse, odds in horses.items():
                closing_prob = round(1.0 / (odds + 1), 4) if odds > 0 else 0
                closing_rows.append({
                    'race': int(race_num),
                    'horse_number': int(horse) if horse.isdigit() else horse,
                    'openai_closing_prob': closing_prob,
                })

        if closing_rows:
            closing_df = pd.DataFrame(closing_rows)
            existing['horse_number'] = pd.to_numeric(existing['horse_number'], errors='coerce')
            closing_df['horse_number'] = pd.to_numeric(closing_df['horse_number'], errors='coerce')

            if 'openai_closing_prob' in existing.columns:
                existing = existing.drop(columns=['openai_closing_prob'])

            merged = existing.merge(closing_df, on=['race', 'horse_number'], how='left')
            merged.to_csv(out_path, index=False)
            n_filled = merged['openai_closing_prob'].notna().sum()
            print(f"\nMerged closing odds into {out_path} ({n_filled}/{len(merged)} horses)")
        else:
            print("  No closing data parsed")
        return

    # Merge into rows
    all_horses = set()
    for race_num in sorted(set(list(beyer_data.keys()) + list(winprob_data.keys()) +
                               list(closing_data.keys()))):
        for d in [beyer_data, winprob_data, closing_data]:
            for h in d.get(race_num, {}).keys():
                all_horses.add((race_num, h))

    rows = []
    for race_num, horse in sorted(all_horses):
        beyer = beyer_data.get(race_num, {}).get(horse, '')
        prob = winprob_data.get(race_num, {}).get(horse, '')
        closing = closing_data.get(race_num, {}).get(horse, '')

        if beyer != '':
            beyer = int(beyer)
        if prob != '':
            prob = round(float(prob), 4)
        if closing != '':
            # Convert decimal odds to implied probability
            odds_val = float(closing)
            closing_prob = round(1.0 / (odds_val + 1), 4) if odds_val > 0 else ''

        rows.append([race_num, horse, beyer, prob, closing_prob if closing != '' else '',
                     args.track, args.date])

    # Save
    os.makedirs(out_dir, exist_ok=True)

    header = ['race', 'horse_number', 'openai_expected_beyer', 'openai_win_prob',
              'openai_closing_prob', 'track', 'date']

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_path}")

    # Show comparison: where do predicted closing odds differ most from win prob?
    if closing_data:
        print(f"\nBiggest discrepancies (closing odds vs win prob):")
        discrepancies = []
        for race_num, horse in sorted(all_horses):
            wp = winprob_data.get(race_num, {}).get(horse)
            cl = closing_data.get(race_num, {}).get(horse)
            if wp and cl and cl > 0:
                closing_impl = 1.0 / (cl + 1)
                diff = wp - closing_impl  # positive = model thinks better than public
                discrepancies.append((race_num, horse, wp, closing_impl, diff))

        discrepancies.sort(key=lambda x: abs(x[4]), reverse=True)
        for rn, h, wp, ci, diff in discrepancies[:10]:
            label = "OVERBET" if diff < 0 else "UNDERBET"
            print(f"  R{rn} #{h}: model={wp:.1%} public={ci:.1%} → {label} by {abs(diff):.1%}")


if __name__ == '__main__':
    asyncio.run(main())