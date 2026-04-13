"""
send_claude.py

Send chopped race PDFs to Claude in parallel. Each race gets TWO prompts:
  1. Estimate mean Beyer speed figure per horse
  2. Estimate win probability per horse

Usage:
    python send_claude.py --date 2026-04-05 --track gulfstream
    python send_claude.py --date 2026-04-08 --track keeneland --model claude-sonnet-4-20250514

Reads from:  {DATA_PATH}/chopped_forms/{date}/{track}/race_*.pdf
Writes to:   {DATA_PATH}/ai_picks/{date}/{track}/claude.csv

Requires: .env with CLAUDE_API_KEY and DATA_PATH
"""

import asyncio
import argparse
import base64
import os
import re
import csv
import time
from pathlib import Path

from dotenv import load_dotenv
from anthropic import AsyncAnthropic

# Load .env from repo root (one level up from scripts/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = os.environ['DATA_PATH']


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

Given DRF past performance data for a single race, estimate each horse's probability of winning.

RESPOND WITH ONLY CSV DATA. No headers, no markdown, no explanations, no commentary, no asterisks.
Each line must be exactly: horse_number,win_probability
where horse_number is the program number (integer) and win_probability is a decimal (e.g. 0.15).

Example of a CORRECT response (the entire response should look like this):
1,0.15
2,0.08
3,0.25
4,0.52

DO NOT include horse names, analysis, bold text, or any text besides the CSV rows.

Rules:
- Probabilities should sum to approximately 1.0 across all horses in the race
- Use decimal format (0.15 not 15%)
- Include ALL horses in the race (main body + Also Eligibles)
- For first-time starters, consider pedigree, trainer debut stats, sale price, workouts
- Factor in pace scenario: who benefits from the expected shape of the race?
- Consider jockey/trainer win rates at this track, distance, surface
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


async def call_claude(client: AsyncAnthropic, race_num: int, pdf_bytes: bytes,
                      system: str, model: str, label: str,
                      max_retries: int = 3) -> tuple[int, str, str]:
    """Send one race PDF to Claude, return (race_num, label, csv_text)."""
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode('utf-8')

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
            },
            {
                "type": "text",
                "text": f"Here is the DRF past performance PDF for Race {race_num}. "
                        f"Provide your estimates for each horse.",
            },
        ],
    }]

    for attempt in range(max_retries):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=2048,
                system=system,
                messages=messages,
            )
            return (race_num, label, resp.content[0].text.strip())
        except Exception as e:
            wait = 2 ** attempt
            print(f"  R{race_num} ({label}) attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

    print(f"  R{race_num} ({label}) FAILED after {max_retries} attempts")
    return (race_num, label, "")


def parse_csv_response(raw: str) -> list[tuple[str, float]]:
    """Parse LLM CSV response into [(horse_number, value), ...].
    Strictly validates: horse_number must be a simple integer or 1A/1B style."""
    # Strip markdown fences and any preamble
    text = re.sub(r'```(?:csv)?\n?', '', raw).strip()
    lines = text.strip().split('\n')

    # Skip header if present
    if lines and any(kw in lines[0].lower() for kw in ['horse', 'beyer', 'prob', 'number', 'win']):
        lines = lines[1:]

    results = []
    for line in lines:
        line = line.strip()
        if not line or '**' in line or ':' in line:
            continue  # skip markdown or explanatory text
        parts = line.split(',')
        if len(parts) >= 2:
            horse = parts[0].strip()
            # Horse number must be a simple integer or entry-style like "1A"
            if not re.match(r'^\d+[A-Za-z]?$', horse):
                continue
            try:
                val = float(parts[1].strip())
                results.append((horse, val))
            except ValueError:
                continue
    return results


async def main():
    parser = argparse.ArgumentParser(description='Send race PDFs to Claude for Beyer + win prob estimates')
    parser.add_argument('--date', type=str, required=True, help='Date folder (e.g. 2026-04-05)')
    parser.add_argument('--track', type=str, required=True, help='Track folder (e.g. keeneland)')
    parser.add_argument('--model', type=str, default='claude-opus-4-6', help='Claude model')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')
    args = parser.parse_args()

    # Paths from DATA_PATH
    chopped_dir = os.path.join(DATA_PATH, 'chopped_forms', args.date, args.track)
    out_dir = os.path.join(DATA_PATH, 'ai_picks', args.date, args.track)

    # Load race PDFs
    races = load_race_pdfs(chopped_dir)

    if not races:
        print(f"No race PDFs found in {chopped_dir}")
        return

    n = len(races)
    total_mb = sum(len(b) for b in races.values()) / 1024 / 1024
    print(f"{n} races loaded from {chopped_dir} ({total_mb:.1f} MB)")
    print(f"Model: {args.model}")

    # Load track conditions from historical_meta.pkl
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
            if track_meta.get('temperature'):
                print(f"  Temp: {track_meta['temperature']}, Wind: {track_meta.get('wind', '?')}")
        else:
            print(f"\n⚠  No track conditions found for {args.date} / {args.track}")
            print(f"   Run: python track_condition.py --track {args.track} --date {args.date}")
            if not args.yes:
                confirm = input("   Continue without track conditions? [y/N] ")
                if confirm.lower() != 'y':
                    print("Aborted.")
                    return
    else:
        print(f"\n⚠  No historical_meta.pkl found at {meta_path}")
        if not args.yes:
            confirm = input("   Continue without track conditions? [y/N] ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return

    # Build system prompts (inject track conditions if available)
    beyer_system = BEYER_SYSTEM
    winprob_system = WINPROB_SYSTEM
    if track_prompt:
        beyer_system = BEYER_SYSTEM + "\n\n" + track_prompt
        winprob_system = WINPROB_SYSTEM + "\n\n" + track_prompt

    # Load race PDFs
    races = load_race_pdfs(chopped_dir)

    if not races:
        print(f"No race PDFs found in {chopped_dir}")
        return

    n = len(races)
    total_mb = sum(len(b) for b in races.values()) / 1024 / 1024
    print(f"{n} races loaded from {chopped_dir} ({total_mb:.1f} MB)")
    print(f"Model: {args.model}")
    print(f"Sending {n * 2} requests ({n} beyer + {n} win prob)...")

    # Build all tasks: 2 per race
    client = AsyncAnthropic(api_key=os.environ['CLAUDE_API_KEY'])
    tasks = []
    for rn, pdf_bytes in sorted(races.items()):
        tasks.append(call_claude(client, rn, pdf_bytes, beyer_system, args.model, 'beyer'))
        tasks.append(call_claude(client, rn, pdf_bytes, winprob_system, args.model, 'winprob'))

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"All {len(results)} responses in {elapsed:.1f}s")

    # Parse into dicts: {race_num: {horse: value}}
    beyer_data = {}
    winprob_data = {}

    for race_num, label, raw_text in results:
        parsed = parse_csv_response(raw_text)
        if parsed:
            print(f"  R{race_num:>2} {label:>7}: {len(parsed)} horses")
            d = {horse: val for horse, val in parsed}
            if label == 'beyer':
                beyer_data[race_num] = d
            else:
                winprob_data[race_num] = d
        else:
            print(f"  R{race_num:>2} {label:>7}: *** PARSE FAILED ***")
            if raw_text:
                print(f"    First 200 chars: {raw_text[:200]}")

    # Diagnose disagreements between beyer and winprob horse lists
    print(f"\nHorse list disagreements:")
    any_disagreement = False
    for race_num in sorted(set(list(beyer_data.keys()) + list(winprob_data.keys()))):
        b_horses = set(beyer_data.get(race_num, {}).keys())
        w_horses = set(winprob_data.get(race_num, {}).keys())
        if b_horses != w_horses:
            any_disagreement = True
            beyer_only = b_horses - w_horses
            winprob_only = w_horses - b_horses
            if beyer_only:
                print(f"  R{race_num:>2} beyer only:   {sorted(beyer_only, key=lambda x: (len(x), x))}")
            if winprob_only:
                print(f"  R{race_num:>2} winprob only: {sorted(winprob_only, key=lambda x: (len(x), x))}")
    if not any_disagreement:
        print("  None — perfect agreement!")

    # Merge into rows
    all_horses = set()
    for race_num in sorted(set(list(beyer_data.keys()) + list(winprob_data.keys()))):
        beyers = beyer_data.get(race_num, {})
        probs = winprob_data.get(race_num, {})
        for h in set(list(beyers.keys()) + list(probs.keys())):
            all_horses.add((race_num, h))

    rows = []
    for race_num, horse in sorted(all_horses):
        beyer = beyer_data.get(race_num, {}).get(horse, '')
        prob = winprob_data.get(race_num, {}).get(horse, '')
        if beyer != '':
            beyer = int(beyer)
        if prob != '':
            prob = round(float(prob), 4)
        rows.append([race_num, horse, beyer, prob, args.track, args.date])

    # Save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'claude.csv')

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['race', 'horse_number', 'claude_expected_beyer', 'claude_win_prob', 'track', 'date'])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    asyncio.run(main())