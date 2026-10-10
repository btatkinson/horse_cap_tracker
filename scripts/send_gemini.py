"""
send_gemini.py

Send chopped race PDFs to Gemini in parallel. Each race gets THREE prompts:
  1. Estimate mean Beyer speed figure per horse
  2. Estimate win probability per horse
  3. Predict closing tote odds per horse

Usage:
    python send_gemini.py --date 2026-04-05 --track gulfstream
    python send_gemini.py --date 2026-04-08 --track keeneland --model gemini-3.1-pro-preview
    python send_gemini.py --date 2026-04-08 --track keeneland --concurrency 3
    python send_gemini.py --date 2026-04-08 --track keeneland --races 3,7

Reads from:  {DATA_PATH}/chopped_forms/{date}/{track}/race_*.pdf
Writes to:   {DATA_PATH}/ai_picks/{date}/{track}/gemini.csv

Requires: .env with GEMINI_API_KEY and DATA_PATH
"""

import asyncio
import argparse
import base64
import os
import re
import csv
import time
import pickle
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load .env from repo root (one level up from scripts/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = os.environ['DATA_PATH']

# Default max concurrent requests to Gemini API
DEFAULT_CONCURRENCY = 5


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


def load_race_pdfs(chopped_dir: str, only_races: set | None = None) -> dict:
    """Load race_*.pdf files, return {race_number: pdf_bytes}.

    If `only_races` is provided, only load PDFs whose race number is in
    that set. Used by --races for repair runs.
    """
    races = {}
    for f in sorted(Path(chopped_dir).glob("race_*.pdf")):
        match = re.search(r'race_(\d+)', f.name)
        if match:
            rn = int(match.group(1))
            if only_races is not None and rn not in only_races:
                continue
            races[rn] = f.read_bytes()
    return races


async def call_gemini(client: genai.Client, race_num: int, pdf_bytes: bytes,
                      system: str, model: str, label: str,
                      semaphore: asyncio.Semaphore,
                      max_retries: int = 3):
    """Send one race PDF to Gemini, return (race_num, label, csv_text).
    
    Uses a semaphore to limit concurrent requests.
    """
    async with semaphore:
        contents = [
            genai.types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            f"Here is the DRF past performance PDF for Race {race_num}. "
            f"Provide your estimates for each horse.",
        ]

        config = genai.types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=2048,
        )

        for attempt in range(max_retries):
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return (race_num, label, resp.text.strip())
            except Exception as e:
                wait = 2 ** attempt
                print(f"  R{race_num} ({label}) attempt {attempt+1} failed: {e}. "
                      f"Retrying in {wait}s...")
                await asyncio.sleep(wait)

        print(f"  R{race_num} ({label}) FAILED after {max_retries} attempts")
        return (race_num, label, "")


def parse_csv_response(raw: str):
    """Parse LLM CSV response into [(horse_number, value), ...].
    Strictly validates: horse_number must be a simple integer or 1A/1B style."""
    text = re.sub(r'```(?:csv)?\n?', '', raw).strip()
    lines = text.strip().split('\n')

    if lines and any(kw in lines[0].lower() for kw in ['horse', 'beyer', 'prob', 'number', 'win']):
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


def _parse_races_arg(races_str: str):
    """Parse '3,7' or '3' into {3, 7}. Used by --races."""
    out = set()
    for part in races_str.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            raise SystemExit(f"--races values must be integers; got '{part}'")
    return out


async def main():
    parser = argparse.ArgumentParser(description='Send race PDFs to Gemini for Beyer + win prob estimates')
    parser.add_argument('--date', type=str, required=True, help='Date folder (e.g. 2026-04-05)')
    parser.add_argument('--track', type=str, required=True, help='Track folder (e.g. keeneland)')
    parser.add_argument('--model', type=str, default='gemini-3.1-pro-preview', help='Gemini model')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--closing-only', action='store_true',
                        help='Only send closing odds prompt, merge into existing CSV')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Max concurrent API requests (default {DEFAULT_CONCURRENCY})')
    parser.add_argument('--races', type=str, default=None,
                        help='Comma-separated race numbers to (re)run, e.g. "3,7". '
                             'When set, only those races are sent; existing rows for '
                             'other races in gemini.csv are preserved unchanged. '
                             'Used by `backfill_ai.py --repair` to patch specific '
                             'races without re-running the whole card.')
    args = parser.parse_args()

    only_races = _parse_races_arg(args.races) if args.races else None

    chopped_dir = os.path.join(DATA_PATH, 'chopped_forms', args.date, args.track)
    out_dir = os.path.join(DATA_PATH, 'ai_picks', args.date, args.track)
    out_path = os.path.join(out_dir, 'gemini.csv')

    races = load_race_pdfs(chopped_dir, only_races=only_races)
    if not races:
        if only_races:
            print(f"No race PDFs found in {chopped_dir} matching --races={sorted(only_races)}")
            print(f"  (check that race_N.pdf exists for each requested race)")
        else:
            print(f"No race PDFs found in {chopped_dir}")
        return

    n = len(races)
    total_mb = sum(len(b) for b in races.values()) / 1024 / 1024
    print(f"{n} races loaded from {chopped_dir} ({total_mb:.1f} MB)")
    if only_races:
        print(f"  Filtered to --races={sorted(only_races)}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")

    # Load track conditions
    meta_path = os.path.join(DATA_PATH, 'historical_meta.pkl')
    track_prompt = ""
    if os.path.exists(meta_path):
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
            if not args.yes:
                confirm = input("   Continue without track conditions? [y/N] ")
                if confirm.lower() != 'y':
                    return
    else:
        print(f"\n⚠  No historical_meta.pkl found at {meta_path}")
        if not args.yes:
            confirm = input("   Continue without track conditions? [y/N] ")
            if confirm.lower() != 'y':
                return

    beyer_system = BEYER_SYSTEM
    winprob_system = WINPROB_SYSTEM
    closing_system = CLOSING_ODDS_SYSTEM
    if track_prompt:
        beyer_system += "\n\n" + track_prompt
        winprob_system += "\n\n" + track_prompt
        closing_system += "\n\n" + track_prompt

    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []

    if args.closing_only:
        print(f"Sending {n} requests (closing odds only, max {args.concurrency} at a time)...")
        for rn, pdf_bytes in sorted(races.items()):
            tasks.append(call_gemini(client, rn, pdf_bytes, closing_system,
                                     args.model, 'closing', semaphore))
    else:
        total_reqs = n * 3
        print(f"Sending {total_reqs} requests "
              f"({n} beyer + {n} win prob + {n} closing odds, "
              f"max {args.concurrency} at a time)...")
        for rn, pdf_bytes in sorted(races.items()):
            tasks.append(call_gemini(client, rn, pdf_bytes, beyer_system,
                                     args.model, 'beyer', semaphore))
            tasks.append(call_gemini(client, rn, pdf_bytes, winprob_system,
                                     args.model, 'winprob', semaphore))
            tasks.append(call_gemini(client, rn, pdf_bytes, closing_system,
                                     args.model, 'closing', semaphore))

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"All {len(results)} responses in {elapsed:.1f}s")

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
        import pandas as pd
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
        else:
            print(f"  No existing gemini.csv — run without --closing-only first")
            return

        closing_rows = []
        for race_num, horses in closing_data.items():
            for horse, odds in horses.items():
                closing_prob = round(1.0 / (odds + 1), 4) if odds > 0 else 0
                closing_rows.append({
                    'race': int(race_num),
                    'horse_number': int(horse) if horse.isdigit() else horse,
                    'gemini_closing_prob': closing_prob,
                })

        if closing_rows:
            closing_df = pd.DataFrame(closing_rows)
            existing['horse_number'] = pd.to_numeric(existing['horse_number'], errors='coerce')
            closing_df['horse_number'] = pd.to_numeric(closing_df['horse_number'], errors='coerce')

            # If --races was specified, only update closing_prob for rows in
            # those races. Otherwise this is a full closing-odds refresh —
            # drop the old column and merge new values across the whole file.
            if only_races is not None:
                if 'gemini_closing_prob' not in existing.columns:
                    existing['gemini_closing_prob'] = None
                merged = existing.copy()
                lookup = {(int(r['race']), int(r['horse_number'])): r['gemini_closing_prob']
                         for r in closing_rows}
                mask = merged['race'].astype(int).isin(only_races)
                for idx in merged[mask].index:
                    r = int(merged.at[idx, 'race'])
                    h_val = merged.at[idx, 'horse_number']
                    if pd.notna(h_val):
                        h = int(h_val)
                        if (r, h) in lookup:
                            merged.at[idx, 'gemini_closing_prob'] = lookup[(r, h)]
                merged.to_csv(out_path, index=False)
                n_filled = merged.loc[mask, 'gemini_closing_prob'].notna().sum()
                print(f"\nMerged closing odds for {sorted(only_races)} into {out_path} "
                      f"({n_filled}/{mask.sum()} horses in those races)")
            else:
                if 'gemini_closing_prob' in existing.columns:
                    existing = existing.drop(columns=['gemini_closing_prob'])
                merged = existing.merge(closing_df, on=['race', 'horse_number'], how='left')
                merged.to_csv(out_path, index=False)
                n_filled = merged['gemini_closing_prob'].notna().sum()
                print(f"\nMerged closing odds into {out_path} ({n_filled}/{len(merged)} horses)")
        else:
            print("  No closing data parsed")
        return

    # Diagnose disagreements
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
        closing_prob = ''
        if closing != '':
            odds_val = float(closing)
            closing_prob = round(1.0 / (odds_val + 1), 4) if odds_val > 0 else ''

        rows.append([race_num, horse, beyer, prob, closing_prob, args.track, args.date])

    os.makedirs(out_dir, exist_ok=True)

    # Two write modes:
    #
    #   Full run (no --races): overwrite the file entirely. Same as before.
    #
    #   Partial run (--races set): preserve rows for races we did NOT touch,
    #   replace rows for races we DID touch. Without this, the partial run
    #   would wipe out the good data for unaffected races — making --races
    #   actively destructive.
    if only_races is not None and os.path.exists(out_path):
        import pandas as pd
        existing = pd.read_csv(out_path)
        existing['race'] = pd.to_numeric(existing['race'], errors='coerce').astype('Int64')
        # Keep only rows whose race is NOT in only_races (the others get replaced)
        keep = existing[~existing['race'].astype('Int64').isin(only_races)].copy()

        # Build new rows as a DataFrame matching the same column order
        new_df = pd.DataFrame(rows, columns=['race', 'horse_number', 'gemini_expected_beyer',
                                             'gemini_win_prob', 'gemini_closing_prob',
                                             'track', 'date'])
        new_df['race'] = pd.to_numeric(new_df['race'], errors='coerce').astype('Int64')

        # Some existing files might not have all the columns yet (e.g. older
        # gemini.csv predates the closing_prob column). Add missing columns
        # so concat doesn't lose them.
        for col in new_df.columns:
            if col not in keep.columns:
                keep[col] = ''

        merged = pd.concat([keep, new_df], ignore_index=True)
        # Sort by race then horse_number for readability
        merged['_horse_sort'] = merged['horse_number'].astype(str).str.zfill(3)
        merged = merged.sort_values(['race', '_horse_sort']).drop(columns='_horse_sort')

        merged.to_csv(out_path, index=False)
        print(f"\nPartial update: replaced races {sorted(only_races)} in {out_path} "
              f"({len(rows)} new rows, {len(keep)} preserved from other races)")
    else:
        # Full overwrite — original behavior
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['race', 'horse_number', 'gemini_expected_beyer', 'gemini_win_prob',
                             'gemini_closing_prob', 'track', 'date'])
            writer.writerows(rows)

        print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    asyncio.run(main())