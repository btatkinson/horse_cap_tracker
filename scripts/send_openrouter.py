"""
send_openrouter.py

Send chopped race forms to any OpenRouter model (DeepSeek, GLM, ...). Each
race gets the same THREE prompts as send_claude/send_gemini/send_openai:
  1. Estimate mean Beyer speed figure per horse
  2. Estimate win probability per horse
  3. Predict closing tote odds per horse

Most cheap OpenRouter models can't read PDFs, so the race PDF is converted to
text with PyMuPDF (content-stream order, which keeps each horse's block
intact). DRF's special-font glyphs (fifths, beaten lengths, month codes)
survive as odd symbols, so a decoder legend is appended to every prompt.

By default the models read ML-redacted forms (chopped_forms_blind, created
on the fly by make_blind_forms.py): they handicap from the record, and the
market is added back in a second stage by project_openrouter.py. Sighted
picks just echo the ML (0.93 log-correlation on 2026-10-08 vs 0.67 blind).

Usage:
    python send_openrouter.py --date 2026-10-08 --track keeneland --model deepseek/deepseek-v4.1-flash --name deepseek_blind
    # GLM barely reasons by default (median 12 reasoning tokens/call on 10/08);
    # high effort pinned to Z.AI plus --deliberate gets it working through the field.
    python send_openrouter.py --date 2026-10-08 --track keeneland --model z-ai/glm-5.3-flash --name glm_blind \
        --reasoning-effort high --provider-order z-ai --deliberate
    # Sighted variant (models see the morning line), all three prompts:
    python send_openrouter.py ... --name deepseek --forms chopped_forms --prompts beyer,winprob,closing

Reads from:  {DATA_PATH}/{forms}/{date}/{track}/race_*.pdf   (forms defaults to chopped_forms_blind)
Writes to:   {DATA_PATH}/ai_picks/{date}/{track}/{name}.csv
             {DATA_PATH}/ai_picks/{date}/{track}/{name}_raw.jsonl  (raw responses, for auditing)

Requires: .env with OPENROUTER_KEY and DATA_PATH
"""

import asyncio
import argparse
import base64
import json
import os
import random
import re
import csv
import time
from pathlib import Path

import fitz  # pip install pymupdf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = os.environ['DATA_PATH']
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# OpenRouter fans out across many upstream providers, so it tolerates far
# more parallelism than the direct Anthropic/OpenAI tiers did.
DEFAULT_CONCURRENCY = 12
ALL_PROMPTS = ('beyer', 'winprob', 'closing', 'lengths')


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

LENGTHS_SYSTEM = """You are an expert horse racing handicapper projecting finishing margins.

Given DRF past performance data for a single race, estimate how far ahead of or behind the MEDIAN horse in this field each horse will finish, in lengths at the finish line. Positive = ahead of the median horse, negative = behind it. The median horse itself is about 0, so roughly half the field should be positive and half negative.

RESPOND WITH ONLY CSV DATA. No headers, no markdown, no explanations, no commentary, no asterisks.
Each line must be exactly: horse_number,lengths_vs_median
where horse_number is the program number (integer) and lengths_vs_median is a decimal number of lengths (e.g. 4.5 or -3.0).

Example of a CORRECT response (the entire response should look like this):
1,5.5
2,-1.0
3,0.0
4,-7.5
5,2.0

DO NOT include horse names, analysis, bold text, or any text besides the CSV rows.

Rules:
- Include ALL horses in the race (main body + Also Eligibles)
- Project the margin each horse will ACTUALLY run today, not its best past effort
- Use realistic spreads: winners typically beat the median horse by several lengths, and gaps grow with distance
- Consider pace scenario, class, distance/surface fit, form cycle, trainer/jockey, workouts
"""

SYSTEMS = {'beyer': BEYER_SYSTEM, 'winprob': WINPROB_SYSTEM, 'closing': CLOSING_ODDS_SYSTEM,
           'lengths': LENGTHS_SYSTEM}

# The PDF -> text conversion keeps DRF's special-font glyphs as Latin-1
# symbols. Decoded from known races on the 2026-10-08 Keeneland card
# (fractional times, beaten lengths, and work dates that must be in order).
DRF_TEXT_LEGEND = """NOTE ON THE DATA FORMAT: the past performances were extracted as plain text from a DRF PDF, one field per line. {header_note} DRF's special-font characters appear as symbols:
- Superscript digits (fifths of a second in times; lengths in running lines): ¥=0 ¦=1 §=2 ¨=3 ©=4 ª=5 «=6 ¬=7 ­=8 ®=9. Examples: ":45©" = 45 4/5 sec, "1:10©" = 1:10 4/5, running-line "4©ô" = 4th beaten 4½ lengths, "11¦¦" = 11th by 11 lengths, "1¨õ" = 1st by 3¾ lengths.
- Fractions: ô=½  õ=¾  ö=¼  Â=1/16  (e.g. "6ôf" = 6½ furlongs, "1Â MILES" = 1 1/16 miles).
- Dates are day + month glyph + 2-digit year: Ü=Jun, Û=Jul, Ý=Aug, æ=Sep, å=Oct (e.g. "16æ26" = 16 Sep 2026). Other months use similar glyphs. Past-performance lines and workouts are listed most recent first.
- "ê" after a distance or track marks turf. In works, a leading "ò" marks a bullet (fastest work of the day at that distance).
- The record rows (Life / year / track / D.Fst / Wet / Synth / Turf / Dst) are: starts, wins, seconds, thirds, earnings, best Beyer."""
HEADER_NOTE = "Each horse block starts with program number, morning line odds, then horse name."
BLIND_HEADER_NOTE = ("Each horse block starts with program number, then horse name. Today's morning-line "
                     "odds have been deliberately removed: judge each horse on its record alone.")

# Native-PDF models (Gemini, Claude, GPT) see the rendered page, so no glyph
# legend is needed — only the note about the blacked-out odds.
PDF_BLIND_NOTE = ("The attached PDF is the DRF past performances for this race. Today's morning-line "
                  "odds have been deliberately blacked out: judge each horse on its record alone.")
PDF_SIGHTED_NOTE = "The attached PDF is the DRF past performances for this race."

# GLM-style light reasoners answer on gut feel unless told to deliberate.
DELIBERATE_NOTE = """Before answering, work through every horse carefully: recent form and speed figures, class level, distance and surface fit, the likely pace scenario, trainer and jockey statistics, and workouts. Only after that analysis, output the CSV."""


def format_legend(sample_text: str, blind: bool) -> str:
    """Data-format note for the prompt. The glyph legend only applies to DRF
    PDFs; the April 2026 Keeneland cards are Equibase PPs and two Churchill
    cards are Brisnet Ultimate PPs."""
    blind_note = (" Today's morning-line odds have been deliberately removed: judge each horse "
                  "on its record alone.") if blind else ""
    if 'equibase.com' in sample_text or 'APPROX. POST' in sample_text:
        return ("NOTE ON THE DATA FORMAT: the past performances were extracted as plain text from an "
                "Equibase PDF, one field per line; a few symbols are font glyphs." + blind_note)
    if 'Ultimate PP' in sample_text or 'brisnet' in sample_text.lower():
        return ("NOTE ON THE DATA FORMAT: the past performances were extracted as plain text from a "
                "Brisnet Ultimate PP PDF, one field per line; a few symbols are font glyphs." + blind_note)
    return DRF_TEXT_LEGEND.format(header_note=BLIND_HEADER_NOTE if blind else HEADER_NOTE)


def extract_race_text(pdf_path: Path, track: str) -> str:
    """PDF -> plain text in content-stream order.

    Race 1's first page also carries the card-wide entry/trainer index,
    which lists every horse on the card; strip it so the model doesn't
    think those horses are in this race.
    """
    doc = fitz.open(str(pdf_path))
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    if 'INDEX TO TRAINERS' in text:
        idx_end = text.find('INDEX TO TRAINERS')
        post = text.find('Post time:', idx_end)
        if post != -1:
            # Race header ("Keeneland / Alw 50000s 1 / conditions...") sits
            # just before "Post time:" — cut at the last track-name line.
            hdr = text.rfind(track.split('-')[0].capitalize(), idx_end, post)
            if hdr != -1:
                text = text[hdr:]
    return text.strip()


def load_race_texts(chopped_dir: str, track: str) -> dict[int, str]:
    """Load race_*.pdf files, return {race_number: text}."""
    races = {}
    for f in sorted(Path(chopped_dir).glob("race_*.pdf")):
        match = re.search(r'race_(\d+)', f.name)
        if match:
            races[int(match.group(1))] = extract_race_text(f, track)
    return races


def load_race_pdfs(chopped_dir: str) -> dict[int, bytes]:
    """Load race_*.pdf files, return {race_number: pdf_bytes}."""
    races = {}
    for f in sorted(Path(chopped_dir).glob("race_*.pdf")):
        match = re.search(r'race_(\d+)', f.name)
        if match:
            races[int(match.group(1))] = f.read_bytes()
    return races


async def call_openrouter(client: AsyncOpenAI, race_num: int, race_input,
                          system: str, model: str, label: str,
                          extra_body: dict, max_tokens: int,
                          semaphore: asyncio.Semaphore,
                          max_retries: int = 5) -> tuple[int, str, str, dict]:
    """Send one race to OpenRouter, return (race_num, label, csv_text, meta).

    race_input is extracted text (str) or the race PDF (bytes) for models
    that read PDFs natively.
    """
    async with semaphore:
        if isinstance(race_input, bytes):
            b64 = base64.standard_b64encode(race_input).decode('utf-8')
            user_content = [
                {"type": "file", "file": {"filename": f"race_{race_num}.pdf",
                                          "file_data": f"data:application/pdf;base64,{b64}"}},
                {"type": "text", "text": f"Here is the DRF past performance PDF for Race {race_num}. "
                                         f"Provide your estimates for each horse."},
            ]
        else:
            user_content = (f"Here is the DRF past performance data for Race {race_num}.\n\n"
                            f"<past_performances>\n{race_input}\n</past_performances>\n\n"
                            f"Provide your estimates for each horse.")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                if not resp.choices:
                    raise RuntimeError(f"no choices in response: {resp}")
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                if not content:
                    raise RuntimeError(f"empty content (finish_reason={resp.choices[0].finish_reason})")
                usage = resp.usage.model_dump() if resp.usage else {}
                meta = {
                    'provider': getattr(resp, 'provider', None),
                    'finish_reason': resp.choices[0].finish_reason,
                    'usage': usage,
                    'reasoning_chars': len(getattr(msg, 'reasoning', None) or ''),
                }
                return (race_num, label, content, meta)
            except RateLimitError as e:
                retry_after = None
                try:
                    retry_after = e.response.headers.get("retry-after")
                except AttributeError:
                    pass
                try:
                    wait = float(retry_after) if retry_after is not None else 20.0
                except ValueError:
                    wait = 20.0
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
        return (race_num, label, "", {})


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


def build_extra_body(reasoning_effort='medium', provider_sort='throughput', provider_order=None):
    extra_body = {}
    if reasoning_effort == 'none':
        extra_body['reasoning'] = {'enabled': False}
    else:
        extra_body['reasoning'] = {'effort': reasoning_effort, 'exclude': True}
    if provider_order:
        extra_body['provider'] = {'order': list(provider_order), 'allow_fallbacks': True}
    elif provider_sort != 'none':
        extra_body['provider'] = {'sort': provider_sort}
    return extra_body


def load_track_prompt(date: str, track: str, verbose: bool = True) -> str:
    meta_path = os.path.join(DATA_PATH, 'historical_meta.pkl')
    if not os.path.exists(meta_path):
        return ""
    import pickle
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    track_meta = meta.get(date, {}).get(track)
    if not track_meta:
        if verbose:
            print(f"\n⚠  No track conditions found for {date} / {track}")
        return ""
    if verbose:
        print(f"\nTrack conditions ({track_meta.get('source', '?')}):")
        print(f"  Dirt: {track_meta.get('dirt', '?')}")
        print(f"  Turf: {track_meta.get('turf', '?')}")
    return track_meta.get('prompt', '')


def build_systems(prompts, sample_input, blind: bool, deliberate: bool, track_prompt: str) -> dict:
    """System prompt per label: task + data-format note (+ deliberate) + track conditions."""
    if isinstance(sample_input, bytes):
        note = PDF_BLIND_NOTE if blind else PDF_SIGHTED_NOTE
    else:
        note = format_legend(sample_input, blind)
    systems = {}
    for label in prompts:
        s = SYSTEMS[label] + "\n\n" + note
        if deliberate:
            s += "\n\n" + DELIBERATE_NOTE
        if track_prompt:
            s += "\n\n" + track_prompt
        systems[label] = s
    return systems


def prepare_card(date: str, track: str, forms: str, forms_track: str | None, pdf: bool,
                 races_filter: set | None, verbose: bool):
    """Resolve the forms folder (creating blind copies if needed) and load each race.

    Returns (races, track_prompt) where races is {race: text or pdf bytes}.
    """
    forms_track = forms_track or track
    chopped_dir = os.path.join(DATA_PATH, forms, date, forms_track)
    if 'blind' in forms and not list(Path(chopped_dir).glob('race_*.pdf')):
        from make_blind_forms import blind_card
        counts = blind_card(date, forms_track)
        print(f"Created ML-redacted forms for {date}/{forms_track}: {sum(counts.values())} odds blacked out")
    races = load_race_pdfs(chopped_dir) if pdf else load_race_texts(chopped_dir, forms_track)
    if races_filter:
        races = {rn: t for rn, t in races.items() if rn in races_filter}
    if not races:
        print(f"No race PDFs found in {chopped_dir}")
    track_prompt = (load_track_prompt(date, track, verbose=verbose)
                    or load_track_prompt(date, forms_track, verbose=False))
    return races, track_prompt


def write_outputs(date: str, track: str, name: str, model: str, results, *,
                  elapsed: float = 0.0, verbose: bool = True) -> dict:
    """Write {name}.csv + {name}_raw.jsonl from [(race, label, text, meta)].

    Merges into an existing {name}.csv: columns for the prompts in this run
    are replaced, the others are kept, so a Beyer-only run doesn't wipe the
    win probabilities from an earlier run. Raw responses are appended.

    meta['usage'] may be OpenAI-style (prompt/completion tokens) or
    Anthropic-style (input/output tokens). Returns a summary dict.
    """
    out_dir = os.path.join(DATA_PATH, 'ai_picks', date, track)
    out_path = os.path.join(out_dir, f'{name}.csv')
    raw_path = os.path.join(out_dir, f'{name}_raw.jsonl')
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(raw_path, 'a', encoding='utf-8') as f:
        for race_num, label, raw_text, meta in results:
            f.write(json.dumps({'race': race_num, 'label': label, 'model': model, 'run_at': stamp,
                                'content': raw_text, **meta}) + '\n')

    data = {label: {} for label in ALL_PROMPTS}
    tok_in = tok_out = 0
    reasoning_toks, failed = [], 0
    for race_num, label, raw_text, meta in results:
        u = meta.get('usage') or {}
        tok_in += (u.get('prompt_tokens') or u.get('input_tokens') or 0)
        tok_out += (u.get('completion_tokens') or u.get('output_tokens') or 0)
        reasoning_toks.append(((u.get('completion_tokens_details') or {}).get('reasoning_tokens')) or 0)
        parsed = parse_csv_response(raw_text)
        if parsed:
            if verbose:
                print(f"  R{race_num:>2} {label:>7}: {len(parsed)} horses")
            data[label][race_num] = {horse: val for horse, val in parsed}
        else:
            failed += 1
            print(f"  {date}/{track} R{race_num:>2} {label:>7}: *** PARSE FAILED ***")
            if raw_text:
                print(f"    First 200 chars: {raw_text[:200]}")

    all_horses = set()
    for d in data.values():
        for race_num, horses in d.items():
            for h in horses:
                all_horses.add((race_num, h))

    rows = []
    for race_num, horse in sorted(all_horses, key=lambda x: (x[0], len(x[1]), x[1])):
        beyer = data['beyer'].get(race_num, {}).get(horse, '')
        prob = data['winprob'].get(race_num, {}).get(horse, '')
        closing = data['closing'].get(race_num, {}).get(horse, '')
        if beyer != '':
            beyer = int(round(beyer))
        if prob != '':
            prob = round(float(prob), 4)
        closing_prob = ''
        if closing != '':
            odds_val = float(closing)
            closing_prob = round(1.0 / (odds_val + 1), 4) if odds_val > 0 else ''
        lengths = data['lengths'].get(race_num, {}).get(horse, '')
        if lengths != '':
            lengths = round(float(lengths), 2)
        rows.append([race_num, horse, beyer, prob, closing_prob, lengths, track, date])

    header = ['race', 'horse_number', f'{name}_expected_beyer', f'{name}_win_prob',
              f'{name}_closing_prob', f'{name}_lengths', 'track', 'date']
    value_cols = {'beyer': header[2], 'winprob': header[3], 'closing': header[4], 'lengths': header[5]}
    out = pd.DataFrame(rows, columns=header)
    out['horse_number'] = out['horse_number'].astype(str)
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, dtype={'horse_number': str})
        for col in header:            # older files predate some columns
            if col not in old:
                old[col] = np.nan
        ran = {label for _, label, _, _ in results}
        m = old[header].merge(out, on=['race', 'horse_number'], how='outer', suffixes=('_old', ''))
        for label, col in value_cols.items():
            if label not in ran:
                m[col] = m[f'{col}_old']
        for col in ('track', 'date'):
            m[col] = m[col].fillna(m[f'{col}_old'])
        out = m[header].copy()
    out['_h'] = pd.to_numeric(out['horse_number'].str.extract(r'(\d+)')[0], errors='coerce')
    out = out.sort_values(['race', '_h', 'horse_number']).drop(columns='_h')
    out.to_csv(out_path, index=False)

    summary = {'card': f'{date}/{track}', 'name': name, 'rows': len(rows), 'failed': failed,
               'calls': len(results), 'tok_in': tok_in, 'tok_out': tok_out,
               'median_reasoning_tokens': float(np.median(reasoning_toks)) if reasoning_toks else 0.0,
               'seconds': round(elapsed, 1)}
    if verbose:
        print(f"All {len(results)} responses in {elapsed:.1f}s")
        print(f"Tokens: {tok_in:,} in / {tok_out:,} out; "
              f"median reasoning tokens/call {summary['median_reasoning_tokens']:.0f}")
        print(f"\nSaved {len(rows)} rows to {out_path}")
        print(f"Raw responses: {raw_path}")
    return summary


async def run_card(date: str, track: str, model: str, name: str, *,
                   client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                   forms: str = 'chopped_forms', prompts=ALL_PROMPTS,
                   extra_body: dict | None = None, deliberate: bool = False,
                   max_tokens: int = 32000, races_filter: set | None = None,
                   forms_track: str | None = None, pdf: bool = False,
                   verbose: bool = True) -> dict:
    """Run every prompt for every race on one card and write {name}.csv.

    forms_track: forms folder name when it differs from the picks/results
    name (2026-04-05 forms live under 'gulfstream', results 'gulfstream-park').
    pdf: send the race PDF itself (for models that read PDFs natively)
    instead of extracted text.
    Returns a summary dict (rows, tokens, parse failures, reasoning tokens).
    """
    races, track_prompt = prepare_card(date, track, forms, forms_track, pdf, races_filter, verbose)
    if not races:
        return {'card': f'{date}/{track}', 'name': name, 'rows': 0, 'failed': 0}
    systems = build_systems(prompts, next(iter(races.values())), 'blind' in forms, deliberate, track_prompt)

    body = dict(extra_body or {})
    if pdf:
        body['plugins'] = [{'id': 'file-parser', 'pdf': {'engine': 'native'}}]
    tasks = [call_openrouter(client, rn, race_input, systems[label], model, label,
                             body, max_tokens, semaphore)
             for rn, race_input in sorted(races.items()) for label in prompts]
    t0 = time.time()
    results = await asyncio.gather(*tasks)
    return write_outputs(date, track, name, model, results, elapsed=time.time() - t0, verbose=verbose)


def parse_races(spec: str) -> set:
    wanted = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            wanted.update(range(int(a), int(b) + 1))
        else:
            wanted.add(int(part))
    return wanted


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=OPENROUTER_BASE, api_key=os.environ['OPENROUTER_KEY'],
                       max_retries=0, timeout=600)


async def main():
    parser = argparse.ArgumentParser(description='Send race forms to an OpenRouter model for estimates')
    parser.add_argument('--date', type=str, required=True, help='Date folder (e.g. 2026-10-08)')
    parser.add_argument('--track', type=str, required=True, help='Track folder (e.g. keeneland)')
    parser.add_argument('--model', type=str, required=True,
                        help='OpenRouter model id (e.g. deepseek/deepseek-v4.1-flash)')
    parser.add_argument('--name', type=str, required=True,
                        help='Short provider name for the CSV + column prefix (e.g. deepseek)')
    parser.add_argument('--races', type=str, default=None,
                        help='Only these races, e.g. "1-3" or "2,5" (default: all)')
    parser.add_argument('--forms', type=str, default='chopped_forms_blind',
                        help='Forms folder under DATA_PATH (default ML-redacted; chopped_forms = sighted)')
    parser.add_argument('--prompts', type=str, default='winprob,beyer',
                        help='Comma list of prompts to run: beyer,winprob,closing '
                             '(closing leans on the ML, so it is off by default)')
    parser.add_argument('--reasoning-effort', type=str, default='medium',
                        choices=['none', 'minimal', 'low', 'medium', 'high'],
                        help='OpenRouter reasoning effort (default medium; "none" disables)')
    parser.add_argument('--provider-sort', type=str, default='throughput',
                        choices=['throughput', 'price', 'latency', 'none'],
                        help='OpenRouter upstream provider routing preference')
    parser.add_argument('--provider-order', type=str, default=None,
                        help='Preferred upstream providers, e.g. "z-ai" (falls back to others)')
    parser.add_argument('--deliberate', action='store_true',
                        help='Tell the model to work through every horse before answering')
    parser.add_argument('--pdf', action='store_true',
                        help='Send the race PDF natively (Gemini/Claude/GPT) instead of extracted text')
    parser.add_argument('--max-tokens', type=int, default=32000)
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--skip-odds', action='store_true', help='Skip closing odds prediction')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Max concurrent API requests (default {DEFAULT_CONCURRENCY})')
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]
    if args.skip_odds and 'closing' in prompts:
        prompts.remove('closing')
    print(f"Model: {args.model}  (as '{args.name}'), forms: {args.forms}, prompts: {'+'.join(prompts)}")
    if not args.yes:
        confirm = input("Proceed? [Y/n] ")
        if confirm.lower() == 'n':
            print("Aborted.")
            return

    order = [p.strip() for p in args.provider_order.split(',')] if args.provider_order else None
    await run_card(args.date, args.track, args.model, args.name,
                   client=make_client(), semaphore=asyncio.Semaphore(args.concurrency),
                   forms=args.forms, prompts=prompts,
                   extra_body=build_extra_body(args.reasoning_effort, args.provider_sort, order),
                   deliberate=args.deliberate, max_tokens=args.max_tokens,
                   races_filter=parse_races(args.races) if args.races else None, pdf=args.pdf)


if __name__ == '__main__':
    asyncio.run(main())
