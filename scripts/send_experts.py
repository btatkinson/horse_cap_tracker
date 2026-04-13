"""
send_experts.py

Send raw expert pick text files to Claude to extract logit deltas.
Each expert's raw .txt gets converted to a structured .csv of
race-level logit adjustments relative to morning line.

Usage:
    python send_experts.py --date 2026-04-08 --track keeneland
    python send_experts.py --date 2026-04-08 --track keeneland --model claude-sonnet-4-20250514

Reads from:
    {DATA_PATH}/raw_expert_picks/{date}/{track}/*.txt

Writes to:
    {DATA_PATH}/expert_picks/{date}/{track}/{expert_name}.csv

CSV format:
    race,horse_number,logit_delta,sentiment,expert
    1,6,0.7,best_bet,scott_hazelton
    1,2,0.3,mentioned,scott_hazelton

Logit delta scale:
    +0.7   Best bet / top pick / strongest conviction
    +0.5   Strong 2nd pick with real conviction
    +0.3   Mentioned favorably / in top 3 but not the focus
    +0.15  Lukewarm / contender without enthusiasm
    -0.3   Mentioned negatively / concerns raised
    -0.5   Explicit avoid / toss
     0.0   Not mentioned (not written to CSV)
"""

import os, re, csv, argparse
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = Path(os.environ['DATA_PATH'])

try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError("pip install anthropic")


# =============================================================================
# LLM EXTRACTION
# =============================================================================

SYSTEM_PROMPT = """You are parsing horse racing expert picks and converting them to logit adjustments.

Given an expert's race-by-race picks and commentary, assign a logit_delta to each horse mentioned.

IMPORTANT: The user may provide a HORSE REFERENCE table mapping horse names to program numbers
per race. Use it to look up the correct race number and program number for each horse name.
If the expert mentions a horse by name, find it in the reference to get the right numbers.
If a horse name doesn't appear in the reference, skip it (it may be scratched).

LOGIT DELTA SCALE (use these exact values):
  +0.7   Best bet / top pick / "Win Bet" / the expert's #1 choice in that race
  +0.5   Strong 2nd pick — real conviction expressed in write-up
  +0.3   3rd pick or mentioned favorably but less conviction, or "Use" horses
  +0.15  Listed as a contender without enthusiasm, or "Consider" / "use underneath"
  -0.3   Mentioned negatively — concerns about form, fitness, post, etc.
  -0.5   Explicit avoid / toss / "I'm skeptical of this horse"
   0.0   Not mentioned — do NOT include these (they get 0.0 by default)

RULES:
- Output program/post position NUMBERS only, not horse names
- "Most Likely" or "Win Bet" = +0.7 (best_bet)
- "Use" horses = +0.3 (mentioned)
- "Consider" horses = +0.15 (contender)
- "Backup" = +0.15 (contender)
- "Spread" race picks: the "Most Likely" horse gets +0.7, "Use" gets +0.3, "Consider" gets +0.15
- The first horse listed for each race typically gets +0.7
- If the expert says two horses are equally good, give both +0.6
- If the expert only covers some races (not a full card), only output those races
- If the expert lists picks as "6-2-8", that's 1st pick #6, 2nd pick #2, 3rd pick #8
- "Top Picks #6, #4 and #7": #6 gets +0.7, #4 gets +0.5, #7 gets +0.3
- "Longshot play of the day" → +0.7 for that race
- Pay attention to negative language: "I'm skeptical" → -0.3
- Pick 3/4/5 sequences list horses per leg — map each to the correct race number
  e.g. "Pick 5 Attack: 7/1,9,10/..." starting Race 5 means R5=#7, R6=#1,#9,#10, etc.

RESPOND WITH ONLY CSV DATA. No headers, no markdown, no explanations.
Each line: race_number,horse_number,logit_delta,sentiment

sentiment must be one of: best_bet, strong, mentioned, contender, negative, avoid

Example response:
1,6,0.7,best_bet
1,2,0.5,strong
1,8,0.3,mentioned
2,11,0.7,best_bet
2,5,0.3,mentioned
2,9,0.15,contender
3,4,-0.3,negative
"""


def _load_entries_reference(track, date_str):
    """
    Build a race-by-race horse reference from entries or UPC entries.
    Tries historical entries for the specific date first (most reliable),
    falls back to UPC entries (which is always the current/latest card).
    Returns string like:
        Race 1: #1 Waggley, #2 Fast Runner, #3 Smoke, ...
        Race 2: #1 First Horse, #2 Second Horse, ...
    """
    entries_dir = DATA_PATH / 'entries'
    upc_dir = DATA_PATH / 'upc_entries'

    df = None

    # Try entries first — date-specific and has horse names
    if entries_dir.exists():
        dfs = []
        for f in entries_dir.glob('*.csv'):
            try:
                edf = pd.read_csv(f)
                if 'date' not in edf.columns:
                    continue
                edf['date'] = pd.to_datetime(edf['date']).dt.strftime('%Y-%m-%d')
                subset = edf[edf['date'] == date_str]
                if len(subset) > 0 and 'horse_name' in subset.columns:
                    dfs.append(subset)
            except Exception:
                continue
        if dfs:
            df = pd.concat(dfs, ignore_index=True)

    # Fall back to UPC entries (current card only)
    if df is None or df.empty:
        for candidate in [upc_dir / f'{track}.csv',
                          upc_dir / f'{track.replace("-","_")}.csv']:
            if candidate.exists():
                df = pd.read_csv(candidate)
                break

    if df is None or df.empty:
        return None

    # Normalize columns
    if 'horse' in df.columns and 'horse_number' not in df.columns:
        df['horse_number'] = df['horse']
    if 'running' in df.columns:
        df = df[df['running'] == 1]

    df['race'] = pd.to_numeric(df['race'], errors='coerce')
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')

    if 'horse_name' not in df.columns:
        return None

    lines = []
    for race_num in sorted(df['race'].dropna().unique()):
        race_df = df[df['race'] == race_num].sort_values('horse_number')
        horses = []
        for _, row in race_df.iterrows():
            hn = int(row['horse_number'])
            name = str(row['horse_name']).strip()
            if name and name != 'nan':
                horses.append(f"#{hn} {name}")
        if horses:
            lines.append(f"Race {int(race_num)}: {', '.join(horses)}")

    return '\n'.join(lines) if lines else None


def extract_deltas(expert_text, expert_name, entries_ref=None,
                   model='claude-sonnet-4-20250514'):
    """Send expert text to Claude, get back logit deltas."""
    client = Anthropic(
        api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    )

    user_content = (
        f"Here are the picks from expert '{expert_name}'. "
        f"Extract logit deltas for every horse mentioned:\n\n"
        f"{expert_text}"
    )

    if entries_ref:
        user_content = (
            f"HORSE REFERENCE (use this to look up program numbers from horse names):\n"
            f"{entries_ref}\n\n"
            f"---\n\n"
            f"Here are the picks from expert '{expert_name}'. "
            f"Extract logit deltas for every horse mentioned. "
            f"Use the reference above to convert horse names to program numbers:\n\n"
            f"{expert_text}"
        )

    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_content}],
    )

    raw = resp.content[0].text.strip()
    return raw, parse_csv_response(raw, expert_name)


def parse_csv_response(raw, expert_name):
    """Parse LLM CSV output into list of dicts."""
    text = re.sub(r'```(?:csv)?\n?', '', raw).strip()
    lines = text.strip().split('\n')

    # Skip header if present
    if lines and any(kw in lines[0].lower() for kw in
                     ['race', 'horse', 'logit', 'delta', 'sentiment']):
        lines = lines[1:]

    results = []
    for line in lines:
        line = line.strip()
        if not line or '**' in line or ':' in line and ',' not in line:
            continue
        parts = line.split(',')
        if len(parts) < 3:
            continue
        try:
            race = int(parts[0].strip())
            horse = parts[1].strip()
            if not re.match(r'^\d+[A-Za-z]?$', horse):
                continue
            delta = float(parts[2].strip())
            # Clamp to valid range
            delta = max(-1.0, min(1.0, delta))
            sentiment = parts[3].strip() if len(parts) >= 4 else 'mentioned'
            # Validate sentiment
            valid_sentiments = {'best_bet', 'strong', 'mentioned',
                                'contender', 'negative', 'avoid'}
            if sentiment not in valid_sentiments:
                sentiment = 'mentioned'
            results.append({
                'race': race,
                'horse_number': horse,
                'logit_delta': round(delta, 2),
                'sentiment': sentiment,
                'expert': expert_name,
            })
        except (ValueError, IndexError):
            continue

    return results


# =============================================================================
# MAIN
# =============================================================================

def _read_expert_file(path):
    """Read expert text from .txt or .pdf file."""
    if path.suffix.lower() == '.pdf':
        try:
            import fitz
            doc = fitz.open(str(path))
            text = '\n'.join(page.get_text() for page in doc)
            doc.close()
            # Clean up PDF extraction artifacts
            import re as _re
            text = _re.sub(r'\n{3,}', '\n\n', text)
            text = '\n'.join(line.strip() for line in text.split('\n'))
            return text.strip()
        except ImportError:
            print("(need pymupdf for PDF)", end=' ')
            return None
        except Exception as e:
            print(f"(PDF read error: {e})", end=' ')
            return None
    else:
        return path.read_text(encoding='utf-8')


def process_experts(date_str, track, model='claude-sonnet-4-20250514'):
    """Process all raw expert .txt and .pdf files for a date/track."""
    raw_dir = DATA_PATH / 'raw_expert_picks' / date_str / track
    sent_dir = DATA_PATH / 'already_sent_eps' / date_str / track
    out_dir = DATA_PATH / 'expert_picks' / date_str / track
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"No raw picks found at {raw_dir}")
        print(f"Run: python scrape_experts.py --track {track} --date {date_str}")
        print(f"Or manually place .txt / .pdf files in {raw_dir}")
        return

    # Find both .txt and .pdf files
    raw_files = sorted(list(raw_dir.glob('*.txt')) + list(raw_dir.glob('*.pdf')))
    if not raw_files:
        print(f"No .txt or .pdf files in {raw_dir}")
        return

    # Filter out experts already sent
    to_process = []
    skipped = 0
    for raw_path in raw_files:
        sent_path = sent_dir / raw_path.name
        if sent_path.exists():
            skipped += 1
        else:
            to_process.append(raw_path)

    if skipped:
        print(f"Skipping {skipped} already-sent expert(s) (found in already_sent_eps/)")
    if not to_process:
        print(f"All experts already sent for {date_str} / {track}")
        return

    print(f"Processing {len(to_process)} expert(s) for {date_str} / {track}")
    print(f"  Model: {model}")
    print(f"  Output: {out_dir}/")

    # Load horse reference for name→number mapping
    entries_ref = _load_entries_reference(track, date_str)
    if entries_ref:
        n_races = entries_ref.count('\n') + 1
        print(f"  Horse reference: {n_races} races loaded")
    else:
        print(f"  WARNING: No horse reference found — name→number mapping may fail")
        print(f"    Run: python download_upc.py --track {track} --date {date_str}")

    total_rows = 0
    sent_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in to_process:
        expert_name = raw_path.stem
        print(f"\n  {expert_name} ({raw_path.suffix})...", end=' ')

        raw_text = _read_expert_file(raw_path)
        if raw_text is None or len(raw_text.strip()) < 50:
            print("too short or unreadable, skipping")
            continue

        try:
            raw_response, rows = extract_deltas(raw_text, expert_name,
                                                entries_ref=entries_ref, model=model)
        except Exception as e:
            print(f"API error: {e}")
            continue

        if not rows:
            print("no picks extracted")
            print(f"    Raw response: {raw_response[:200]}")
            continue

        # Save CSV
        out_path = out_dir / f"{expert_name}.csv"
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f,
                fieldnames=['race', 'horse_number', 'logit_delta', 'sentiment', 'expert'])
            writer.writeheader()
            writer.writerows(rows)

        # Move raw file to already_sent
        import shutil
        shutil.move(str(raw_path), str(sent_dir / raw_path.name))

        # Stats
        races_covered = len(set(r['race'] for r in rows))
        best_bets = sum(1 for r in rows if r['sentiment'] == 'best_bet')
        negatives = sum(1 for r in rows if r['logit_delta'] < 0)
        print(f"{len(rows)} picks across {races_covered} races "
              f"({best_bets} best bets, {negatives} negatives)")
        total_rows += len(rows)

    print(f"\nDone: {total_rows} total picks saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Send raw expert picks to Claude for logit delta extraction')
    parser.add_argument('--date', type=str, required=True, help='Date (e.g. 2026-04-08)')
    parser.add_argument('--track', type=str, required=True, help='Track (e.g. keeneland)')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514',
                        help='Claude model for extraction')
    args = parser.parse_args()

    process_experts(args.date, args.track, args.model)


if __name__ == '__main__':
    main()