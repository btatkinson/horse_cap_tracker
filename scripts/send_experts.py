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


def _load_entries_dataframe(track, date_str):
    """Resolve entries data for a given (track, date_str) with priority:

      1. entries/{track}_{date}.csv  — direct filename lookup (preferred)
      2. entries/{track_with_underscores}_{date}.csv  — track-name variant
      3. Any CSV in entries/ with a 'date' column matching date_str (legacy)
      4. upc_entries/{track}.csv     — current/latest card only

    For BACKFILL of historical dates, paths 1/2 are the only correct
    answers — UPC always reflects today's card, so falling through to
    UPC would silently give the wrong horse lookups for old dates.
    To make this visible: when we fall through to UPC, print a warning
    that names the date so the caller can spot mismatches.

    Returns a DataFrame with at minimum {race, horse_number, horse_name}
    columns (and others if present), or None if nothing matched.
    """
    entries_dir = DATA_PATH / 'entries'
    upc_dir = DATA_PATH / 'upc_entries'

    df = None

    # Path 1+2: direct filename lookup. Handles both dash and underscore
    # track-name spellings.
    if entries_dir.exists():
        for track_variant in (track, track.replace('-', '_'), track.replace('_', '-')):
            candidate = entries_dir / f'{track_variant}_{date_str}.csv'
            if candidate.exists():
                try:
                    df = pd.read_csv(candidate)
                    break
                except Exception:
                    df = None  # corrupt file — fall through to scan

    # Path 3: scan entries/ for any CSV with a matching date column.
    # Legacy behavior — kept so that aggregated-format entries files
    # (one big CSV containing many dates) keep working.
    if (df is None or df.empty) and entries_dir.exists():
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

    # Path 4: UPC fallback. ALWAYS the current card, so warn if the
    # caller is asking for a date that's likely historical.
    if df is None or df.empty:
        for candidate in [upc_dir / f'{track}.csv',
                          upc_dir / f'{track.replace("-","_")}.csv']:
            if candidate.exists():
                try:
                    df = pd.read_csv(candidate)
                    print(f"  WARNING: no entries/ file found for "
                          f"{track}/{date_str}; using upc_entries (current card "
                          f"only). For historical backfills this WILL be wrong "
                          f"if the upcoming card differs.")
                except Exception:
                    df = None
                break

    if df is None or df.empty:
        return None

    # Normalize columns the same way both callers used to.
    if 'horse' in df.columns and 'horse_number' not in df.columns:
        df['horse_number'] = df['horse']
    df['race'] = pd.to_numeric(df['race'], errors='coerce')
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')

    return df


def _load_entries_reference(track, date_str):
    """
    Build a race-by-race horse reference from entries or UPC entries.
    Returns string like:
        Race 1: #1 Waggley, #2 Fast Runner, #3 Smoke, ...
        Race 2: #1 First Horse, #2 Second Horse, ...
    """
    df = _load_entries_dataframe(track, date_str)
    if df is None or df.empty:
        return None

    # Reference is for sending to Sonnet — only show running horses
    # (scratched horses just clutter the prompt).
    if 'running' in df.columns:
        df = df[df['running'] == 1]

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


# =============================================================================
# DRF "ProPick" PDF PARSING
#
# DRF Picks PDFs (filenames like "ProPickCD20260430.pdf", "propickKEE...pdf")
# bundle four pickers in a single document:
#   - Marcus Hersh    (full prose write-up, pp. 5-6)
#   - Kenny Peck      (HANDICAP column, picks-panel grid only, p. 4)
#   - Michael Hammersly (HERMIS column, picks-panel grid only, p. 4)
#   - SWEEP           (composite/algorithmic column, picks-panel grid only, p. 4)
#
# Strategy:
#   * Hersh has prose -> send to Sonnet (full sentiment extraction)
#   * Other three only have 1st/2nd/3rd picks per race in the panel
#       -> deterministic mapping: 1st=+0.7 best_bet, 2nd=+0.5 strong, 3rd=+0.3 mentioned
#       (No API call needed; just look up program numbers from entries.)
# =============================================================================

def _normalize_horse_name(name):
    """Lowercase, drop country suffix like (IRE)/(GB)/(FR), strip punct, collapse spaces."""
    if not name:
        return ''
    s = str(name).lower().strip()
    s = re.sub(r'\s*\([a-z]{2,4}\)\s*', '', s)        # drop (IRE), (GB), etc.
    s = re.sub(r"[\u2018\u2019\u201c\u201d'\".,]", '', s)  # strip apostrophes/quotes/punct
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _load_entries_lookup(track, date_str):
    """Build a {(race, normalized_horse_name): horse_number} dict for fast
    name->number lookup. Uses the shared _load_entries_dataframe helper so
    the file-resolution priority (entries/{track}_{date}.csv → scan →
    upc_entries) is consistent with _load_entries_reference.

    NOTE: We deliberately do NOT filter on `running == 1` here, unlike
    _load_entries_reference. Some scrapers mark whole races (e.g. Arabian-only
    races on a Thoroughbred card) as running=0, which would erase those races
    from the lookup and leave panel picks unmappable. For pure name->number
    lookup, scratches still give us the right number.

    If both running=1 and running=0 rows exist for the same (race, name),
    prefer running=1 by sorting it last (last-write-wins in the dict).
    """
    df = _load_entries_dataframe(track, date_str)
    if df is None or df.empty or 'horse_name' not in df.columns:
        return None

    if 'running' in df.columns:
        df = df.sort_values('running', kind='stable')  # 0s first, 1s last

    lookup = {}
    for _, row in df.iterrows():
        if pd.isna(row['race']) or pd.isna(row['horse_number']):
            continue
        race = int(row['race'])
        hn = int(row['horse_number'])
        name = str(row['horse_name']).strip()
        if not name or name == 'nan':
            continue
        lookup[(race, _normalize_horse_name(name))] = hn
    return lookup if lookup else None


def _parse_propick_panel(pdf_path):
    """Parse the picks panel page of a DRF ProPick PDF using PyMuPDF (fitz).

    Algorithm:
      1. For each page, use page.get_text("words") to get every word with its
         (x, y) bounding box.
      2. Cluster words into rows by y-coordinate.
      3. Find the header row containing "Marcus Hersh ... HANDICAP ... CONSENSUS".
      4. Use header word x-coordinates (and the sub-header line with "Kenny Peck"
         and "Michael Hammersly") to lock the 4 expert column x-starts.
      5. Find race-number marker rows (single 1-2 digit number at far left) and
         walk i-1 / i / i+1 row triples, bucketing each row's words into columns
         by x-coordinate.

    Returns dict {expert_name: {race_num: [pick1_name, pick2_name, pick3_name]}},
    with expert_name in {marcus_hersh, kenny_peck, michael_hammersly, sweep}.
    Returns None if the panel cannot be located.
    """
    try:
        import fitz
    except ImportError:
        print("(need pymupdf for ProPick PDF parsing — pip install pymupdf)", end=' ')
        return None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"(could not open PDF: {e})", end=' ')
        return None

    try:
        for page in doc:
            # get_text("words") -> list of (x0, y0, x1, y1, "text", block, line, word)
            raw_words = page.get_text("words")
            words = [{'text': w[4], 'x': w[0], 'y': w[1]} for w in raw_words]
            if not words:
                continue

            joined = ' '.join(w['text'] for w in words)
            if not ('Picks Panel' in joined and 'Marcus Hersh' in joined
                    and 'CONSENSUS' in joined):
                continue

            # --- Cluster words into rows by y-coordinate ---
            words.sort(key=lambda w: (w['y'], w['x']))
            rows = []
            ROW_TOL = 3.0  # points; rows within ~3pt vertically are considered the same line
            for w in words:
                if rows and abs(w['y'] - rows[-1][0]['y']) <= ROW_TOL:
                    rows[-1].append(w)
                else:
                    rows.append([w])
            for r in rows:
                r.sort(key=lambda w: w['x'])

            # --- Find header row 1 (Marcus Hersh ... HANDICAP ... CONSENSUS) ---
            h1_idx = None
            for i, row in enumerate(rows):
                text = ' '.join(w['text'] for w in row)
                if 'Marcus' in text and 'HANDICAP' in text and 'CONSENSUS' in text:
                    h1_idx = i
                    break
            if h1_idx is None or h1_idx + 1 >= len(rows):
                continue

            h1 = rows[h1_idx]
            h2 = rows[h1_idx + 1]  # has "Kenny Peck" and "Michael Hammersly"

            def x_of(row, target):
                for w in row:
                    if w['text'] == target:
                        return w['x']
                return None

            hersh_x = x_of(h1, 'Marcus')
            peck_x  = x_of(h2, 'Kenny')   or x_of(h1, 'HANDICAP')
            ham_x   = x_of(h2, 'Michael') or x_of(h1, 'HERMIS')
            sweep_x = x_of(h1, 'SWEEP')
            cons_x  = x_of(h1, 'CONSENSUS')

            if any(v is None for v in (hersh_x, peck_x, ham_x, sweep_x, cons_x)):
                continue

            # Column x-ranges. Small left tolerance handles minor word-position drift.
            TOL = 5.0
            bounds = [
                ('marcus_hersh',      hersh_x - TOL, peck_x  - TOL),
                ('kenny_peck',        peck_x  - TOL, ham_x   - TOL),
                ('michael_hammersly', ham_x   - TOL, sweep_x - TOL),
                ('sweep',             sweep_x - TOL, cons_x  - TOL),
            ]

            # --- Find race-number marker rows ---
            # The middle row of each 3-row race block has a 1-2 digit race number
            # at the far left (well left of the Hersh column).
            race_markers = []
            for i in range(h1_idx + 2, len(rows)):
                for w in rows[i]:
                    if w['x'] < hersh_x - 10 and re.fullmatch(r'\d{1,2}', w['text']):
                        n = int(w['text'])
                        if 1 <= n <= 14:
                            race_markers.append((n, i))
                        break

            # --- Walk row triples and bucket words into columns by x-coord ---
            panel = {name: {} for name, _, _ in bounds}
            for race_num, mid_idx in race_markers:
                triple_indices = [mid_idx - 1, mid_idx, mid_idx + 1]
                triple = [rows[i] for i in triple_indices
                          if 0 <= i < len(rows)]
                for name, x_min, x_max in bounds:
                    picks = []
                    for row in triple:
                        cell_words = [w['text'] for w in row
                                      if x_min <= w['x'] < x_max]
                        if cell_words:
                            picks.append(' '.join(cell_words).strip())
                    if picks:
                        panel[name][race_num] = picks

            if any(panel[n] for n, _, _ in bounds):
                return panel
    finally:
        doc.close()

    return None


def _extract_hersh_prose(pdf_path):
    """Extract Marcus Hersh's "Expert Analysis" prose section from a propick PDF.

    Returns the prose as a string suitable for sending to Sonnet, or None if
    no analysis section is found (e.g. PDF doesn't include Hersh's write-up).
    """
    try:
        import fitz
    except ImportError:
        return None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None

    try:
        full = '\n'.join(page.get_text() for page in doc)
    finally:
        doc.close()

    # The analysis section starts with "Expert Analysis" followed by "By Marcus Hersh"
    # and runs to end of document.
    m = re.search(r'Expert Analysis\s*\n.*?By Marcus Hersh', full, re.DOTALL)
    if not m:
        return None
    prose = full[m.start():]

    # Trim page-number footers like "Page 5 of 6"
    prose = re.sub(r'\n\s*Page \d+ of \d+\s*\n', '\n', prose)
    # Collapse excessive blank lines
    prose = re.sub(r'\n{3,}', '\n\n', prose)
    prose = prose.strip()

    return prose if len(prose) >= 200 else None


def _panel_picks_to_rows(picks_by_race, expert_name, name_lookup):
    """Convert {race: [pick1_name, pick2_name, pick3_name]} into CSV row dicts.

    Mapping: 1st -> +0.7 best_bet, 2nd -> +0.5 strong, 3rd -> +0.3 mentioned.
    Skips any pick whose name can't be resolved to a program number via name_lookup
    (e.g. scratched horses or country-suffix mismatches not handled by normalization).
    """
    deltas = [(0.7, 'best_bet'), (0.5, 'strong'), (0.3, 'mentioned')]
    rows = []
    unmatched = []
    for race in sorted(picks_by_race.keys()):
        picks = picks_by_race[race]
        for i, name in enumerate(picks[:3]):
            delta, sentiment = deltas[i]
            key = (race, _normalize_horse_name(name))
            if name_lookup is None or key not in name_lookup:
                unmatched.append((race, name))
                continue
            rows.append({
                'race': race,
                'horse_number': str(name_lookup[key]),
                'logit_delta': delta,
                'sentiment': sentiment,
                'expert': expert_name,
            })
    return rows, unmatched


# =============================================================================
# STRUCTURED CSV TIER FILES (top / A / B columns, no LLM needed)
#
# Some experts publish their picks as structured CSVs with a tier-per-column
# format. Example:
#     Race,top,A,B,Comments
#     1,4,"1,5",,Win bet on #4
#     2,,"3,7,9",,
#     3,,"5,6,9","2,10",
#
# Each column maps to a logit_delta tier:
#     top → +0.7   (best_bet)
#     A   → +0.5   (strong)
#     B   → +0.3   (mentioned)
#     C   → +0.15  (contender) — used by 4-tier sources like FatBaldGuyRacing
#
# These files are parsed deterministically — no API call. The expected
# schema is a CSV with a Race column and at least one of {top, A, B, C}.
# Comments / extra columns are ignored. Cells can be empty, a single
# program number ("4"), or a quoted comma-separated list ("1,5,7").
# Coupled-entry letters (1A, 2B) are allowed.
#
# If the file doesn't have the expected schema, it's skipped with a
# warning (not sent to the API — structured CSVs should be fixed by
# editing, not reinterpreted by an LLM).
# =============================================================================

# Tier column → (logit_delta, sentiment) mapping. Lookup is case-
# insensitive and tolerates leading/trailing whitespace in column names.
# 'tp' (Top Pick) is an accepted alias for 'top' since some sources
# (e.g. extracted Twitter-image picks) label that column "TP".
# 'c' (Consider / Contender) is the lowest-conviction tier — used by
# sources like FatBaldGuyRacing that publish four tiers: top/A/B/C.
# Most sources publish three (top/A/B); C is silently absent for them.
_CSV_TIER_MAP = {
    'top': (0.7,  'best_bet'),
    'tp':  (0.7,  'best_bet'),  # alias for 'top'
    'a':   (0.5,  'strong'),
    'b':   (0.3,  'mentioned'),
    'c':   (0.15, 'contender'),
}


def _parse_tier_cell(cell):
    """Parse a CSV cell containing program number(s).

    Accepts: empty/NaN ('' or 'nan'), a single number ('4'), a comma-
    separated list ('1,5,7'), and coupled-entry letters ('1A', '2B').
    Returns a list of validated program-number strings (preserving
    coupled-entry letters), or [] for empty cells.

    Skips any token that doesn't match ^\\d+[A-Za-z]?$ — preserves
    the same horse-number validation the LLM-output parser uses, so a
    typo or stray text in a cell silently becomes a no-op rather than
    crashing.
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == 'nan':
        return []
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if re.match(r'^\d+[A-Za-z]?$', tok):
            out.append(tok)
    return out


def _read_structured_csv(csv_path):
    """Read a tier-format CSV. Returns list of row dicts ready for
    _write_expert_csv, or None if the schema doesn't match.

    Schema requirements:
      - 'Race' column (case-insensitive). Each row's Race must parse
        to a positive integer.
      - At least one of {top, TP, A, B, C} (case-insensitive). Missing
        tier columns are silently skipped (no rows for that tier).
        'TP' is treated as an alias for 'top' (Top Pick).
        'C' is the lowest-conviction tier (+0.15 / contender) — used
        by 4-tier sources; absent from most.

    Other columns ('Comments', blanks from trailing commas, etc.)
    are ignored entirely.

    Returns None if the schema is unusable. The caller should call
    _diagnose_csv_failure(csv_path) to print a SPECIFIC reason — this
    function returns None for several distinct failure modes (file
    unreadable, empty, missing Race column, missing all tier columns,
    every row's Race value invalid) and the caller deserves to know
    which one happened.
    """
    expert_name = csv_path.stem
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if df.empty:
        return None

    cols_lower = {c.strip().lower(): c for c in df.columns if c and c.strip()}
    race_col = cols_lower.get('race')
    if race_col is None:
        return None

    # Build the list of tier columns to read. _CSV_TIER_MAP can have
    # multiple keys mapping to the same sentiment (e.g. 'top' and 'tp'
    # both → best_bet). Dedupe by sentiment so a CSV that somehow has
    # both columns doesn't double-count a horse listed in both.
    tier_cols = []  # [(actual_col_name, logit_delta, sentiment)]
    seen_sentiments = set()
    for tier_key, (delta, sentiment) in _CSV_TIER_MAP.items():
        if tier_key in cols_lower and sentiment not in seen_sentiments:
            tier_cols.append((cols_lower[tier_key], delta, sentiment))
            seen_sentiments.add(sentiment)
    if not tier_cols:
        return None

    rows = []
    for _, raw in df.iterrows():
        race_val = str(raw[race_col]).strip()
        if not race_val or race_val.lower() == 'nan':
            continue
        try:
            race_num = int(float(race_val))
        except (ValueError, TypeError):
            continue
        if race_num < 1 or race_num > 99:
            continue

        for col, delta, sentiment in tier_cols:
            for hn in _parse_tier_cell(raw.get(col, '')):
                rows.append({
                    'race': race_num,
                    'horse_number': hn,
                    'logit_delta': round(delta, 2),
                    'sentiment': sentiment,
                    'expert': expert_name,
                })

    return rows


def _diagnose_csv_failure(csv_path):
    """When _read_structured_csv returns None, figure out which specific
    failure mode hit and return a human-readable explanation. Used by
    callers (image extraction validation, structured-CSV pass) to print
    actionable error messages instead of a generic "schema mismatch"
    that's wrong about half the time it fires.

    Returns a string describing exactly what's wrong.
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return "file is completely empty (no header, no rows)"
    except pd.errors.ParserError as e:
        return f"file is unparseable as CSV: {e}"
    except Exception as e:
        return f"could not read file: {type(e).__name__}: {e}"

    cols_found = [c for c in df.columns if c and c.strip()]
    cols_lower = {c.strip().lower(): c for c in df.columns if c and c.strip()}

    if df.empty:
        return (f"file has the header columns {cols_found} but ZERO data rows. "
                f"This usually means an extraction (e.g. vision API) returned "
                f"only the header without any picks.")
    if 'race' not in cols_lower:
        return (f"missing 'Race' column. Found columns: {cols_found}. "
                f"Need 'Race' (case-insensitive) plus at least one of "
                f"{{top, TP, A, B, C}}.")
    matched_tiers = [k for k in _CSV_TIER_MAP if k in cols_lower]
    if not matched_tiers:
        return (f"no tier columns found. Found columns: {cols_found}. "
                f"Need at least one of {{top, TP, A, B, C}} (case-insensitive).")

    # Schema looks fine; rows must be present but every row's Race
    # value failed to parse as a valid 1-99 integer.
    return (f"schema looks correct (columns={cols_found}, {len(df)} data rows) "
            f"but every row's 'Race' value failed to parse as a valid race "
            f"number 1-99. Check that the Race column contains plain integers.")


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


def _write_expert_csv(rows, out_dir, expert_name):
    """Write expert rows to {out_dir}/{expert_name}.csv and return summary stats."""
    out_path = out_dir / f"{expert_name}.csv"
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['race', 'horse_number', 'logit_delta', 'sentiment', 'expert'])
        writer.writeheader()
        writer.writerows(rows)
    races = len(set(r['race'] for r in rows))
    best_bets = sum(1 for r in rows if r['sentiment'] == 'best_bet')
    negatives = sum(1 for r in rows if r['logit_delta'] < 0)
    return races, best_bets, negatives


# =============================================================================
# IMAGE EXTRACTION (vision API → structured tier CSV)
#
# Some sources publish picks as images (Twitter screenshots, posted PNGs,
# etc.) with the same per-race tier structure as the CSV format. Rather
# than build a separate parsing path for these, we use Sonnet's vision
# capability to extract them into the SAME structured CSV format the
# CSV pass already handles. The flow is:
#
#   image.png in raw_dir
#       → vision API extracts tiered picks, writes image.csv to raw_dir
#       → image.png moves to sent_dir
#       → structured-CSV pass picks up image.csv on the same run,
#         processes deterministically, writes expert_picks/image.csv,
#         moves image.csv to sent_dir
#
# Two artifacts end up in sent_dir per source image: the original .png
# (so we don't re-extract on the next run) and the intermediate .csv
# (so the user can inspect what the vision API produced).
# =============================================================================

# Image extensions the Anthropic vision API accepts.
_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}

# Vision prompt is its own thing — focused, not the prose-handling SYSTEM_PROMPT.
# Asks for the exact structured-CSV schema so the output drops straight into
# the existing CSV pipeline with no additional parsing.
IMAGE_EXTRACTION_PROMPT = """You are extracting horse racing picks from an image \
of a tiered picks table (the kind that handicappers post on Twitter or in newsletters).

The image contains a table where each row is a race and columns are tiers of picks.
Tiers may be labeled differently across sources, but the meaning is consistent:

  - "Top Pick" / "TP" / "Best Bet" / "Win" / "1st"          → the TOP tier
  - "A" / "A horses" / "Use" / "Strong" / "2nd"             → the A tier
  - "B" / "B horses" / "Backup" / "Spread"                  → the B tier
  - "C" / "C horses" / "Consider" / "Contender" / "Toss-in" → the C tier (lowest)

Most sources publish three tiers (top + A + B). Some publish four (adding C).
A handful publish only one or two. ALWAYS output all four tier columns in your
response, even if the source has fewer — just leave the absent tiers empty.

Cells contain program numbers (1, 2, 3...). Multiple picks per cell are separated \
by commas WITHIN the cell. Empty cells mean no pick at that tier for that race. \
Cells with non-numeric content like "abs", "tai", "TBD", "—", "scratched" mean no \
pick (treat as empty). DO NOT skip the race row just because a cell has non-numeric \
content — output the row with empty values for those tiers.

OUTPUT FORMAT — pipe-delimited, one row per race:

The first line of your response must be exactly:
    Race|top|A|B|C

Each subsequent line is one race, with fields separated by pipe characters (|):
    <race_number>|<top_picks>|<A_picks>|<B_picks>|<C_picks>

Why pipes (not commas): cells often contain multiple picks separated by commas \
within the cell (e.g. "3,6,7"). Using `|` as the field separator means we never \
have to quote anything — pipe never appears inside a cell, comma never breaks the row.

Examples:
    1|1||3,6,7|           → Race 1: top=1, A=empty, B=3,6,7, no C tier (3-tier source)
    2|5|5||               → Race 2: top=5, A=5, B=empty, no C tier
    3|3|2,3,9|6|8,5       → Race 3: top=3, A=2,3,9, B=6, C=8,5 (4-tier source)
    6||||                 → Race 6: all tiers empty (still output the row!)

Even if the source image has tier columns in a different order (e.g. A,B,TP) or \
uses different column names (e.g. "TP" instead of "top"), REMAP them to the output \
order Race|top|A|B|C in your response.

Strict rules:
- Output one row PER RACE in the image, even if all four tier cells are empty.
- Within a cell, separate multiple picks with commas, no spaces (e.g. `2,3,9` not `2, 3, 9`).
- Empty cells: nothing between the pipes. Examples above.
- ALWAYS output 4 tier fields (top, A, B, C) — empty for missing tiers.
- Coupled entries with letters (1A, 2B, etc.) are valid program numbers — preserve them.
- Plain pipe-delimited text ONLY. No markdown code fences (no ```...```). No prose, \
  no explanations, no preamble, no postamble. No commas as field separators — \
  commas appear ONLY inside cells, between picks.
- The first line of your response must be exactly: Race|top|A|B|C

If you cannot read ANY data from the table, respond with exactly the single word \
`INVALID` and nothing else. But if you can read SOME data (even just a few rows), \
output what you can read — partial extraction is better than nothing."""


def extract_picks_from_image(image_path, model='claude-sonnet-4-20250514'):
    """Send an image to Sonnet's vision API and request a structured CSV
    of tiered picks (Race / top / A / B / C). Returns the raw CSV text from
    the response, or the literal string 'INVALID' if the model couldn't
    read the image as a picks table.

    Caller is responsible for writing the CSV to disk and validating
    that it parses via _read_structured_csv. If parsing fails, the
    caller can leave the .csv in place and let the user inspect/fix.
    """
    import base64
    import mimetypes

    client = Anthropic(
        api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    )

    media_type, _ = mimetypes.guess_type(str(image_path))
    if media_type is None:
        # Fallback: derive from extension. Anthropic accepts these specific types.
        ext = image_path.suffix.lower()
        media_type = {
            '.png':  'image/png',
            '.jpg':  'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif':  'image/gif',
            '.webp': 'image/webp',
        }.get(ext, 'image/png')

    with open(image_path, 'rb') as f:
        image_b64 = base64.standard_b64encode(f.read()).decode('utf-8')

    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': media_type,
                        'data': image_b64,
                    },
                },
                {
                    'type': 'text',
                    'text': IMAGE_EXTRACTION_PROMPT,
                },
            ],
        }],
    )

    return resp.content[0].text.strip()


def _process_image_picks(raw_dir, sent_dir, model):
    """Find image files (PNG/JPG/etc) in raw_dir, extract via vision API,
    write the result as a structured CSV in raw_dir for the CSV pass to
    process on the same run.

    Each image yields ONE intermediate .csv file with the same stem.
    On success, the original image moves to sent_dir (the .csv moves
    too, but only after the structured-CSV pass processes it).

    Returns the number of CSVs successfully written (NOT the number
    of pick rows — those will be counted by the CSV pass).

    Skip conditions (image stays in raw_dir):
      - target .csv stem already exists (don't overwrite a manually-
        edited or previously-extracted CSV)
      - vision API returns 'INVALID' (couldn't read it as a picks table)
      - extracted text doesn't validate as a structured CSV
    """
    import shutil

    image_files = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        and not (sent_dir / p.name).exists()
    ]
    if not image_files:
        return 0

    print(f"\n  Found {len(image_files)} image(s) to extract via vision API")
    n_extracted = 0

    for img_path in sorted(image_files):
        target_csv = raw_dir / f'{img_path.stem}.csv'
        print(f"\n  IMAGE: {img_path.name}  →  {target_csv.name}")

        if target_csv.exists():
            print(f"    Target CSV already exists ({target_csv.name}); "
                  f"leaving image in place to avoid overwrite.")
            print(f"    (Delete or rename {target_csv.name} to re-extract.)")
            continue

        try:
            raw_text = extract_picks_from_image(img_path, model=model)
        except Exception as e:
            print(f"    Vision API error: {e}")
            continue

        if raw_text.strip().upper() == 'INVALID':
            print(f"    Vision API returned INVALID — image doesn't look like "
                  f"a picks table. Leaving image in place.")
            continue

        # The vision prompt asks for pipe-delimited output (Race|top|A|B)
        # because cells contain commas (e.g. "3,6,7" for three picks). Pipes
        # eliminate the quoting problem entirely — pipe never appears inside
        # a pick cell, so we can split on `|` without worrying about quoting.
        # We then write a proper standard CSV (with auto-quoting via
        # pandas.to_csv) so the structured-CSV pass reads it correctly.
        cleaned_lines = []
        for line in raw_text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('```'):
                continue
            # Skip markdown table separator rows (|---|---|).
            # A real data row will have digits, so won't match this.
            if re.match(r'^[\s|:.\-]+$', stripped):
                continue
            cleaned_lines.append(stripped)

        if not cleaned_lines:
            print(f"    Vision response was empty after cleanup.")
            print(f"    Raw response (first 300 chars):")
            for raw_line in raw_text[:300].split('\n'):
                print(f"      {raw_line}")
            print(f"    Leaving image in place.")
            continue

        # Validate the first line is the expected pipe-delimited header.
        # Allow case-insensitive 'race|' since the model occasionally drifts.
        header_line = cleaned_lines[0].lower()
        if not header_line.startswith('race|'):
            print(f"    Extraction didn't produce expected pipe-delimited header.")
            print(f"    First cleaned line: {cleaned_lines[0][:80]!r}")
            print(f"    (Expected something like: 'Race|top|A|B')")
            print(f"    Raw response (first 300 chars):")
            for raw_line in raw_text[:300].split('\n'):
                print(f"      {raw_line}")
            print(f"    Leaving image in place.")
            continue

        # Parse pipe-delimited body. Each line should have the same number
        # of fields as the header. Skip lines that don't (with a warning so
        # the user knows the model partially fumbled).
        header_fields = [f.strip() for f in cleaned_lines[0].split('|')]
        n_cols = len(header_fields)
        data_rows = []
        skipped_lines = []
        for line in cleaned_lines[1:]:
            fields = [f.strip() for f in line.split('|')]
            if len(fields) != n_cols:
                skipped_lines.append((line, len(fields)))
                continue
            data_rows.append(fields)

        if skipped_lines:
            print(f"    {len(skipped_lines)} line(s) had wrong field count "
                  f"(expected {n_cols}); skipping them:")
            for line, n_got in skipped_lines[:5]:
                print(f"      got {n_got} fields: {line[:80]}")
            if len(skipped_lines) > 5:
                print(f"      ... and {len(skipped_lines)-5} more")

        if not data_rows:
            print(f"    No valid data rows extracted from vision response.")
            print(f"    Raw response (first 300 chars):")
            for raw_line in raw_text[:300].split('\n'):
                print(f"      {raw_line}")
            print(f"    Leaving image in place.")
            continue

        # Build a DataFrame and write as proper CSV with auto-quoting.
        # pd.to_csv defaults to QUOTE_MINIMAL, which wraps any field that
        # contains the delimiter (",") in double-quotes. Result: cells
        # like "3,6,7" become `"3,6,7"`, which the structured-CSV pass
        # parses correctly.
        try:
            df_extracted = pd.DataFrame(data_rows, columns=header_fields)
            df_extracted.to_csv(target_csv, index=False, encoding='utf-8')
        except Exception as e:
            print(f"    Could not write {target_csv.name}: {e}")
            continue

        # Quick validation: did we actually produce a parseable CSV?
        rows = _read_structured_csv(target_csv)
        if rows is None:
            # Use the diagnostic helper for a specific reason, and dump
            # the model's raw response so the user can see what the
            # vision API actually said.
            print(f"    Extracted CSV failed schema validation.")
            print(f"    Reason: {_diagnose_csv_failure(target_csv)}")
            print(f"    Raw vision response (first 500 chars):")
            for raw_line in raw_text[:500].split('\n'):
                print(f"      {raw_line}")
            print(f"    Leaving image and CSV in place for inspection.")
            continue
        if not rows:
            print(f"    Extracted CSV has 0 picks (every cell was empty?); "
                  f"leaving image and CSV in place for inspection.")
            continue

        # Quick validation: did we actually produce a parseable CSV?
        rows = _read_structured_csv(target_csv)
        if rows is None:
            # Use the diagnostic helper for a specific reason, and dump
            # the model's raw response so the user can see what the
            # vision API actually said.
            print(f"    Extracted CSV failed schema validation.")
            print(f"    Reason: {_diagnose_csv_failure(target_csv)}")
            print(f"    Raw vision response (first 500 chars):")
            for raw_line in raw_text[:500].split('\n'):
                print(f"      {raw_line}")
            print(f"    Leaving image and CSV in place for inspection.")
            # Note: we leave the .csv in raw_dir. The structured-CSV pass
            # will re-validate it and print the same warning, which is
            # repetitive but not harmful.
            continue
        if not rows:
            print(f"    Extracted CSV has 0 picks (every cell was empty?); "
                  f"leaving image and CSV in place for inspection.")
            continue

        # Success: move the image to sent. (The CSV will be moved by
        # the structured-CSV pass when it processes it.)
        shutil.move(str(img_path), str(sent_dir / img_path.name))
        print(f"    Extracted {len(rows)} picks; image moved to "
              f"already_sent_eps. CSV will be processed in next pass.")
        n_extracted += 1

    return n_extracted


def _process_structured_csvs(raw_dir, sent_dir, out_dir):
    """Find and process tier-format CSV files in raw_dir.

    Each file is one expert. Filename stem becomes the expert name.
    Schema: Race column + at least one of {top, A, B}. Other columns
    are ignored. Cells are program numbers, possibly comma-separated.

    Files that don't match the schema are skipped with a warning and
    LEFT IN PLACE — they're probably mis-formatted and the user should
    fix them. We never send a CSV to the API; the contract is
    "structured tier file or not processed at all."

    Returns total rows written across all CSVs. Moves processed CSVs
    to sent_dir on success.
    """
    import shutil

    csv_files = [
        p for p in raw_dir.glob('*.csv')
        if not (sent_dir / p.name).exists()
    ]
    if not csv_files:
        return 0

    print(f"\n  Found {len(csv_files)} structured CSV(s) to process")
    total_rows = 0

    for csv_path in csv_files:
        expert_name = csv_path.stem
        print(f"\n  CSV: {csv_path.name}  (expert={expert_name})")

        rows = _read_structured_csv(csv_path)
        if rows is None:
            # Schema mismatch. The diagnostic helper inspects the file
            # again to figure out the SPECIFIC reason — generic "expected
            # vs found columns" lies when the columns match but rows are
            # missing or unparseable.
            print(f"    Schema mismatch — leaving file in place.")
            print(f"    Reason: {_diagnose_csv_failure(csv_path)}")
            continue
        if not rows:
            # Schema OK but every row was empty/invalid — typically a
            # CSV where all picks-cells are blank. Treat as a soft skip.
            print(f"    No picks found in any row — leaving file in place.")
            continue

        races, best_bets, _ = _write_expert_csv(rows, out_dir, expert_name)
        # CSV files are deterministic, so any negative-tier mapping would
        # have to come from a future schema; for now, all tiers are
        # positive (top/A/B), so we don't print a negatives count.
        n_tiers = len(set(r['sentiment'] for r in rows))
        print(f"    {len(rows)} picks across {races} races, "
              f"{best_bets} best bets, {n_tiers} tier(s) used [from CSV]")
        total_rows += len(rows)

        # Move the file out of raw_dir into already_sent.
        shutil.move(str(csv_path), str(sent_dir / csv_path.name))

    return total_rows


def _process_propick_pdfs(raw_dir, sent_dir, out_dir, track, date_str,
                          entries_ref, name_lookup, model):
    """Find and process any DRF ProPick bundle PDFs (filenames matching propick*.pdf,
    case-insensitive) in raw_dir.

    Each propick PDF yields up to 4 expert CSVs:
      - marcus_hersh:      from prose write-up via Sonnet (if prose found),
                           else from picks panel (deterministic)
      - kenny_peck:        from picks panel (deterministic)
      - michael_hammersly: from picks panel (deterministic)
      - sweep:             from picks panel (deterministic)

    Returns total rows written across all 4 experts. Moves processed PDFs to sent_dir.
    """
    import shutil

    # Case-insensitive glob for propick*.pdf
    propick_files = [
        p for p in raw_dir.glob('*.pdf')
        if p.stem.lower().startswith('propick') and not (sent_dir / p.name).exists()
    ]
    if not propick_files:
        return 0

    print(f"\n  Found {len(propick_files)} ProPick bundle(s) to process")
    total_rows = 0

    for pdf_path in propick_files:
        print(f"\n  ProPick: {pdf_path.name}")

        panel = _parse_propick_panel(pdf_path)
        if not panel:
            print(f"    Could not parse picks panel — skipping (file left in place)")
            continue

        if name_lookup is None:
            print(f"    ERROR: no entries lookup available — cannot map names to "
                  f"program numbers. Skipping (file left in place).")
            continue

        # --- Panel-only experts: Peck, Hammersly, Sweep ---
        for expert_name in ['kenny_peck', 'michael_hammersly', 'sweep']:
            picks_by_race = panel.get(expert_name, {})
            if not picks_by_race:
                print(f"    {expert_name}: no picks found in panel, skipping")
                continue
            rows, unmatched = _panel_picks_to_rows(
                picks_by_race, expert_name, name_lookup)
            if not rows:
                print(f"    {expert_name}: no rows after name lookup "
                      f"({len(unmatched)} unmatched)")
                continue
            races, best_bets, _ = _write_expert_csv(rows, out_dir, expert_name)
            tag = f" ({len(unmatched)} unmatched)" if unmatched else ""
            print(f"    {expert_name}: {len(rows)} picks across {races} races, "
                  f"{best_bets} best bets [from panel]{tag}")
            total_rows += len(rows)

        # --- Marcus Hersh: prose -> Sonnet if available, else panel fallback ---
        hersh_prose = _extract_hersh_prose(pdf_path)
        if hersh_prose:
            print(f"    marcus_hersh: prose found ({len(hersh_prose)} chars), "
                  f"sending to Sonnet...", end=' ')
            try:
                raw_response, rows = extract_deltas(
                    hersh_prose, 'marcus_hersh',
                    entries_ref=entries_ref, model=model)
            except Exception as e:
                print(f"API error: {e}")
                rows = None
                raw_response = ''
            if rows:
                races, best_bets, negatives = _write_expert_csv(
                    rows, out_dir, 'marcus_hersh')
                print(f"{len(rows)} picks across {races} races "
                      f"({best_bets} best bets, {negatives} negatives) [from prose]")
                total_rows += len(rows)
            else:
                print("no picks extracted from prose")
                if raw_response:
                    print(f"      Raw response: {raw_response[:200]}")
        else:
            # No prose — fall back to panel for Hersh as well
            picks_by_race = panel.get('marcus_hersh', {})
            if picks_by_race:
                rows, unmatched = _panel_picks_to_rows(
                    picks_by_race, 'marcus_hersh', name_lookup)
                if rows:
                    races, best_bets, _ = _write_expert_csv(
                        rows, out_dir, 'marcus_hersh')
                    tag = f" ({len(unmatched)} unmatched)" if unmatched else ""
                    print(f"    marcus_hersh: no prose found — fell back to panel: "
                          f"{len(rows)} picks across {races} races, "
                          f"{best_bets} best bets [from panel]{tag}")
                    total_rows += len(rows)
            else:
                print(f"    marcus_hersh: no prose and no panel picks — skipped")

        # Move propick PDF to sent_dir
        shutil.move(str(pdf_path), str(sent_dir / pdf_path.name))

    return total_rows


def process_experts(date_str, track, model='claude-sonnet-4-20250514'):
    """Process all raw expert .txt, .pdf, .csv, and image files for a date/track.

    Handles four kinds of inputs:
      1. DRF ProPick bundle PDFs (filenames starting with 'propick'): parsed
         locally — the picks-panel grid (Peck, Hammersly, Sweep) is converted
         deterministically, and only Hersh's prose write-up is sent to Sonnet.
      2. Image files (.png, .jpg, .jpeg, .gif, .webp): sent to Sonnet's vision
         API to extract a structured tier CSV. The intermediate CSV is then
         processed by pass 3.
      3. Structured tier CSVs (Race + top/TP/A/B/C columns): parsed locally,
         no API call. Each tier maps to a logit_delta:
            top/TP → +0.7  A → +0.5  B → +0.3  C → +0.15
      4. Single-expert .txt or .pdf files (one expert per file): full text sent
         to Sonnet for sentiment extraction.
    """
    raw_dir = DATA_PATH / 'raw_expert_picks' / date_str / track
    sent_dir = DATA_PATH / 'already_sent_eps' / date_str / track
    out_dir = DATA_PATH / 'expert_picks' / date_str / track
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"No raw picks found at {raw_dir}")
        print(f"Run: python scrape_experts.py --track {track} --date {date_str}")
        print(f"Or manually place .txt / .pdf / .csv / image files in {raw_dir}")
        return

    # Find any input file the four passes know how to handle.
    raw_files = (list(raw_dir.glob('*.txt'))
                 + list(raw_dir.glob('*.pdf'))
                 + list(raw_dir.glob('*.csv'))
                 + [p for p in raw_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS])
    raw_files = sorted(set(raw_files))
    if not raw_files:
        print(f"No .txt / .pdf / .csv / image files in {raw_dir}")
        return

    sent_dir.mkdir(parents=True, exist_ok=True)

    # Load horse references (used by both propick and single-expert flows;
    # CSVs don't need them — picks are program numbers, not names)
    entries_ref = _load_entries_reference(track, date_str)
    name_lookup = _load_entries_lookup(track, date_str)
    if entries_ref:
        n_races = entries_ref.count('\n') + 1
        print(f"Horse reference: {n_races} races loaded "
              f"({len(name_lookup) if name_lookup else 0} name->number entries)")
    else:
        print(f"WARNING: No horse reference found — name→number mapping may fail")
        print(f"  Run: python download_upc.py --track {track} --date {date_str}")

    total_rows = 0

    # ----- Pass 1: ProPick bundle PDFs (parsed locally; only Hersh prose -> API) -----
    total_rows += _process_propick_pdfs(
        raw_dir, sent_dir, out_dir, track, date_str,
        entries_ref, name_lookup, model)

    # ----- Pass 2: Image extraction (vision API → intermediate CSVs) -----
    # Each image yields a .csv in raw_dir that pass 3 then processes.
    # Run BEFORE pass 3 so the same run handles both extraction and parsing.
    _process_image_picks(raw_dir, sent_dir, model)

    # ----- Pass 3: Structured tier CSVs (parsed locally; no API) -----
    # Includes both pre-existing CSVs and any produced by pass 2 just now.
    total_rows += _process_structured_csvs(raw_dir, sent_dir, out_dir)

    # ----- Pass 4: Standalone expert .txt / .pdf files (full text -> API) -----
    # Re-glob in case earlier passes moved files. Skip propick* files (handled
    # in pass 1), .csv files (handled in pass 3 — leftover .csv files here
    # mean schema-mismatched files we left in place; don't re-process them),
    # image files (handled in pass 2), and anything already in sent_dir.
    raw_files = sorted(list(raw_dir.glob('*.txt')) + list(raw_dir.glob('*.pdf')))
    to_process = []
    skipped = 0
    for raw_path in raw_files:
        if raw_path.stem.lower().startswith('propick'):
            continue  # propick bundles handled in pass 1; if still here, parse failed
        if (sent_dir / raw_path.name).exists():
            skipped += 1
        else:
            to_process.append(raw_path)

    if skipped:
        print(f"\nSkipping {skipped} already-sent expert(s) (found in already_sent_eps/)")

    if to_process:
        print(f"\nProcessing {len(to_process)} standalone expert file(s) "
              f"for {date_str} / {track}")
        print(f"  Model: {model}")
        print(f"  Output: {out_dir}/")

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

        races, best_bets, negatives = _write_expert_csv(rows, out_dir, expert_name)

        # Move raw file to already_sent
        import shutil
        shutil.move(str(raw_path), str(sent_dir / raw_path.name))

        print(f"{len(rows)} picks across {races} races "
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