"""
scrape_tipmeerkat.py

Fetch community picks from tipmeerkat.com and write logit-delta CSVs
directly into the expert_picks tree. Unlike scrape_experts.py, the
tipmeerkat page is structured enough that we don't need an LLM step —
each tipster posts an ordered top-3 with horse number, name, and odds.

Two outputs per (date, track):
  - One CSV per tracked tipster: tm_{handle}.csv
  - One consensus CSV aggregating crowd picks: tm_consensus.csv

CSV format matches what expert_elo.py expects:
    race,horse_number,logit_delta

Logit deltas applied to top-3 picks (1st/2nd/3rd):
  +0.75 / +0.50 / +0.25
(matches the scale used by your other scraped expert CSVs)

Tipster ratings are seeded into EXPERT_INITIAL_RATINGS via the helper
in this module (see export_seeds_for_elo). Mapping is based on each
tipster's "Odds Adjusted" score on their profile page, then pulled
toward 1500 by --shrinkage to mitigate leakage between tipmeerkat's
recent-form score and cards we're about to replay. Default shrinkage
is 0.5 (halve the spread). Raw mapping (before shrinkage):
  +1.0+        → 1700  (clear edge over odds — top tier)
  +0.7 .. +1.0 → 1625-1675
  +0.4 .. +0.7 → 1550-1625
  +0.1 .. +0.4 → 1500-1550
   0   .. +0.1 → 1500   (neutral)
  negative     → 1450-1500 (faded)

Usage:
    python scrape_tipmeerkat.py --track churchill-downs
    python scrape_tipmeerkat.py --track churchill-downs --date 2026-04-28
    python scrape_tipmeerkat.py --track churchill-downs --shrinkage 0.7
    python scrape_tipmeerkat.py --refresh-tipsters --track churchill-downs
    python scrape_tipmeerkat.py --reshrink-seeds --shrinkage 0.5
    python scrape_tipmeerkat.py --list-tipsters
    python scrape_tipmeerkat.py --export-seeds
    python scrape_tipmeerkat.py --leaderboard            # all experts
    python scrape_tipmeerkat.py --leaderboard --source tm --min-races 5
    python scrape_tipmeerkat.py --diagnose tm_great_scott   # see why a rating is what it is
    python scrape_tipmeerkat.py --diagnose scott_hazelton --only-track keeneland
    python scrape_tipmeerkat.py --migrate-csvs    # one-shot header/scale fix

Writes to:
    {DATA_PATH}/expert_picks/{date}/{track}/tm_{handle}.csv
    {DATA_PATH}/expert_picks/{date}/{track}/tm_consensus.csv
    {DATA_PATH}/tipmeerkat_tipsters.json    (id, handle, odds_adj, seed_rating)
"""

import os, re, json, time, argparse
from pathlib import Path
from datetime import datetime, date

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = Path(os.environ['DATA_PATH'])
TIPSTERS_PATH = DATA_PATH / 'tipmeerkat_tipsters.json'

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("pip install requests beautifulsoup4")

# pandas and numpy are only required by --diagnose. Imported lazily so
# the rest of the scraper still works on a minimal install.
try:
    import pandas as pd
    import numpy as np
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# =============================================================================
# CONFIG
# =============================================================================

BASE_URL = 'https://tipmeerkat.com'

# Logit deltas applied to a tipster's ordered top-3 picks.
# Matches the scale used by your other scraped expert CSVs so the
# tm_* experts and track-scraped experts can be compared directly.
PICK_DELTAS = [0.75, 0.5, 0.25]

# Track name → tipmeerkat slug. Mostly identical to your existing
# track names; add aliases here as you start scraping new tracks.
TRACK_SLUGS = {
    'churchill-downs':  'churchill-downs',
    'gulfstream-park':  'gulfstream',
    'gulfstream':       'gulfstream',
    'oaklawn':          'oaklawn-park',
    'oaklawn-park':     'oaklawn-park',
    'tampa':            'tampa-bay-downs',
    'tampa-bay-downs':  'tampa-bay-downs',
    'aqueduct':         'aqueduct',
    'saratoga':         'saratoga',
    'del-mar':          'del-mar',
    'fair-grounds':     'fair-grounds',
    'keeneland':        'keeneland',
    'belmont':          'belmont-park',
    'laurel':           'laurel-park',
    'parx':             'parx-racing',
    'parx-racing':      'parx-racing',
    'santa-anita':      'santa-anita',
}

# Minimum tips a tipster needs before we trust their odds-adj enough
# to seed them above 1500. Below this they get the neutral seed.
MIN_TIPS_FOR_SEED = 200


# =============================================================================
# HTTP
# =============================================================================

def _fetch(url, timeout=20):
    """GET with a browser-ish user agent. Be polite."""
    return requests.get(url, timeout=timeout, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })


# =============================================================================
# PARSING — meeting page (per-race tipster picks)
# =============================================================================

def _slugify_handle(name):
    """'Thorough Brett' → 'thorough_brett'; 'Charles Flippy' → 'charles_flippy'."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", '', name)
    return '_'.join(clean.lower().split())


def _parse_fractional_odds(s):
    """'8/5' → 1.6 (decimal odds excluding stake), '5' → 5.0. Returns None on fail."""
    if not s:
        return None
    s = s.strip()
    try:
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def parse_meeting_page(html, target_date_str):
    """
    Parse a tipmeerkat meeting page into a structured list of picks.

    Returns: {
      'date': 'YYYY-MM-DD',
      'tipsters': {
        tipster_id: {
          'handle': 'thorough_brett',
          'display': 'Thorough Brett',
          'points': 17300,
          'picks': {race_num: [(horse_num, horse_name, odds_str), ...]}
        }
      },
      'crowd_favorites': {race_num: [(horse_num, odds_str), ...]}
    }

    Strategy: rather than try to parse the DOM exactly (the page uses
    deeply nested divs and the same race section is rendered twice for
    SEO), we work from the *rendered text* of the page. The text layout
    is highly regular:

        Race N <time> ... Crowd favorites:
        <num>
        <odds>
        <num>
        <odds>
        <num>
        <odds>
        ... (tipster cards) ...
        Tipster Name
        N.Nk points
        <num>
        Horse Name
        <odds>
        <num>
        Horse Name
        <odds>
        ...

    To attribute picks to tipsters, we anchor on the tipster *links*
    (which give us id and display) and then split the rendered text
    on those anchors so each tipster owns the text up to the next
    tipster link.
    """
    soup = BeautifulSoup(html, 'html.parser')

    result = {
        'date': target_date_str,
        'tipsters': {},
        'crowd_favorites': {},
    }

    # ------------------------------------------------------------------
    # Pass 1: split rendered text into per-race chunks so we can locate
    # crowd favorites and assign tipster picks to the right race.
    # ------------------------------------------------------------------
    body = soup.body or soup
    full_text = body.get_text('\n', strip=True)

    # Find all "Race N" section starts and their offsets in the text.
    race_marker = re.compile(
        r'^Race\s+(\d+)\s+\d{1,2}:\d{2}\s*[AP]M', re.MULTILINE
    )
    race_offsets = [(int(m.group(1)), m.start()) for m in race_marker.finditer(full_text)]

    # For each race number keep only the first occurrence — the page
    # renders the same race in a top "popular picks" list and again
    # in the detailed cards section; the detailed one comes second
    # but is by far the longer chunk. Easier: we'll grab ALL chunks,
    # then merge results across duplicates.
    chunks = []  # list of (race_num, text_chunk)
    for i, (race_num, start) in enumerate(race_offsets):
        end = race_offsets[i + 1][1] if i + 1 < len(race_offsets) else len(full_text)
        chunks.append((race_num, full_text[start:end]))

    # ------------------------------------------------------------------
    # Pass 2: extract crowd favorites per race.
    # The text after "Crowd favorites:" is a sequence of (number, odds)
    # pairs, one per favored horse, before "The most popular bets" or
    # the first tipster card.
    # ------------------------------------------------------------------
    cf_block_re = re.compile(
        r'Crowd favorites:\s*\n'
        r'((?:\s*\d{1,2}\s*\n\s*\d+(?:/\d+)?\s*\n){1,6})',
        re.MULTILINE
    )
    pair_re = re.compile(r'(\d{1,2})\s*\n\s*(\d+(?:/\d+)?)')

    for race_num, chunk in chunks:
        if race_num in result['crowd_favorites']:
            continue
        m = cf_block_re.search(chunk)
        if not m:
            continue
        pairs = pair_re.findall(m.group(1))
        result['crowd_favorites'][race_num] = [(int(h), o) for h, o in pairs]

    # ------------------------------------------------------------------
    # Pass 3: partition the document into per-race regions using the
    # race h4 headers as boundaries. The detailed "card" section uses
    # h4s like:
    #     <h4><a href="#race-1-section">Race 1 12:45 PM ...</a></h4>
    # We assign each tipster link to a race based on document position
    # (every link between this race's h4 and the next race's h4).
    #
    # The page also renders a top "popular picks" summary which uses
    # the literal text "Race N" but typically without h4 wrapping. We
    # rely on h4 anchoring to avoid double-counting.
    # ------------------------------------------------------------------
    race_h4_re = re.compile(r'Race\s+(\d+)\s+\d{1,2}:\d{2}\s*[AP]M')

    # Build a sorted list of (doc_position, race_num) for race h4s.
    # We use the SourceLine/SourcePos if available, but more portable:
    # walk all elements in order and record the index when we hit one.
    # Easier: use find_all('h4') in document order (BS4 guarantees this).
    race_h4_positions = []  # list of (race_num, h4_element)
    for h4 in soup.find_all('h4'):
        txt = h4.get_text(' ', strip=True)
        m = race_h4_re.search(txt)
        if m:
            race_h4_positions.append((int(m.group(1)), h4))

    if not race_h4_positions:
        # Page may not have detailed sections (e.g. dark day). Bail.
        return result

    # For each h4, gather all /tipster/ links between this h4 and the
    # next race h4 in document order. We walk the descendants of the
    # nearest common ancestor; simplest correct approach: walk forward
    # via element.find_all_next() and stop when we hit the next h4.
    race_to_links = {}  # race_num -> [link_element, ...]
    for i, (race_num, h4) in enumerate(race_h4_positions):
        next_h4 = race_h4_positions[i + 1][1] if i + 1 < len(race_h4_positions) else None
        links = []
        for el in h4.find_all_next():
            if el is next_h4:
                break
            if el.name == 'a' and el.get('href') and re.search(r'/tipster/\d+', el.get('href', '')):
                links.append(el)
        # If multiple race h4s share the same race_num (top + bottom
        # render), prefer the one with more links.
        prev = race_to_links.get(race_num, [])
        if len(links) > len(prev):
            race_to_links[race_num] = links

    # ------------------------------------------------------------------
    # Pass 4: extract picks per tipster per race. For each race's set of
    # tipster links, walk forward in document order and read the
    # following text up to the next tipster link (or the next race h4).
    # We collect text from sibling/next elements until we hit the next
    # tipster link.
    # ------------------------------------------------------------------
    pick_triple_re = re.compile(
        r'(\d{1,2})\s*\n+([^\n]+?)\s*\n+(\d+(?:/\d+)?)(?=\s*\n|\s*$)'
    )

    # Build a global "stop set" of all tipster link elements + race h4
    # elements so we know where to stop walking forward.
    all_tipster_links = set()
    for links in race_to_links.values():
        for l in links:
            all_tipster_links.add(id(l))
    race_h4_set = {id(h4) for _, h4 in race_h4_positions}

    for race_num, links in race_to_links.items():
        for idx, link in enumerate(links):
            href = link.get('href', '')
            tm = re.search(r'/tipster/(\d+)', href)
            if not tm:
                continue
            tid = tm.group(1)

            # Pull display + points from link text
            link_text = link.get_text('\n', strip=True)
            lines = [l.strip() for l in link_text.split('\n') if l.strip()]
            if not lines:
                continue
            display = None
            points = 0
            for ln in lines:
                pm = re.match(r'^([\d,\.]+)\s*(k?)\s*points$', ln, re.IGNORECASE)
                if pm:
                    raw = pm.group(1).replace(',', '')
                    try:
                        val = float(raw)
                        if pm.group(2).lower() == 'k':
                            val *= 1000
                        points = int(val)
                    except ValueError:
                        pass
                elif display is None and not re.match(r'^\d+$', ln):
                    display = ln
            if not display:
                display = lines[0]

            # Walk forward from this link, collecting text, until we
            # hit either (a) the next tipster link or (b) the next
            # race h4. We accumulate text from all elements we pass.
            collected = []
            for el in link.find_all_next():
                if id(el) in all_tipster_links and el is not link:
                    break
                if id(el) in race_h4_set:
                    break
                # Only record terminal text — using .string avoids
                # double-counting parent + child text. For elements
                # without a direct string, skip and let children handle.
                if el.string is not None:
                    s = el.string.strip()
                    if s:
                        collected.append(s)

            block_text = '\n'.join(collected)

            picks = []
            for pm in pick_triple_re.finditer(block_text):
                horse_num = int(pm.group(1))
                horse_name = pm.group(2).strip()
                odds_str = pm.group(3).strip()
                if not re.search(r'[A-Za-z]', horse_name):
                    continue
                if any(bad in horse_name.lower() for bad in
                       ('points', 'top score', 'crowd', 'race ',
                        'racing', 'tipsters', 'most popular',
                        'tipster', 'leaderboard')):
                    continue
                picks.append((horse_num, horse_name, odds_str))
                if len(picks) >= 3:
                    break

            if not picks:
                continue

            handle = _slugify_handle(display)
            entry = result['tipsters'].setdefault(tid, {
                'handle': handle,
                'display': display,
                'points': points,
                'picks': {},
            })
            if points > entry['points']:
                entry['points'] = points
            existing = entry['picks'].get(race_num, [])
            if len(picks) > len(existing):
                entry['picks'][race_num] = picks

    return result


# =============================================================================
# PARSING — tipster profile page (odds-adjusted)
# =============================================================================

def fetch_tipster_profile(tipster_id):
    """
    Fetch one tipster's profile page and pull odds_adj + total_tips.

    Returns dict: {
      'id': '740',
      'handle': 'ballywho',
      'display': 'Ballywho',
      'odds_adj': 0.56,    # None if not available
      'total_tips': 48091, # 0 if not parseable
      'strike_rate': 0.50, # 0.0 if not parseable
    }
    """
    url = f"{BASE_URL}/tipster/{tipster_id}"
    try:
        resp = _fetch(url)
        if resp.status_code != 200:
            return None
    except Exception as e:
        print(f"    fetch failed for tipster {tipster_id}: {e}")
        return None

    soup = BeautifulSoup(resp.content, 'html.parser')
    text = soup.get_text('\n', strip=True)

    # Display name: first <h1>
    h1 = soup.find('h1')
    display = h1.get_text(strip=True) if h1 else f"tipster_{tipster_id}"

    # Odds-adjusted: appears as "Odds Adjusted\n+0.56" or "-0.12"
    oa = None
    oa_match = re.search(r'Odds Adjusted\s*\n\s*([+\-]?\d+(?:\.\d+)?)', text)
    if oa_match:
        try:
            oa = float(oa_match.group(1))
        except ValueError:
            pass

    # Total tips: "Total Tips\n48,091"
    total_tips = 0
    tt_match = re.search(r'Total Tips\s*\n\s*([\d,]+)', text)
    if tt_match:
        try:
            total_tips = int(tt_match.group(1).replace(',', ''))
        except ValueError:
            pass

    # Strike rate: "Strike Rate\n50%"
    sr = 0.0
    sr_match = re.search(r'Strike Rate\s*\n\s*(\d+)%', text)
    if sr_match:
        try:
            sr = int(sr_match.group(1)) / 100.0
        except ValueError:
            pass

    return {
        'id': tipster_id,
        'handle': _slugify_handle(display),
        'display': display,
        'odds_adj': oa,
        'total_tips': total_tips,
        'strike_rate': sr,
    }


# =============================================================================
# RATING SEEDS
# =============================================================================

def odds_adj_to_seed(odds_adj, total_tips, shrinkage=0.5):
    """
    Map a tipster's odds-adjusted score → initial Elo rating.

    The cap is 1700 (per design — reserves higher slots like 1750
    for known elite handicappers like Kyle Herron). Anything below
    MIN_TIPS_FOR_SEED tips gets neutral 1500 regardless of their
    odds_adj — small samples are noisy.

    Shrinkage (0.0 - 1.0) pulls the seed toward 1500 to account for
    leakage between tipmeerkat's recent-form odds-adj score and the
    cards we're about to replay. shrinkage=1.0 means "use the raw
    mapping"; shrinkage=0.5 means "halve the distance from 1500".
    Default 0.5 is moderately aggressive — it preserves rank order
    but compresses the spread (1700 → 1600, 1450 → 1475).
    """
    if odds_adj is None:
        return 1500
    if total_tips < MIN_TIPS_FOR_SEED:
        return 1500

    # Piecewise linear mapping from odds_adj → "raw" seed
    if odds_adj >= 1.0:
        raw = 1700
    elif odds_adj >= 0.7:
        # 0.7 → 1625, 1.0 → 1700  (slope 250 per unit oa)
        raw = 1625 + (odds_adj - 0.7) * 250
    elif odds_adj >= 0.4:
        # 0.4 → 1550, 0.7 → 1625
        raw = 1550 + (odds_adj - 0.4) * 250
    elif odds_adj >= 0.1:
        # 0.1 → 1500, 0.4 → 1550
        raw = 1500 + (odds_adj - 0.1) * 167
    elif odds_adj >= 0.0:
        raw = 1500
    elif odds_adj >= -0.3:
        # 0.0 → 1500, -0.3 → 1450
        raw = 1500 + odds_adj * 167
    else:
        raw = 1450

    # Apply shrinkage toward 1500
    shrunk = 1500 + (raw - 1500) * shrinkage
    return int(round(shrunk))


def load_tipsters():
    """Load saved tipster registry (id → metadata)."""
    if TIPSTERS_PATH.exists():
        with open(TIPSTERS_PATH) as f:
            return json.load(f)
    return {}


def save_tipsters(tipsters):
    TIPSTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIPSTERS_PATH, 'w') as f:
        json.dump(tipsters, f, indent=2, sort_keys=True)


def refresh_tipster(tipster_id, registry, force=False, sleep=0.5, shrinkage=0.5):
    """
    Fetch + cache one tipster's profile. Returns the registry entry.

    If `force` is False and we already have an entry, no fetch happens.
    """
    if not force and tipster_id in registry:
        return registry[tipster_id]

    profile = fetch_tipster_profile(tipster_id)
    if not profile:
        return None

    profile['seed_rating'] = odds_adj_to_seed(
        profile['odds_adj'], profile['total_tips'], shrinkage=shrinkage
    )
    profile['seed_shrinkage'] = shrinkage
    profile['last_refreshed'] = datetime.now().isoformat(timespec='seconds')
    registry[tipster_id] = profile

    time.sleep(sleep)  # be polite
    return profile


# =============================================================================
# WRITING — logit-delta CSVs
# =============================================================================

def write_tipster_csv(out_dir, handle, picks_by_race):
    """
    Write a logit-delta CSV for one tipster.
    Format matches what expert_elo.py / score_expert_race expects:
      race,horse_number,logit_delta
    """
    out_path = out_dir / f"tm_{handle}.csv"
    rows = ['race,horse_number,logit_delta']
    for race in sorted(picks_by_race.keys()):
        picks = picks_by_race[race]
        for i, (horse_num, _name, _odds) in enumerate(picks[:len(PICK_DELTAS)]):
            rows.append(f"{race},{horse_num},{PICK_DELTAS[i]:+.2f}")
    out_path.write_text('\n'.join(rows) + '\n')
    return out_path


def write_consensus_csv(out_dir, parsed):
    """
    Write a consensus CSV. For each race we tally how often each horse
    appears in *any* tipster's top-3, weighted by pick position
    (1st pick: 1.5, 2nd: 1.0, 3rd: 0.5), then normalize so the most-
    backed horse gets +1.5 and others scale down proportionally.

    This gives a single 'tm_consensus' expert that captures community
    wisdom across all tipsters who showed up that day.
    """
    # Tally weighted votes per (race, horse_num)
    tally = {}  # (race, horse) -> weighted vote sum
    for tdata in parsed['tipsters'].values():
        for race, picks in tdata['picks'].items():
            for i, (horse_num, _name, _odds) in enumerate(picks[:len(PICK_DELTAS)]):
                key = (race, horse_num)
                tally[key] = tally.get(key, 0.0) + PICK_DELTAS[i]

    # Per-race normalize to peak at +1.5 (max delta in PICK_DELTAS)
    by_race = {}
    for (race, horse), votes in tally.items():
        by_race.setdefault(race, []).append((horse, votes))

    rows = ['race,horse_number,logit_delta']
    peak = max(PICK_DELTAS)
    for race in sorted(by_race):
        race_picks = sorted(by_race[race], key=lambda x: -x[1])
        if not race_picks:
            continue
        max_votes = race_picks[0][1]
        if max_votes <= 0:
            continue
        # Only emit horses that got at least 25% of the top horse's votes
        # — keeps the signal clean and avoids long tails of noise.
        cutoff = max_votes * 0.25
        for horse, votes in race_picks:
            if votes < cutoff:
                continue
            delta = peak * (votes / max_votes)
            rows.append(f"{race},{horse},{delta:+.2f}")

    out_path = out_dir / 'tm_consensus.csv'
    out_path.write_text('\n'.join(rows) + '\n')
    return out_path


def reshrink_seeds(shrinkage):
    """
    Recompute seed_rating for every cached tipster using a new
    shrinkage value, without re-fetching any profile pages.

    Useful when you want to pull all seeds further toward 1500
    (or, conversely, expand them) after seeing how an initial replay
    behaves. Updates tipmeerkat_tipsters.json in place.
    """
    registry = load_tipsters()
    if not registry:
        print("No tipsters cached yet. Nothing to re-shrink.")
        return

    n_changed = 0
    for tid, p in registry.items():
        old_seed = p.get('seed_rating', 1500)
        new_seed = odds_adj_to_seed(
            p.get('odds_adj'), p.get('total_tips', 0), shrinkage=shrinkage
        )
        if new_seed != old_seed:
            n_changed += 1
        p['seed_rating'] = new_seed
        p['seed_shrinkage'] = shrinkage

    save_tipsters(registry)
    print(f"Re-shrunk {len(registry)} tipster seed(s) "
          f"(shrinkage={shrinkage:.2f}, {n_changed} changed)")
    print(f"Updated → {TIPSTERS_PATH}")


def migrate_csv_headers():
    """
    One-shot migration for tm_*.csv files written by an older version
    of this scraper. Two things get fixed:
      1. Column header 'delta' → 'logit_delta' (so expert_elo.py can
         read the file).
      2. Delta values get rescaled from the old 1.5/1.0/0.5 scale to
         the current 0.75/0.5/0.25 scale (matches other scraped
         experts).

    Walks expert_picks/ and rewrites in place. Idempotent — files that
    already have the correct header AND a max delta <= 1.0 are left
    alone. The two fixes are coupled because every old file has both
    issues at once.
    """
    expert_picks_dir = DATA_PATH / 'expert_picks'
    if not expert_picks_dir.exists():
        print(f"No expert_picks dir at {expert_picks_dir}")
        return

    n_checked = 0
    n_header_only = 0
    n_full_migrated = 0
    n_skipped = 0

    for csv_path in expert_picks_dir.rglob('tm_*.csv'):
        n_checked += 1
        with open(csv_path, 'r') as f:
            content = f.read()
        if not content.strip():
            continue

        lines = content.split('\n')
        first_line = lines[0]

        # Decide which migration(s) this file needs
        needs_header = (first_line == 'race,horse_number,delta')
        is_already_new_header = (first_line == 'race,horse_number,logit_delta')

        if not (needs_header or is_already_new_header):
            print(f"  skipped (unexpected header): "
                  f"{csv_path.relative_to(DATA_PATH)}")
            n_skipped += 1
            continue

        # Detect old scale by checking the max absolute delta value.
        # Old scale: max is 1.5; new scale: max is 0.75.
        # Threshold of 1.0 cleanly separates them.
        max_abs_delta = 0.0
        data_rows = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 3:
                continue
            try:
                race = int(parts[0])
                horse = int(parts[1])
                delta = float(parts[2])
            except ValueError:
                continue
            data_rows.append((race, horse, delta))
            if abs(delta) > max_abs_delta:
                max_abs_delta = abs(delta)

        needs_rescale = max_abs_delta > 1.0

        if not needs_header and not needs_rescale:
            continue  # fully up-to-date

        # Rebuild file
        new_rows = ['race,horse_number,logit_delta']
        for race, horse, delta in data_rows:
            if needs_rescale:
                delta = delta * 0.5  # 1.5→0.75, 1.0→0.5, 0.5→0.25
            new_rows.append(f"{race},{horse},{delta:+.2f}")

        with open(csv_path, 'w') as f:
            f.write('\n'.join(new_rows) + '\n')

        rel = csv_path.relative_to(DATA_PATH)
        if needs_header and needs_rescale:
            print(f"  migrated header + rescaled deltas: {rel}")
            n_full_migrated += 1
        elif needs_header:
            print(f"  migrated header only: {rel}")
            n_header_only += 1
        elif needs_rescale:
            print(f"  rescaled deltas only: {rel}")
            n_full_migrated += 1

    print(f"\nChecked {n_checked} tm_*.csv file(s):")
    print(f"  fully migrated (header + rescale): {n_full_migrated}")
    print(f"  header-only fix:                    {n_header_only}")
    print(f"  skipped (unexpected format):        {n_skipped}")


# =============================================================================
# DRIVER
# =============================================================================

def scrape_track(track, target_date=None, refresh_profiles=False, sleep=0.5,
                 shrinkage=0.5):
    """
    Scrape a tipmeerkat meeting and write per-tipster + consensus CSVs.
    """
    slug = TRACK_SLUGS.get(track)
    if not slug:
        print(f"Unknown track: {track}")
        print(f"Known: {sorted(TRACK_SLUGS.keys())}")
        print(f"Add a slug to TRACK_SLUGS in scrape_tipmeerkat.py")
        return

    if target_date is None:
        target_date = date.today()
    date_str = target_date.strftime('%Y-%m-%d')

    # Today's URL has no date suffix; historical URLs do. Try the dated
    # form first if we're not asking for today, else go straight to the
    # un-dated 'today' page.
    if target_date == date.today():
        url = f"{BASE_URL}/horse-racing-tips-picks/{slug}"
    else:
        url = f"{BASE_URL}/horse-racing-tips-picks/{slug}/{date_str}"

    print(f"Scraping tipmeerkat {track} for {date_str}")
    print(f"  URL: {url}")

    try:
        resp = _fetch(url)
    except Exception as e:
        print(f"  fetch error: {e}")
        return

    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} — no meeting page")
        return

    parsed = parse_meeting_page(resp.text, date_str)

    n_tipsters = len(parsed['tipsters'])
    n_picks = sum(len(t['picks']) for t in parsed['tipsters'].values())
    n_races = len(parsed['crowd_favorites']) or len(
        {r for t in parsed['tipsters'].values() for r in t['picks']}
    )

    print(f"  Parsed: {n_races} races, {n_tipsters} tipsters, {n_picks} race-picks")

    if not parsed['tipsters']:
        print(f"  No tipster picks parsed — check page layout.")
        return

    # Refresh tipster profiles for any new IDs we haven't seen, so we
    # have odds-adj seeds available for expert_elo.py.
    registry = load_tipsters()
    new_ids = [tid for tid in parsed['tipsters'] if tid not in registry]
    if new_ids or refresh_profiles:
        targets = list(parsed['tipsters']) if refresh_profiles else new_ids
        print(f"  Fetching profiles for {len(targets)} tipster(s) "
              f"({'all' if refresh_profiles else 'new only'})...")
        for tid in targets:
            profile = refresh_tipster(tid, registry, force=refresh_profiles,
                                      sleep=sleep, shrinkage=shrinkage)
            if profile:
                print(f"    {profile['handle']:<25s} "
                      f"odds_adj={profile['odds_adj']!s:>6}  "
                      f"tips={profile['total_tips']:>6,}  "
                      f"→ seed {profile['seed_rating']}")
        save_tipsters(registry)

    # Write CSVs
    out_dir = DATA_PATH / 'expert_picks' / date_str / track
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for tid, tdata in parsed['tipsters'].items():
        # Use the registry handle if present (it's based on the profile's
        # canonical display name, which can differ from the meeting page
        # in rare cases).
        handle = registry.get(tid, {}).get('handle') or tdata['handle']
        path = write_tipster_csv(out_dir, handle, tdata['picks'])
        written.append(path)

    # consensus_path = write_consensus_csv(out_dir, parsed)
    # written.append(consensus_path)

    # print(f"  Wrote {len(written)} CSV(s) → {out_dir}/")
    # print(f"    (incl. tm_consensus.csv aggregating all tipsters)")


def list_tipsters():
    """Print the cached tipster registry sorted by seed rating."""
    registry = load_tipsters()
    if not registry:
        print(f"No tipsters cached yet. Run a scrape first.")
        return
    print(f"\n{'='*80}")
    print(f"TIPMEERKAT TIPSTERS  ({len(registry)} cached)")
    print(f"{'='*80}")
    print(f"  {'Handle':<25s} {'Seed':>5s} {'OddsAdj':>8s} {'Tips':>7s} {'Strike':>7s}")
    print(f"  {'-'*25} {'-'*5} {'-'*8} {'-'*7} {'-'*7}")
    for tid, p in sorted(registry.items(), key=lambda x: -x[1].get('seed_rating', 0)):
        oa = p.get('odds_adj')
        oa_str = f"{oa:+.2f}" if oa is not None else "  —  "
        sr = p.get('strike_rate', 0)
        sr_str = f"{sr:.0%}" if sr else "  —"
        print(f"  {p['handle']:<25s} {p.get('seed_rating', 1500):>5d} "
              f"{oa_str:>8s} {p.get('total_tips', 0):>7,d} {sr_str:>7s}")


def export_seeds_for_elo():
    """
    Print a Python snippet of EXPERT_INITIAL_RATINGS additions you can
    paste into expert_elo.py. Only includes tipsters with non-default
    seeds (i.e. those that earned a non-1500 rating).
    """
    registry = load_tipsters()
    seeded = {f"tm_{p['handle']}": p['seed_rating']
              for p in registry.values()
              if p.get('seed_rating', 1500) != 1500}
    seeded['tm_consensus'] = 1525  # default seed for the crowd

    if not seeded:
        print("# No non-default seeds yet. Run scrape first.")
        return

    print("# Add to EXPERT_INITIAL_RATINGS in expert_elo.py:")
    print("EXPERT_INITIAL_RATINGS.update({")
    for k, v in sorted(seeded.items(), key=lambda x: -x[1]):
        print(f"    {k!r:<32s}: {v},")
    print("})")


def show_leaderboard(filter_source='all', min_races=0, top=None, sort_by='rating'):
    """
    Render a unified leaderboard from expert_ratings.json.

    Combines current Elo ratings with tipmeerkat metadata (odds_adj,
    total_tips) where available so you can see how the tm_* crowd
    ranks alongside your scraped track experts.

    Args:
      filter_source: 'all', 'tm', or 'scraped'
                     'tm'      → only tm_* experts
                     'scraped' → only non-tm_* experts
      min_races: hide experts with fewer than this many scored races
      top: show only the top N rows after sorting (None = all)
      sort_by: 'rating' (default), 'imp_per_race', or 'delta'
               imp_per_race surfaces best per-race performers regardless
               of seed; delta shows biggest movers from initial rating
    """
    ratings_path = DATA_PATH / 'expert_ratings.json'
    if not ratings_path.exists():
        print(f"No ratings file at {ratings_path}")
        print(f"Run: python expert_elo.py --force")
        return

    with open(ratings_path) as f:
        ratings = json.load(f)

    if not ratings:
        print("Ratings file is empty.")
        return

    # Pull tipmeerkat metadata for cross-referencing tm_* experts
    tm_registry = load_tipsters()
    # Index by handle (without tm_ prefix) for fast lookup
    tm_by_handle = {p['handle']: p for p in tm_registry.values()}

    # Build rows
    rows = []
    for name, data in ratings.items():
        is_tm = name.startswith('tm_')

        # Apply source filter
        if filter_source == 'tm' and not is_tm:
            continue
        if filter_source == 'scraped' and is_tm:
            continue

        n = data.get('races_scored', 0)
        if n < min_races:
            continue

        rating = data.get('rating', 1500)
        initial = data.get('initial_rating', 1500)
        improvement = data.get('total_improvement', 0.0)
        imp_per_race = improvement / n if n > 0 else 0.0
        delta_from_seed = rating - initial

        # tm-specific metadata
        odds_adj = None
        total_tips = None
        if is_tm:
            handle = name[3:]  # strip 'tm_'
            if handle == 'consensus':
                odds_adj = None  # no profile for the synthetic consensus
            elif handle in tm_by_handle:
                p = tm_by_handle[handle]
                odds_adj = p.get('odds_adj')
                total_tips = p.get('total_tips')

        rows.append({
            'name': name,
            'is_tm': is_tm,
            'rating': rating,
            'initial': initial,
            'delta': delta_from_seed,
            'races': n,
            'improvement': improvement,
            'imp_per_race': imp_per_race,
            'odds_adj': odds_adj,
            'total_tips': total_tips,
        })

    if not rows:
        crit = []
        if filter_source != 'all':
            crit.append(f"source={filter_source}")
        if min_races > 0:
            crit.append(f"min_races={min_races}")
        print(f"No experts match filter ({', '.join(crit) or 'none'}).")
        return

    # Sort
    sort_keys = {
        'rating':       lambda r: -r['rating'],
        'imp_per_race': lambda r: -r['imp_per_race'],
        'delta':        lambda r: -r['delta'],
    }
    if sort_by not in sort_keys:
        print(f"Unknown sort_by: {sort_by!r}. Using 'rating'.")
        sort_by = 'rating'
    rows.sort(key=sort_keys[sort_by])
    if top:
        rows = rows[:top]

    # Header
    title = "EXPERT LEADERBOARD"
    if filter_source == 'tm':
        title += "  (tipmeerkat only)"
    elif filter_source == 'scraped':
        title += "  (scraped experts only)"
    if sort_by != 'rating':
        title += f"  — sorted by {sort_by}"

    print(f"\n{'='*92}")
    print(f"{title}")
    print(f"{'='*92}")
    print(f"  {'Expert':<28s} {'Rating':>7s} {'Init':>5s} {'Δ':>6s} "
          f"{'Races':>5s} {'Imp/R':>7s} {'OddsAdj':>8s} {'TM Tips':>8s}")
    print(f"  {'-'*28} {'-'*7} {'-'*5} {'-'*6} {'-'*5} {'-'*7} {'-'*8} {'-'*8}")

    for r in rows:
        rank_marker = 'Ⓜ' if r['is_tm'] else ' '  # discreet TM marker
        delta_str = f"{r['delta']:+.0f}" if r['races'] > 0 else "  —"
        races_str = '—' if r['races'] == 0 else str(r['races'])
        impr_str = f"{r['imp_per_race']:+.3f}" if r['races'] > 0 else "    —"
        oa_str = f"{r['odds_adj']:+.2f}" if r['odds_adj'] is not None else "    —"
        tips_str = f"{r['total_tips']:,}" if r['total_tips'] else "       —"

        print(f"  {rank_marker}{r['name']:<27s} {r['rating']:>7.1f} "
              f"{r['initial']:>5.0f} {delta_str:>6s} {races_str:>5s} "
              f"{impr_str:>7s} {oa_str:>8s} {tips_str:>8s}")

    # Summary footer
    n_tm = sum(1 for r in rows if r['is_tm'])
    n_scraped = sum(1 for r in rows if not r['is_tm'])
    n_with_history = sum(1 for r in rows if r['races'] > 0)
    above = sum(1 for r in rows if r['rating'] > 1500 and r['races'] > 0)

    print(f"\n  Showing {len(rows)} expert(s): {n_tm} tipmeerkat (Ⓜ), "
          f"{n_scraped} scraped")
    print(f"  {n_with_history} have replay history; {above} are currently above 1500")
    print(f"  Δ = current rating - seed rating  |  Imp/R = log-loss gain per race")


def diagnose_expert(expert_name, max_races=None, only_track=None):
    """
    Walk through one expert's entire pick history and show, race by
    race, what they picked, what won, what the ML probabilities were,
    and how each race contributed to their rating.

    This imports expert_elo so the scoring is exactly what the replay
    uses — no risk of the diagnostic and the replay disagreeing.

    Args:
      expert_name: e.g. 'tm_great_scott' or 'scott_hazelton'
      max_races: cap output to first N races (None = all)
      only_track: filter to one track (e.g. 'churchill-downs')
    """
    if not _HAS_PANDAS:
        print("--diagnose requires pandas + numpy. Install with:")
        print("  pip install pandas numpy")
        return
    # Import expert_elo from the same directory. We do this lazily so
    # `--diagnose` works even if expert_elo isn't importable for some
    # reason (we'll surface the import error directly).
    import importlib.util
    elo_path = Path(__file__).parent / 'expert_elo.py'
    if not elo_path.exists():
        print(f"Can't find expert_elo.py at {elo_path}")
        return
    spec = importlib.util.spec_from_file_location('expert_elo', elo_path)
    elo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(elo)

    # Find every CSV for this expert across all dates/tracks
    expert_picks_dir = DATA_PATH / 'expert_picks'
    if not expert_picks_dir.exists():
        print(f"No expert_picks dir at {expert_picks_dir}")
        return

    # Walk dates chronologically (matches replay order)
    pick_files = []  # list of (date_str, track, csv_path)
    for date_dir in sorted(expert_picks_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        for track_dir in sorted(date_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            track = track_dir.name
            if only_track and track != only_track:
                continue
            csv_path = track_dir / f"{expert_name}.csv"
            if csv_path.exists():
                pick_files.append((date_str, track, csv_path))

    if not pick_files:
        print(f"No pick files found for {expert_name}")
        if only_track:
            print(f"  (filtered to track={only_track})")
        return

    # Load all entries once
    entries_df = elo.load_all_entries()
    if entries_df.empty or 'won' not in entries_df.columns:
        print("No entries with results available — can't diagnose.")
        return

    # Pull seed for this expert
    seed = elo.EXPERT_INITIAL_RATINGS.get(expert_name, elo.INITIAL_RATING)

    # If it's a tm_ expert, also pull from the tipmeerkat registry so
    # we can show the user what odds-adj seeded them at
    tm_meta = None
    if expert_name.startswith('tm_'):
        registry = load_tipsters()
        handle = expert_name[3:]
        for p in registry.values():
            if p.get('handle') == handle:
                tm_meta = p
                seed = p.get('seed_rating', seed)
                break

    print(f"\n{'='*92}")
    print(f"DIAGNOSING: {expert_name}")
    if tm_meta:
        print(f"  tm profile: odds_adj={tm_meta.get('odds_adj')}, "
              f"total_tips={tm_meta.get('total_tips'):,}, "
              f"strike_rate={tm_meta.get('strike_rate', 0):.0%}")
    print(f"  seed rating: {seed}")
    print(f"  found {len(pick_files)} day(s) with picks")
    print(f"{'='*92}")

    # Replay this expert through their history
    rating = float(seed)
    n_scored = 0
    cumulative_imp = 0.0
    races_shown = 0

    # Per-race breakdown
    print(f"\n  {'Date':<12} {'Track':<18} {'R':>3s} {'Pick→':<12s} "
          f"{'Won':<4s} {'P(ml)':>6s} {'P(exp)':>7s} "
          f"{'ml_loss':>7s} {'exp_loss':>8s} {'imp':>7s}")
    print(f"  {'-'*12} {'-'*18} {'-'*3} {'-'*12} "
          f"{'-'*4} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")

    day_summaries = []  # list of (date, track, n_races, day_imp, rating_after)

    for date_str, track, csv_path in pick_files:
        expert_df = pd.read_csv(csv_path)
        if expert_df.empty:
            continue

        day_entries = entries_df[
            (entries_df['date'] == date_str) & (entries_df['track'] == track)
        ]
        if day_entries.empty:
            continue

        race_results = []
        n_before = n_scored
        rating_before = rating

        for race_num in sorted(expert_df['race'].unique()):
            race_df = day_entries[day_entries['race'] == race_num].copy()
            if race_df.empty:
                continue
            expert_race = expert_df[expert_df['race'] == race_num]
            result = elo.score_expert_race(race_df, expert_race)
            if result is None:
                continue
            ml_loss, expert_loss, _n = result
            improvement = ml_loss - expert_loss
            race_results.append((ml_loss, expert_loss))

            if max_races is None or races_shown < max_races:
                # Identify pick details for display
                picks_sorted = expert_race.sort_values(
                    'logit_delta', ascending=False
                )
                top_pick_row = picks_sorted.iloc[0]
                top_pick_horse = int(top_pick_row['horse_number'])
                top_pick_delta = float(top_pick_row['logit_delta'])

                # Find the winner
                winners = race_df[race_df['won'] == 1]
                if len(winners) == 0:
                    continue
                winner_horse = int(winners.iloc[0]['horse_number'])

                # Was top pick the winner?
                hit = "✓" if top_pick_horse == winner_horse else " "

                # Compute per-horse probs to show top_pick's probability
                # under both ML and expert distributions
                ml_raw = race_df['ml_prob'].values
                if np.any(np.isnan(ml_raw)):
                    continue
                ml_norm = ml_raw / ml_raw.sum()
                delta_map = {int(r['horse_number']): r['logit_delta']
                             for _, r in expert_race.iterrows()}
                deltas = race_df['horse_number'].map(
                    lambda h: delta_map.get(int(h), 0.0))
                expert_probs = elo.apply_logit_deltas(
                    pd.Series(ml_norm, index=race_df.index), deltas)

                # Probability the expert assigned to the actual winner
                winner_idx = winners.index[0]
                p_ml_winner = ml_norm[race_df.index.get_loc(winner_idx)]
                p_exp_winner = float(expert_probs.loc[winner_idx])

                pick_str = f"#{top_pick_horse}({top_pick_delta:+.2f})"
                won_str = f"#{winner_horse}{hit}"
                print(f"  {date_str:<12} {track:<18} {race_num:>3d} "
                      f"{pick_str:<12s} {won_str:<4s} "
                      f"{p_ml_winner:>6.3f} {p_exp_winner:>7.3f} "
                      f"{ml_loss:>7.3f} {expert_loss:>8.3f} "
                      f"{improvement:>+7.3f}")
                races_shown += 1

        if not race_results:
            continue

        # Aggregate the day, just like run_full_replay does
        ml_total = sum(r[0] for r in race_results)
        expert_total = sum(r[1] for r in race_results)
        day_imp = ml_total - expert_total

        # Mirror the Elo update from expert_elo.run_full_replay
        ELO_SCALE = 3000
        k = elo.get_k_factor(n_before)
        expected = len(race_results) * (rating_before - elo.INITIAL_RATING) / ELO_SCALE
        surprise = day_imp - expected
        rating = rating_before + k * surprise

        n_scored = n_before + len(race_results)
        cumulative_imp += day_imp
        day_summaries.append(
            (date_str, track, len(race_results), day_imp,
             rating_before, rating, k, expected, surprise)
        )

        if max_races is not None and races_shown >= max_races:
            print(f"  ... (truncated at {max_races} races; "
                  f"use --max-races N or omit to see all)")
            break

    # Day-by-day rating walk
    print(f"\n  {'='*92}")
    print(f"  DAY-BY-DAY RATING WALK")
    print(f"  {'='*92}")
    print(f"  {'Date':<12} {'Track':<18} {'Races':>5s} {'DayImp':>8s} "
          f"{'Expected':>9s} {'Surprise':>9s} {'K':>5s} "
          f"{'Before':>7s} → {'After':>7s}")
    print(f"  {'-'*12} {'-'*18} {'-'*5} {'-'*8} {'-'*9} {'-'*9} "
          f"{'-'*5} {'-'*7}   {'-'*7}")
    for (date_str, track, n_races, day_imp, rb, ra, k,
         expected, surprise) in day_summaries:
        arrow = '▲' if ra > rb else ('▼' if ra < rb else '·')
        print(f"  {date_str:<12} {track:<18} {n_races:>5d} "
              f"{day_imp:>+8.3f} {expected:>+9.3f} {surprise:>+9.3f} "
              f"{k:>5.1f} {rb:>7.1f} → {ra:>7.1f} {arrow}")

    # Summary
    if n_scored > 0:
        avg_imp = cumulative_imp / n_scored
        print(f"\n  Final: {n_scored} races scored, "
              f"total improvement {cumulative_imp:+.3f} "
              f"(avg {avg_imp:+.4f}/race)")
        print(f"  Rating walk: {seed} → {rating:.1f} "
              f"(net {rating - seed:+.1f})")
        # Sanity check vs. saved rating
        ratings_path = DATA_PATH / 'expert_ratings.json'
        if ratings_path.exists():
            with open(ratings_path) as f:
                saved = json.load(f)
            if expert_name in saved:
                saved_rating = saved[expert_name].get('rating')
                if saved_rating is not None:
                    diff = abs(saved_rating - rating)
                    flag = "✓ matches saved" if diff < 0.5 else f"⚠ DIFFERS from saved ({saved_rating})"
                    print(f"  Saved rating in expert_ratings.json: "
                          f"{saved_rating}  {flag}")


# =============================================================================
# MAIN
# =============================================================================

def parse_date_arg(date_str):
    for fmt in ['%Y-%m-%d', '%m-%d-%Y', '%m-%d-%y']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Can't parse date: {date_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape tipmeerkat community picks → expert_picks CSVs'
    )
    parser.add_argument('--track', type=str, default=None,
                        help=f'Track name (e.g. churchill-downs)')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--refresh-tipsters', action='store_true',
                        help='Re-fetch profile odds-adj for all tipsters in this meeting')
    parser.add_argument('--list-tipsters', action='store_true',
                        help='Print cached tipster registry sorted by seed')
    parser.add_argument('--export-seeds', action='store_true',
                        help='Print EXPERT_INITIAL_RATINGS snippet for expert_elo.py')
    parser.add_argument('--sleep', type=float, default=0.5,
                        help='Seconds between profile fetches (default 0.5)')
    parser.add_argument('--shrinkage', type=float, default=0.5,
                        help='Pull seeds toward 1500 by this factor (0=full pull, '
                             '1=no shrinkage). Default 0.5. Used both when '
                             'computing new seeds and with --reshrink-seeds.')
    parser.add_argument('--reshrink-seeds', action='store_true',
                        help='Recompute seed_rating for all cached tipsters using '
                             '--shrinkage value, without re-fetching profiles')
    parser.add_argument('--migrate-csvs', action='store_true',
                        help='One-shot fix for tm_*.csv files written by an '
                             'older version: rewrites "delta" header to '
                             '"logit_delta" and rescales delta values from '
                             '1.5/1.0/0.5 to 0.75/0.5/0.25')
    parser.add_argument('--leaderboard', action='store_true',
                        help='Show ranked leaderboard from expert_ratings.json')
    parser.add_argument('--source', type=str, default='all',
                        choices=['all', 'tm', 'scraped'],
                        help='Filter --leaderboard by expert source '
                             '(default: all)')
    parser.add_argument('--min-races', type=int, default=0,
                        help='Hide leaderboard entries with fewer than this '
                             'many scored races (default 0)')
    parser.add_argument('--top', type=int, default=None,
                        help='Show only top N leaderboard rows (default: all)')
    parser.add_argument('--sort-by', type=str, default='rating',
                        choices=['rating', 'imp_per_race', 'delta'],
                        help='Leaderboard sort: rating (default), '
                             'imp_per_race (best per-race performers), '
                             'or delta (biggest movers from seed)')
    parser.add_argument('--diagnose', type=str, default=None, metavar='EXPERT',
                        help='Walk through one expert\'s pick history race-by-race '
                             'showing pick, winner, ML/expert probabilities, '
                             'log-loss, and rating walk. Helpful for debugging '
                             'why a tipster is rated higher/lower than expected. '
                             'Example: --diagnose tm_great_scott')
    parser.add_argument('--max-races', type=int, default=None,
                        help='With --diagnose, cap per-race detail rows '
                             '(default: show all)')
    parser.add_argument('--only-track', type=str, default=None,
                        help='With --diagnose or --leaderboard, restrict to '
                             'one track (e.g. churchill-downs)')
    args = parser.parse_args()

    if args.list_tipsters:
        list_tipsters()
        return
    if args.export_seeds:
        export_seeds_for_elo()
        return
    if args.reshrink_seeds:
        reshrink_seeds(args.shrinkage)
        return
    if args.migrate_csvs:
        migrate_csv_headers()
        return
    if args.leaderboard:
        show_leaderboard(filter_source=args.source,
                         min_races=args.min_races,
                         top=args.top,
                         sort_by=args.sort_by)
        return
    if args.diagnose:
        diagnose_expert(args.diagnose,
                        max_races=args.max_races,
                        only_track=args.only_track)
        return

    if not args.track:
        parser.error('--track is required (or use --list-tipsters / '
                     '--export-seeds / --reshrink-seeds / --migrate-csvs '
                     '/ --leaderboard / --diagnose)')

    target = parse_date_arg(args.date) if args.date else None
    scrape_track(args.track, target,
                 refresh_profiles=args.refresh_tipsters,
                 sleep=args.sleep,
                 shrinkage=args.shrinkage)


if __name__ == '__main__':
    main()