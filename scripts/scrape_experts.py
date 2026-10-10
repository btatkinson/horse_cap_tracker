"""
scrape_experts.py

Fetch raw expert pick text from track websites and save to disk.
No LLM processing — just grab the text for each expert.

Usage:
    python scrape_experts.py --track keeneland
    python scrape_experts.py --track keeneland --date 2026-04-08

Writes to:
    {DATA_PATH}/raw_expert_picks/{date}/{track}/{expert_name}.txt

Each .txt file is the raw text content from that expert's page for
that date. Different experts have wildly different formats — the
downstream send_experts.py script handles parsing via LLM.
"""

import os, re, argparse
from pathlib import Path
from datetime import datetime, date

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = Path(os.environ['DATA_PATH'])

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("pip install requests beautifulsoup4")


# =============================================================================
# TRACK CONFIGS
# =============================================================================
# Each track has a list of experts with an ID and a scraper function.
# Add new tracks by adding a config and a scraper function.

TRACK_CONFIGS = {
    'keeneland': {
        'index_url': 'https://www.keeneland.com/racing/expert-picks',
        'experts': [
            {'id': '236', 'name': 'scott_hazelton'},
            {'id': '100', 'name': 'jeremy_plonk'},
            {'id': '644', 'name': 'roberto_rodriguez'},
            {'id': '234', 'name': 'ed_derosa'},
            {'id': '33',  'name': 'tom_leach'},
            {'id': '31',  'name': 'ellis_starr'},
            {'id': '858', 'name': 'eric_solomon'},
            {'id': '92',  'name': 'kim_nelson'},
            {'id': '862', 'name': 'kyle_hampton'},
        ],
        'scraper': 'keeneland',
    },
    'churchill-downs': {
        # Single page contains all experts. We don't enumerate experts
        # by ID — the scraper discovers them from h4 headers in the page.
        # 'known_experts' is informational only: a warning is printed
        # when the page contains experts not in this list (someone new
        # joined the staff) or when an expected expert is missing
        # (someone's section didn't render today, e.g. Kaitlin Benson
        # was on leave at time of this writing).
        'index_url': 'https://www.churchilldowns.com/wager/expert-picks/',
        # Secondary source: At The Races publishes consolidated tips
        # (single voice, not per-expert) for Churchill on a separate
        # page. Saved alongside the Churchill house experts as a
        # virtual expert named 'at_the_races'.
        'atr_url': 'https://www.attheraces.com/tips/race-by-race-guides/Churchill-Downs',
        # Tertiary source: Fan Odds publishes numeric-only picks
        # (just program numbers, no analysis prose). Today-only —
        # no historical URL is exposed, so this is skipped during
        # --backfill-atr runs alongside the Churchill house picks.
        # Saved as virtual expert 'fan_odds'.
        'fanodds_url': 'https://www.fanodds.com/us/horse-racing/picks/churchill-downs-usa',
        'known_experts': ['joe_kristufek', 'kevin_kilroy', 'tony_calo',
                          'kaitlin_benson', 'at_the_races', 'fan_odds'],
        'scraper': 'churchill-downs',
    },
}


# =============================================================================
# SCRAPERS
# =============================================================================

def _fetch(url, timeout=15):
    """GET with a browser-ish user agent."""
    return requests.get(url, timeout=timeout, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })


def scrape_keeneland(expert_id, target_date=None):
    """
    Scrape one Keeneland expert page.
    Page=0 is always the current/latest picks.
    We don't validate the date — the caller knows what date they're scraping for.

    Returns (raw_text, target_date) or (None, None) if no content found.
    """
    url = f"https://www.keeneland.com/racing/expert-picks/{expert_id}"
    try:
        resp = _fetch(url)
        if resp.status_code != 200:
            print(f"(HTTP {resp.status_code})", end=' ')
            return None, None

        text = BeautifulSoup(resp.content, 'html.parser').get_text()

        # Check for "no picks available" — only if there's no race content
        if 'no picks available' in text.lower() and 'Race' not in text:
            return None, None

        # Find where the picks content starts
        content = text

        # Try to find a date header and start from there
        date_match = re.search(
            r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+'
            r'(?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2},\s+\d{4}',
            content
        )
        if date_match:
            content = content[date_match.start():]

        # Otherwise try to find start from first race mention
        if not date_match:
            race_match = re.search(
                r'(?:1st\s+Race|Race\s+1\b|Day\s+\d+\s+Thoughts)', content
            )
            if race_match:
                content = content[race_match.start():]

        # Cut footer
        for marker in ['Keeneland Home', 'Privacy Policy', '©',
                       'Twitter Icon', 'Sales\nRacing',
                       'Follow Me on X', 'Follow Me on Twitter',
                       'Follow Me on BlueSky']:
            if marker in content:
                content = content[:content.index(marker)]

        # Clean up whitespace: collapse runs of blank lines to max 2
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Strip leading/trailing whitespace per line
        content = '\n'.join(line.strip() for line in content.split('\n'))
        # Remove lines that are just nav boilerplate
        skip_patterns = [
            r'^Skip to main content$', r'^Main Navigation Menu$',
            r'^Buy Tickets$', r'^Back$', r'^Expert Picks$',
            r'^Home$', r'^Racing$', r'^Next page$',
        ]
        lines = content.split('\n')
        lines = [l for l in lines if not any(re.match(p, l.strip()) for p in skip_patterns)]
        content = '\n'.join(lines)
        # Final collapse of blank lines again after removing boilerplate
        content = re.sub(r'\n{3,}', '\n\n', content).strip()

        if len(content) < 50:
            return None, None

        return content, target_date or date.today()

    except Exception as e:
        print(f"(error: {e})", end=' ')
        return None, None


def scrape_churchill_all(target_date=None):
    """
    Scrape all Churchill Downs experts in one fetch.

    Churchill structure differs from Keeneland in important ways:
    - All experts live on a single page (/wager/expert-picks/), not
      one URL per expert
    - The page has no machine-readable date — it shows whatever the
      latest picks are. We trust the caller's target_date.
    - Each expert's section is delimited by an `#### {Name}` header
      (rendered as <h4>), followed by Race blocks
    - The page renders expert sections TWICE (once near the top, once
      lower for SEO/layout reasons). We dedupe by taking the first
      occurrence of each expert name.

    Returns: dict mapping expert_name (snake_case) -> raw_text
             plus a special key '_target_date' so the caller knows
             which date to stamp the files with.

    Returns ({}, target_date) if the page doesn't have parseable
    expert content (e.g. dark day, off-meet).
    """
    url = 'https://www.churchilldowns.com/wager/expert-picks/'
    try:
        resp = _fetch(url)
        if resp.status_code != 200:
            print(f"(HTTP {resp.status_code})", end=' ')
            return {}, target_date

        soup = BeautifulSoup(resp.content, 'html.parser')

        # Each expert is introduced by an <h4>{Name}</h4>. The picks
        # content is everything between that h4 and the next h4 (or
        # end of the picks region). We grab the page text and split
        # on h4 markers using BeautifulSoup, which is more reliable
        # than regex on the raw HTML.

        h4s = soup.find_all('h4')
        if not h4s:
            print("(no h4 headers found — page layout may have changed)", end=' ')
            return {}, target_date

        # Walk h4 tags; for each one whose text looks like a person name
        # (not a sponsor / footer block), collect text until the next h4.
        results = {}
        for i, h4 in enumerate(h4s):
            name_raw = h4.get_text().strip()
            # Filter out non-expert h4s: must look like "First Last",
            # only letters/spaces/apostrophes, 2-4 words.
            if not re.fullmatch(r"[A-Za-z][A-Za-z\s'\-\.]+", name_raw):
                continue
            words = name_raw.split()
            if not (2 <= len(words) <= 4):
                continue
            # Skip if all-caps section headers like "JOE K'S FINAL 2025 FALL STATS"
            if re.search(r'\d', name_raw) or 'STATS' in name_raw.upper():
                continue

            snake = '_'.join(re.sub(r"[^a-zA-Z\s]", '', name_raw).lower().split())
            if not snake or snake in results:
                # First occurrence wins (Churchill renders sections twice)
                continue

            # Collect text between this h4 and the next h4. We can't
            # rely on a section wrapper — Churchill renders everything
            # as flat siblings. We walk forward through the DOM, but
            # only collect text from elements whose text is NOT already
            # contained in a parent we've collected, to avoid the
            # parent/child double-count problem (e.g., <li> containing
            # <p> would otherwise both contribute the inner text).
            chunks = []
            collected_texts = set()
            stop_h2_h3 = {'EXPERT PICKS AND YOU', 'FOLLOW FOR UPDATES AND EXCLUSIVES'}

            for sib in h4.find_all_next():
                if sib.name == 'h4':
                    break
                if sib.name in ('h2', 'h3') and sib.get_text().strip().upper() in stop_h2_h3:
                    break
                if sib.name not in ('h5', 'h6', 'p', 'li', 'span', 'div',
                                    'strong', 'em', 'ul', 'ol'):
                    continue
                txt = sib.get_text(separator=' ', strip=True)
                if not txt:
                    continue
                # Skip if this exact text is already a (sub)string of a
                # collected chunk — that means this is a child of a node
                # we already grabbed.
                if any(txt in prev for prev in collected_texts):
                    continue
                # Conversely, drop previously-collected chunks that are
                # substrings of this new one — this is a parent of nodes
                # we already grabbed; replace them with the broader text.
                collected_texts = {p for p in collected_texts if p not in txt}
                chunks = [c for c in chunks if c not in txt]
                chunks.append(txt)
                collected_texts.add(txt)

            if not chunks:
                continue

            # Heuristic: a real picks block must mention "Race" multiple times
            joined = '\n'.join(chunks)
            if len(re.findall(r'\bRace\s+\d+\b', joined, re.IGNORECASE)) < 2:
                # Probably not a picks section (e.g. bio paragraph only)
                continue

            # Lightweight cleanup
            joined = re.sub(r'\n{3,}', '\n\n', joined).strip()
            if len(joined) < 100:
                continue

            results[snake] = joined

        return results, (target_date or date.today())

    except Exception as e:
        print(f"(error: {e})", end=' ')
        return {}, target_date


def scrape_atr(track_url_name, target_date=None):
    """
    Scrape At The Races consolidated tips for one track.

    ATR publishes a single set of tips per track (not per expert),
    so this returns a single text blob. We treat it as one virtual
    expert named 'at_the_races' at the caller.

    Page quirks worth knowing:
    - On the day of racing, the page has NO date header — just the
      picks. After the day rolls over, a header like "Track - Tuesday
      28th April 2026" appears. So a missing header means fresh,
      not stale.
    - The picks region always contains race-card links of the form
      /racecard/{track}/{D}-{Month}-{YYYY}/{HHMM}. We anchor on
      these links — they're present in both states.
    - We validate freshness using the date in those race-card URLs,
      not the (sometimes-missing) header. If those URL dates don't
      match target_date, the page is stale and we return None.

    Args:
        track_url_name: track segment in the ATR URL, e.g.
                        'Churchill-Downs'. Case-sensitive — must
                        match ATR's URL exactly.
        target_date: date we expect the picks to cover.

    Backfilling past dates: ATR exposes a date-suffixed URL of the
    form .../race-by-race-guides/{track}/{YYYY-MM-DD} that returns
    the same lean tips format for that day. We only use that URL
    when target_date is in the past — for today we keep the bare
    URL (it's the canonical "current tips" surface and the dated
    URL for today's date isn't guaranteed to be populated).

    Returns: (raw_text, target_date) or (None, None) if no content
             or page is stale.
    """
    base = f"https://www.attheraces.com/tips/race-by-race-guides/{track_url_name}"
    if target_date is not None and target_date < date.today():
        url = f"{base}/{target_date.strftime('%Y-%m-%d')}"
    else:
        url = base
    try:
        resp = _fetch(url)
        if resp.status_code != 200:
            print(f"(HTTP {resp.status_code})", end=' ')
            return None, None

        soup = BeautifulSoup(resp.content, 'html.parser')

        # Find the first race-card link for this track. The picks
        # block lives in the same parent container as these links.
        # URL pattern: /racecard/Churchill-Downs/28-April-2026/1745
        race_link_pattern = re.compile(
            rf'/racecard/{re.escape(track_url_name)}/'
            r'(\d{1,2}-[A-Za-z]+-\d{4})/\d{3,4}'
        )

        race_links = []
        for a in soup.find_all('a', href=True):
            m = race_link_pattern.search(a['href'])
            if m:
                race_links.append((a, m.group(1)))

        if not race_links:
            print(f"(no race links found for {track_url_name})", end=' ')
            return None, None

        # Freshness check: every race link should agree on the date,
        # and that date should match target_date. We pick the modal
        # date among the links (defensive against stray cross-track
        # links), and compare.
        date_strs = [d for _, d in race_links]
        page_date_str = max(set(date_strs), key=date_strs.count)
        try:
            page_date = datetime.strptime(page_date_str, "%d-%B-%Y").date()
        except ValueError:
            print(f"(couldn't parse race-link date '{page_date_str}')",
                  end=' ')
            return None, None

        if target_date is not None and page_date != target_date:
            print(f"(stale: race links are for {page_date}, "
                  f"target {target_date})", end=' ')
            return None, None

        # Find the picks container — walk up from the first race link
        # to find the smallest ancestor that contains all race links.
        # In practice ATR wraps each race in its own block but they
        # share a common parent. The simplest robust anchor: start
        # from the parent of the first race link and walk forward
        # collecting siblings/descendants until we hit a stop marker.
        first_link = race_links[0][0]
        # Walk up to find a parent that's a major block element.
        # We want the row/section containing race 1, then we'll
        # collect text starting from there.
        anchor = first_link
        for _ in range(6):  # don't walk too far
            if anchor.parent is None:
                break
            if anchor.parent.name in ('section', 'article', 'main', 'body'):
                break
            anchor = anchor.parent

        chunks = []
        collected_texts = set()
        stop_phrases = (
            'Whilst studying the form',
            'Also On At The Races',
            'Digital Shows',
        )

        # Collect the anchor itself and then forward siblings/descendants
        nodes_to_scan = [anchor] + list(anchor.find_all_next())
        for sib in nodes_to_scan:
            if not hasattr(sib, 'get_text'):
                continue
            txt = sib.get_text(separator=' ', strip=True)
            if not txt:
                continue
            # Stop check on heading-level/paragraph elements containing
            # a stop phrase
            if sib.name in ('h2', 'h3', 'h4', 'p'):
                if any(sp.lower() in txt.lower() for sp in stop_phrases):
                    break
            if sib.name not in ('h2', 'h3', 'h4', 'h5', 'h6',
                                'p', 'li', 'span', 'div',
                                'strong', 'em', 'ul', 'ol', 'a'):
                continue
            # Parent/child dedupe: skip if already covered by a
            # collected chunk; otherwise drop any prior chunks that
            # are substrings of this broader one.
            if any(txt in prev for prev in collected_texts):
                continue
            collected_texts = {p for p in collected_texts if p not in txt}
            chunks = [c for c in chunks if c not in txt]
            chunks.append(txt)
            collected_texts.add(txt)

        if not chunks:
            return None, None

        content = '\n'.join(chunks)

        # Sanity check: a real picks block should mention "Race"
        # multiple times.
        if len(re.findall(r'\bRace\s+\d+\b', content, re.IGNORECASE)) < 2:
            print("(content found but no race blocks parsed)", end=' ')
            return None, None

        # Lightweight cleanup
        content = re.sub(r'\n{3,}', '\n\n', content).strip()

        if len(content) < 100:
            return None, None

        return content, target_date or page_date

    except Exception as e:
        print(f"(error: {e})", end=' ')
        return None, None


def scrape_fanodds(url, target_date=None):
    """
    Scrape Fan Odds picks for a track.

    Fan Odds publishes a numeric-only pick format (just program
    numbers, no analysis prose), e.g.:
        R1 Race 1 - Starter Allowance · 3,2,5,4 · 12:45pm · Dirt
        R2 Race 2 - Maiden Special Weight · 6,7,5,2 · 01:14pm · Dirt
        ...

    Important caveat: Fan Odds exposes only a today URL — no
    date-suffixed historical variant has been observed. So unlike
    ATR, this can't be backfilled. The caller (scrape_track) skips
    this source during --backfill-atr runs for that reason.

    No date check is possible since the page doesn't include a
    machine-readable date — we trust target_date from the caller,
    same posture as the Churchill house-experts scraper.

    Args:
        url: full Fan Odds picks URL for this track.
        target_date: date the caller says these picks are for. Only
                     used as a fallback return value.

    Returns: (raw_text, target_date) or (None, None) if no content.
    """
    try:
        resp = _fetch(url)
        if resp.status_code != 200:
            print(f"(HTTP {resp.status_code})", end=' ')
            return None, None

        soup = BeautifulSoup(resp.content, 'html.parser')

        # Pull all visible text. Then trim to the picks block:
        # start at the first occurrence of "Race 1" and stop at one
        # of the footer-ish markers we expect to see after the
        # last race. We don't try to be clever with DOM walking
        # here — Fan Odds is a Next.js-style site whose markup
        # changes more often than ATR's, so a text-based extraction
        # is more durable.
        text = soup.get_text(separator='\n')

        # Find the start of the picks block
        start_match = re.search(r'(?im)^\s*(?:R1\s*)?Race\s*1\b', text)
        if not start_match:
            print("(no 'Race 1' anchor found)", end=' ')
            return None, None
        content = text[start_match.start():]

        # Trim footer. These are page sections that always come
        # after the picks list — first match wins.
        for marker in [
            'For all entries at',     # entry-page footer copy
            'For all picks at',       # picks-page footer copy
            'Free Picks',             # nav repeated in footer
            'About Us',
            '© ',
            'Terms of Use',
            'Privacy Policy',
        ]:
            idx = content.find(marker)
            if idx > 0:
                content = content[:idx]

        # Cleanup: collapse blank line runs, strip per-line
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = '\n'.join(line.strip() for line in content.split('\n'))
        content = re.sub(r'\n{3,}', '\n\n', content).strip()

        # Sanity: should have multiple race mentions
        race_count = len(re.findall(r'\bRace\s+\d+\b', content, re.IGNORECASE))
        if race_count < 2:
            print(f"(only {race_count} race mention(s) found)", end=' ')
            return None, None

        if len(content) < 100:
            return None, None

        return content, target_date or date.today()

    except Exception as e:
        print(f"(error: {e})", end=' ')
        return None, None


# Register scrapers
SCRAPERS = {
    'keeneland': scrape_keeneland,
    # Note: 'churchill-downs' uses scrape_churchill_all, which has a different
    # signature than the per-expert scrapers. scrape_track() handles
    # this with a special branch below.
}


# =============================================================================
# MAIN
# =============================================================================

def discover_keeneland_experts():
    """Auto-discover experts from the Keeneland index page."""
    url = 'https://www.keeneland.com/racing/expert-picks'
    try:
        resp = _fetch(url)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.content, 'html.parser')
        experts = []
        seen_ids = set()

        # Find all "view all posts" links
        for link in soup.find_all('a', href=True):
            href = link['href']
            m = re.search(r'/racing/expert-picks/(\d+)', href)
            if m and m.group(1) not in seen_ids:
                eid = m.group(1)
                seen_ids.add(eid)

                # Get the name from the nearest h4
                parent = link.find_parent(['div', 'section', 'article'])
                name_tag = parent.find('h4') if parent else None
                if name_tag:
                    raw_name = name_tag.get_text().strip()
                    # Convert "Scott Hazelton" → "scott_hazelton"
                    clean = re.sub(r'[^a-zA-Z\s]', '', raw_name)
                    snake = '_'.join(clean.lower().split())
                    if snake:
                        experts.append({'id': eid, 'name': snake})

        return experts
    except Exception as e:
        print(f"  Discovery failed: {e}")
        return []


def scrape_track(track, target_date=None, auto_discover=True,
                 backfill_atr=False):
    """Scrape all experts for a track and save raw text files.

    backfill_atr: when True, skip the primary track scraper and ONLY
    fetch the ATR secondary source (if configured). Use this when
    backfilling past dates — the primary track sites generally only
    surface current-day picks (Churchill especially), so running
    them with --date in the past would write today's content under
    a historical date. ATR has true dated URLs, so it backfills
    correctly.
    """
    config = TRACK_CONFIGS.get(track)
    if not config:
        print(f"Unknown track: {track}")
        print(f"Available: {list(TRACK_CONFIGS.keys())}")
        return

    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime('%Y-%m-%d')
    out_dir = DATA_PATH / 'raw_expert_picks' / date_str / track
    sent_dir = DATA_PATH / 'already_sent_eps' / date_str / track
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Churchill path: one fetch returns all experts in a dict.
    # We can't skip individual experts before fetching (single page),
    # so we fetch unconditionally and skip per-file at write time.
    # ---------------------------------------------------------------
    if config['scraper'] == 'churchill-downs':
        if backfill_atr:
            # Skip Churchill house experts entirely — Churchill's
            # expert-picks page has no historical URL, so requesting
            # a past date would just write today's picks under that
            # past date's directory.
            print(f"Backfilling ATR-only for {track} on {date_str}...")
            print(f"  Output: {out_dir}/")
            print(f"  (skipping Churchill house experts; "
                  f"only ATR has dated URLs)")
            print()
        else:
            print(f"Scraping {track} experts for {date_str}...")
            print(f"  Output: {out_dir}/")
            print(f"  (single-page fetch; date is trusted from --date arg)")
            print()

            results, found_date = scrape_churchill_all(target_date)

            if not results:
                print("  No expert content parsed from Churchill page.")
                print("  This can happen on dark days or if page layout changed.")
            else:
                known = set(config.get('known_experts', []))
                scraped_names = set(results.keys())

                # Surface unexpected and missing experts so you notice staff changes
                new_experts = scraped_names - known
                if new_experts:
                    print(f"  ⚠ Found expert(s) not in known list: {sorted(new_experts)}")
                    print(f"    Add to TRACK_CONFIGS['churchill-downs']['known_experts'] when convenient.")
                missing = known - scraped_names
                if missing:
                    print(f"  Note: known expert(s) had no picks today: {sorted(missing)}")

                found = 0
                skipped = 0
                for name, raw_text in results.items():
                    if (sent_dir / f"{name}.txt").exists():
                        skipped += 1
                        continue
                    if (out_dir / f"{name}.txt").exists():
                        skipped += 1
                        continue

                    print(f"  {name}...", end=' ')
                    out_path = out_dir / f"{name}.txt"
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(raw_text)
                    race_count = len(re.findall(r'\bRace\s+\d+\b', raw_text, re.IGNORECASE))
                    print(f"saved ({len(raw_text)} chars, ~{race_count} race mentions)")
                    found += 1

                if skipped:
                    print(f"\n  Skipped {skipped} already-scraped/sent expert(s)")
                print(f"Done: {found}/{len(results)} experts saved to {out_dir}/")

        # ---------------------------------------------------------------
        # Secondary source: At The Races consolidated tips.
        # Saved as a virtual expert 'at_the_races' in the same dir.
        # Independent of the Churchill scrape — runs even when
        # Churchill is skipped (backfill_atr) or returned nothing.
        # ---------------------------------------------------------------
        atr_url = config.get('atr_url')
        if atr_url:
            atr_name = 'at_the_races'
            if (sent_dir / f"{atr_name}.txt").exists() or \
               (out_dir / f"{atr_name}.txt").exists():
                print(f"\n  {atr_name}... already scraped/sent, skipping")
            else:
                # Pull the track segment out of the URL
                # (e.g. '.../race-by-race-guides/Churchill-Downs')
                track_segment = atr_url.rstrip('/').rsplit('/', 1)[-1]
                print(f"\n  {atr_name} ({track_segment})...", end=' ')
                atr_text, _ = scrape_atr(track_segment, target_date)
                if atr_text is None:
                    print("no picks found")
                else:
                    out_path = out_dir / f"{atr_name}.txt"
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(atr_text)
                    race_count = len(re.findall(
                        r'\bRace\s+\d+\b', atr_text, re.IGNORECASE
                    ))
                    print(f"saved ({len(atr_text)} chars, "
                          f"~{race_count} race mentions)")

        # ---------------------------------------------------------------
        # Tertiary source: Fan Odds picks (numeric only).
        # Like Churchill house experts, Fan Odds has no historical URL,
        # so we skip it during --backfill-atr runs (would otherwise
        # write today's picks under a past date directory).
        # ---------------------------------------------------------------
        # fanodds_url = config.get('fanodds_url')
        # if fanodds_url and not backfill_atr:
        #     fo_name = 'fan_odds'
        #     if (sent_dir / f"{fo_name}.txt").exists() or \
        #        (out_dir / f"{fo_name}.txt").exists():
        #         print(f"\n  {fo_name}... already scraped/sent, skipping")
        #     else:
        #         print(f"\n  {fo_name}...", end=' ')
        #         fo_text, _ = scrape_fanodds(fanodds_url, target_date)
        #         if fo_text is None:
        #             print("no picks found")
        #         else:
        #             out_path = out_dir / f"{fo_name}.txt"
        #             with open(out_path, 'w', encoding='utf-8') as f:
        #                 f.write(fo_text)
        #             race_count = len(re.findall(
        #                 r'\bRace\s+\d+\b', fo_text, re.IGNORECASE
        #             ))
        #             print(f"saved ({len(fo_text)} chars, "
        #                   f"~{race_count} race mentions)")
        return

    # ---------------------------------------------------------------
    # Per-expert path (Keeneland and any future track that mirrors it)
    # ---------------------------------------------------------------
    scraper_fn = SCRAPERS.get(config['scraper'])
    if not scraper_fn:
        print(f"No scraper for: {config['scraper']}")
        return

    # Merge hardcoded + discovered experts
    experts = list(config['experts'])
    known_ids = {e['id'] for e in experts}

    if auto_discover and track == 'keeneland':
        discovered = discover_keeneland_experts()
        for e in discovered:
            if e['id'] not in known_ids:
                print(f"  Discovered new expert: {e['name']} (id={e['id']})")
                experts.append(e)
                known_ids.add(e['id'])

    print(f"Scraping {track} experts for {date_str}...")
    print(f"  Output: {out_dir}/")

    found = 0
    skipped = 0
    for expert in experts:
        name = expert['name']
        eid = expert['id']

        # Skip if already sent
        if (sent_dir / f"{name}.txt").exists():
            skipped += 1
            continue

        # Skip if already scraped (raw file exists)
        if (out_dir / f"{name}.txt").exists():
            skipped += 1
            continue

        print(f"\n  {name} (id={eid})...", end=' ')

        raw_text, found_date = scraper_fn(eid, target_date)

        if raw_text is None:
            print("no picks found")
            continue

        if found_date != target_date:
            print(f"wrong date ({found_date})")
            continue

        # Save raw text
        out_path = out_dir / f"{name}.txt"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)

        # Quick stats
        lines = raw_text.strip().split('\n')
        race_count = len(re.findall(r'(?:^|\n)\s*\d+(?:st|nd|rd|th)\s+Race', raw_text))
        print(f"saved ({len(raw_text)} chars, ~{race_count} races)")
        found += 1

    if skipped:
        print(f"\n  Skipped {skipped} already-scraped/sent expert(s)")
    print(f"Done: {found}/{len(experts)} experts saved to {out_dir}/")


def parse_date_arg(date_str):
    """Parse YYYY-MM-DD or M-D-YYYY."""
    for fmt in ['%Y-%m-%d', '%m-%d-%Y', '%m-%d-%y']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Can't parse date: {date_str}")


def main():
    parser = argparse.ArgumentParser(description='Scrape expert picks from track websites')
    parser.add_argument('--track', type=str, required=True,
                        help=f'Track name ({", ".join(TRACK_CONFIGS.keys())})')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--backfill-atr', action='store_true',
                        help='Skip the primary track scraper and only fetch '
                             'At The Races tips. Use with --date for past '
                             'dates: Churchill\'s expert-picks page has no '
                             'historical URL, so without this flag a past '
                             '--date would write today\'s Churchill picks '
                             'under the wrong date directory.')
    args = parser.parse_args()

    target = parse_date_arg(args.date) if args.date else None
    scrape_track(args.track, target, backfill_atr=args.backfill_atr)


if __name__ == '__main__':
    main()