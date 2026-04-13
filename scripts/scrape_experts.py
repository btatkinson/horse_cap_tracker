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


# Register scrapers
SCRAPERS = {
    'keeneland': scrape_keeneland,
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


def scrape_track(track, target_date=None, auto_discover=True):
    """Scrape all experts for a track and save raw text files."""
    config = TRACK_CONFIGS.get(track)
    if not config:
        print(f"Unknown track: {track}")
        print(f"Available: {list(TRACK_CONFIGS.keys())}")
        return

    scraper_fn = SCRAPERS.get(config['scraper'])
    if not scraper_fn:
        print(f"No scraper for: {config['scraper']}")
        return

    if target_date is None:
        target_date = date.today()

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

    date_str = target_date.strftime('%Y-%m-%d')
    out_dir = DATA_PATH / 'raw_expert_picks' / date_str / track
    sent_dir = DATA_PATH / 'already_sent_eps' / date_str / track
    out_dir.mkdir(parents=True, exist_ok=True)

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
    args = parser.parse_args()

    target = parse_date_arg(args.date) if args.date else None
    scrape_track(args.track, target)


if __name__ == '__main__':
    main()