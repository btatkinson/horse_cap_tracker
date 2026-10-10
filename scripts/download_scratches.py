"""
download_scratches.py

Official Keeneland scratches & changes -> today's field.

HRN's entries pages often don't show scratches (2026-10-08: 24 non-starters,
none flagged), and Equibase blocks scripted requests. Keeneland publishes its
own list at https://www.keeneland.com/racing/scratches-changes, e.g.
    Race 10
    Scratch #1 - Classic Cause
    #15 - Phantom Slayer will run        <- also-eligible drawing in

Parses scratches and AE draw-ins for the requested date and, with --apply,
writes them into {DATA_PATH}/upc_entries/{track}.csv (running=0 / running=1),
which project_openrouter.py reads. Run download_upc.py first: it rewrites
that file from HRN.

Usage:
    python download_scratches.py --date 2026-10-09                  # show
    python download_scratches.py --date 2026-10-09 --apply          # apply to upc_entries
    python download_scratches.py --date 2026-10-09 --apply --wait 120   # poll until posted

Keeneland only (the page is Keeneland's own).
"""

import os, re, time, argparse
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '..' / '.env')
DATA_PATH = Path(os.environ['DATA_PATH'])

URL = "https://www.keeneland.com/racing/scratches-changes"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/129.0 Safari/537.36",
           "Cache-Control": "no-cache", "Pragma": "no-cache"}
DATE_RE = re.compile(r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), '
                     r'(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})$')
SCRATCH_RE = re.compile(r'^Scratch #(\d+[A-Z]?)\s*-?\s*(.+)$', re.I)
# Draw-ins can carry a trailing note: "#16 Glassing will run, rider TBA".
DRAW_IN_RE = re.compile(r'^#(\d+[A-Z]?)\s*-?\s*(.+?)\s+will run\b.*$', re.I)


def fetch(date: str):
    """Return (posted_date, scratches, draw_ins, other_changes); posted_date is
    the date the page currently shows ('YYYY-MM-DD', year assumed from --date)."""
    year = int(date[:4])
    text = BeautifulSoup(requests.get(URL, headers=HEADERS, timeout=30).text, 'html.parser').get_text('\n', strip=True)
    posted, race = None, None
    scratches, draw_ins, other = [], [], []
    for line in (l.strip() for l in text.split('\n')):
        m = DATE_RE.match(line)
        if m:
            posted = datetime.strptime(f"{m.group(2)} {m.group(3)} {year}", "%B %d %Y").strftime('%Y-%m-%d')
            continue
        m = re.match(r'^Race (\d+)$', line)
        if m:
            race = int(m.group(1))
            continue
        if race is None:
            continue
        if (m := SCRATCH_RE.match(line)):
            scratches.append((race, int(re.sub(r'\D', '', m.group(1))), m.group(2).strip()))
        elif (m := DRAW_IN_RE.match(line)):
            draw_ins.append((race, int(re.sub(r'\D', '', m.group(1))), m.group(2).strip()))
        elif line.startswith('#'):
            other.append((race, line))
        elif line.lower().startswith(('copyright', 'follow', 'subscribe')):
            break
    return posted, scratches, draw_ins, other


def apply(track: str, date: str, scratches, draw_ins):
    path = DATA_PATH / 'upc_entries' / f'{track}.csv'
    e = pd.read_csv(path)
    if not (e['date'] == date).any():
        raise SystemExit(f"{path} has no rows for {date}; run download_upc.py first")
    num = pd.to_numeric(e['horse'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    key = e['horse_name'].astype(str).str.lower().str.replace(r'\(.*?\)|[^a-z0-9]', '', regex=True)
    for (race, horse, name), running in [(s, 0) for s in scratches] + [(d, 1) for d in draw_ins]:
        in_race = (e['date'] == date) & (e['race'] == race)
        # Match by name first: the page sometimes has the wrong program number
        # (10/09 R10 listed Amoaray, #15, as "#14").
        hit = in_race & (key == re.sub(r'\(.*?\)|[^a-z0-9]', '', name.lower()))
        if not hit.any():
            hit = in_race & (num == horse)
        elif not (hit & (num == horse)).any():
            print(f"  note: R{race} {name} is not #{horse} in the entries; matched by name")
        if not hit.any():
            print(f"  WARNING: R{race} #{horse} {name} not in entries")
        e.loc[hit, 'running'] = running
    e.to_csv(path, index=False)
    print(f"Applied to {path}: {len(scratches)} scratches, {len(draw_ins)} draw-ins")


def main():
    parser = argparse.ArgumentParser(description="Keeneland's official scratches & changes")
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--track', default='keeneland')
    parser.add_argument('--apply', action='store_true', help='Write into upc_entries/{track}.csv')
    parser.add_argument('--wait', type=float, default=0, help='Minutes to keep polling until the date is posted')
    args = parser.parse_args()

    deadline = time.time() + 60 * args.wait
    while True:
        posted, scratches, draw_ins, other = fetch(args.date)
        if posted == args.date:
            break
        if time.time() >= deadline:
            raise SystemExit(f"Keeneland's page still shows {posted}, not {args.date}")
        print(f"{time.strftime('%H:%M')} page shows {posted}; checking again in 5 min", flush=True)
        time.sleep(300)

    print(f"Keeneland scratches & changes for {posted}:")
    for race, horse, name in scratches:
        print(f"  R{race} scratch #{horse} {name}")
    for race, horse, name in draw_ins:
        print(f"  R{race} draws in #{horse} {name}")
    for race, line in other:
        print(f"  R{race} {line}")
    print("--scratch \"" + ",".join(f"{r}:{h}" for r, h, _ in scratches) + "\"")
    print("--draw-in \"" + ",".join(f"{r}:{h}" for r, h, _ in draw_ins) + "\"")
    if args.apply:
        apply(args.track, args.date, scratches, draw_ins)


if __name__ == '__main__':
    main()
