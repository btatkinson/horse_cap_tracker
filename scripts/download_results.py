import os
import re
import requests
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Match the pattern other scripts use: load .env from the parent dir of
# scripts/, so we can run download_results from any working directory.
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])

### helper functions ###
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def clean_horse_name(name):
    """
    Strip Beyer figures and other parenthetical suffixes from horse names.
    Handles both formats:
      - Payout: 'Suspicions (69*)'    -> 'Suspicions'
      - Entry:  'Traject\\n(85)'       -> 'Traject'
      - Entry:  'Without (IRE)\\n(105)' -> 'Without (IRE)'
    
    Note: country tags like (IRE), (AUS), (GB) are KEPT - only trailing
    numeric parentheticals (with optional *) are stripped.
    """
    # First remove trailing parenthetical with number: (69*), (85), (-1)
    name = re.sub(r'\s*\n?\s*\(-?\d+\*?\)\s*$', '', name)
    return name.strip()


def name_key(name):
    """Matching key for a horse name: lowercase, no parentheticals (country
    tags like (IRE) differ between HRN's entries and results tables), no
    punctuation or spaces."""
    n = re.sub(r'\(.*?\)', '', str(name).lower())
    return re.sub(r'[^a-z0-9]', '', n)


def extract_beyer(raw_name):
    """
    Extract Beyer speed figure from the parenthetical suffix.
    Handles both payout format and entry format:
      - Payout: 'Suspicions (69*)'   -> 69
      - Entry:  'Traject\\n(85)'      -> 85
      - Entry:  'Epic Horsepower\\n(-1)' -> -1
      - No beyer: 'Plain Name'       -> NaN
      - Non-numeric: 'Horse (DH)'    -> NaN
    """
    raw_name = raw_name.strip()
    # Match (number) or (number*) at end, possibly with newline before
    match = re.search(r'\((-?\d+)\*?\)\s*$', raw_name)
    if match:
        return int(match.group(1))
    return np.nan


### parsing functions ###
def parse_meta(info, restriction, purse):
    info = info.text.split(',\n')
    info = [x.strip() for x in info]
    race_dist, race_track, race_type = info
    restriction = restriction.text.strip()
    purse = purse.text.replace('Purse: ','').strip()
    return pd.DataFrame([[race_dist, race_track, race_type, restriction, purse]], columns=['Distance','Surface','Type','Restriction','Purse'])

def extract_payouts(row, finish):
    """
    Pull WPS payout from a single payout-table row.

    DEFENSIVE NOTE: not every payout row has all five columns. Some pages
    render the 4th-place finisher's name as a row but with no win/place/show
    cells (because nothing paid in that slot — e.g., a 5-horse field where
    the show pool only covered 1st-3rd, or a track that only displays the
    superfecta-completing horse without payout). Previously this function
    blindly indexed cols[2:4] and crashed on those rows.

    Now: if the row doesn't have at least 5 columns, we still capture the
    horse name and beyer (which always live in col 0) but record $0 for
    any missing payout columns. This is the right default — a missing
    payout cell means that horse didn't cash in that pool.
    """
    cols = row.find_all('td')
    if not cols:
        return None  # genuinely empty row, caller should skip

    raw_name = cols[0].text.strip()
    horse_name = clean_horse_name(raw_name)
    beyer = extract_beyer(raw_name)

    # Defensively pull cols 2/3/4. Each one defaults to "no payout" if
    # missing. Note we use cols[2/3/4] to match the historical layout
    # (cols[0]=horse, cols[1]=program-number-or-blank, cols[2:5]=W/P/S).
    def _payout(idx):
        if idx >= len(cols):
            return 0
        v = cols[idx].text.strip()
        return 0 if v in ('-', '') else v

    win_paid = _payout(2)
    place_paid = _payout(3)
    show_paid = _payout(4)

    return [finish, horse_name, win_paid, place_paid, show_paid, beyer]

def create_payout_table(table, ar):
    """
    Build the per-horse W/P/S payout table for one race.

    The HRN payout table has a header row plus one row per finish slot,
    where slots can be 1st/2nd/3rd (always paid) plus optionally a 4th
    slot if the superfecta required it. The 4th-slot row may or may not
    actually include payout columns — see extract_payouts for the fix
    (this is what was crashing keeneland 2026-04-24 race 5).

    We accept any row count from 1 (rare — race not run yet) up to
    arbitrarily many. Each row is processed with extract_payouts which
    is now safe against short rows.
    """
    rows = table.find_all('tr')

    if len(rows) < 2:
        # No payout rows at all — race didn't run, or page is malformed.
        return pd.DataFrame(columns=['finish','horse_name','win_paid',
                                      'place_paid','show_paid','actual_beyer'])

    payout_data = []
    # Skip header row (rows[0]) and process all data rows. We assign
    # finish positions 1..N in order rather than assuming exactly 3 or 4 rows.
    # Footnote rows ("* Preliminary Speed Figures") are not horses — counting
    # them shifted every also-ran's finish position down by one.
    for row in rows[1:]:
        result = extract_payouts(row, len(payout_data) + 1)
        if result is not None and not str(result[1]).startswith('*'):
            payout_data.append(result)

    # Parse also-rans (each gets finish positions after the last paid horse)
    ar_text = ar.text.strip().replace('Also rans: ','')
    if ar_text and ar_text.lower() != 'also rans:':
        ars = ar_text.split(', ')
        ar_start = len(payout_data) + 1
        for i, raw_name in enumerate(ars):
            raw_name = raw_name.strip()
            if not raw_name:
                continue
            horse = clean_horse_name(raw_name)
            beyer = extract_beyer(raw_name)
            payout_data.append([i + ar_start, horse, 0, 0, 0, beyer])

    return pd.DataFrame(
        payout_data,
        columns=['finish','horse_name','win_paid','place_paid','show_paid','actual_beyer']
    )

def format_exotics(exotic_table, race):
    """
    Pivot the exotics-payout table into one wide row per race with columns
    like exacta_$2_payout, pick_5_$2_payout, pick_5_total_pool, etc.

    DUPLICATE POOL HANDLING — important for horizontals.
    HRN's results page sometimes lists the same pool name twice in one
    race's exotics table:
      - "Pick 5" (full hit, real payout, real pool)        e.g. $436.60 / $712k
      - "Pick 5" (consolation, lower payout, $0 pool)      e.g. $25.76 / $0
      Or "Pick 6" full vs "Pick 6" consolation.
    The original code did `keep='first'`, which kept whichever appeared
    first in the page. Order isn't guaranteed; you could lose the real
    payout to a consolation row.

    Fix: when duplicates exist, keep the row with the **highest $2 payout**.
    Full-hit payouts always exceed consolation payouts in the same pool,
    so this reliably picks the correct row. Carryovers don't change this
    (a carryover INCREASES the payout, never decreases it).

    There's a separate "Consolation Pick 3" / "Consolation Double" naming
    convention HRN uses sometimes — those get their own pool name and
    don't collide with "Pick 3" / "Daily Double". We keep them as their
    own columns; you can ignore them downstream.
    """
    rows = exotic_table.find_all('tr')
    if len(rows) < 2:
        return pd.DataFrame(
            [[np.nan, np.nan, np.nan]],
            columns=['exacta_finish','exacta_$2_payout','exacta_total_pool']
        )

    exotic_payout_data = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 3:
            continue
        exotic_payout_data.append([col.text.strip() for col in cols])

    df = pd.DataFrame(
        exotic_payout_data,
        columns=['pool', 'finish', '$2_payout', 'total_pool']
    )
    df['race'] = race

    # Convert payout to numeric for comparison; keep the original strings
    # for output. We use a separate numeric column purely for sorting.
    def _payout_num(v):
        try:
            return float(str(v).replace('$', '').replace(',', ''))
        except (ValueError, TypeError):
            return 0.0

    df['_payout_num'] = df['$2_payout'].apply(_payout_num)

    # Sort by payout DESC so duplicates keep the highest-payout row first
    df = df.sort_values('_payout_num', ascending=False)
    df = df.drop_duplicates(subset=['pool'], keep='first').reset_index(drop=True)
    df = df.drop(columns=['_payout_num'])

    df = df.pivot(
        index='race',
        columns=['pool'],
        values=['finish', '$2_payout', 'total_pool']
    )
    df.columns = ['_'.join([col[1].lower(), col[0]]) for col in list(df)]

    # Normalize column names: spaces in pool names ("Daily Double",
    # "Pick 3", "Super High Five") become underscores so the columns are
    # easier to reference programmatically downstream.
    df.columns = [c.replace(' ', '_') for c in df.columns]

    return df
    

def parse_entries(table):
    table_data = []
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        
        # Scratch detection - check both old method (row class) and new
        # ("Scratched?" column, which also labels also-eligibles). For a
        # card with results, mark_starters() below has the final say.
        row_classes = row.get('class', [])
        scratch_td = row.find('td', {'data-label': 'Scratched?'})
        scratch_txt = scratch_td.get_text(' ', strip=True).lower() if scratch_td else ''
        also_eligible = int('also' in scratch_txt and 'elig' in scratch_txt)
        if 'scratched' in row_classes or 'scr' in scratch_txt or also_eligible:
            running = 0
        else:
            running = 1
        img = cols[0].find('img')
        if img:
            program_number = str(img['alt'])
        else:
            program_number = str(cols[0]['data-label'].split()[-1])
        
        # New column layout (6 cols)
        raw_horse = cols[2].find('h4').text.strip()
        sire = cols[2].find('p').text.strip()
        trainer = cols[3].find('p').text.strip()
        jockey = cols[3].find_all('p')[-1].text.strip()
        morning_line_odds = cols[5].find('p').text.strip()
        
        # Extract prior Beyer from entry name if present
        # e.g. "Traject\n(85)" or "Nonstop Vino (92)" -> prior_beyer=85 or 92
        prior_beyer = extract_beyer(raw_horse)
        horse = clean_horse_name(raw_horse)
        # Also strip any remaining newlines/whitespace from the name
        horse = ' '.join(horse.split())
        
        table_data.append([running, program_number, horse, sire, trainer, jockey, morning_line_odds, prior_beyer,
                           also_eligible])

    return pd.DataFrame(table_data, columns=['running','horse','horse_name','sire','trainer','jockey','ml_odds',
                                             'prior_beyer','also_eligible'])


def mark_starters(entry_df, payout_df, race_no):
    """
    Attach finish/payouts to entries and decide who actually started.

    A starter is a horse named in the payout table or the also-rans; every
    other entry (scratch, also-eligible that didn't draw in) gets running=0
    and no finish. Names are matched by name_key, then fuzzily for the
    leftovers (HRN spells some names differently in the two tables).

    This replaces assign_missing_finishes(), which gave every unmatched
    horse a made-up last-place finish: HRN often leaves scratches unflagged,
    so on 2026-04-10 Keeneland 22 of 119 "starters" never ran.
    """
    import difflib
    entry_df = entry_df.copy()
    payout_df = payout_df.copy()
    entry_df['key'] = entry_df['horse_name'].map(name_key)
    payout_df['key'] = payout_df['horse_name'].map(name_key)
    unmatched = [k for k in payout_df['key'] if k not in set(entry_df['key'])]
    for k in unmatched:
        pool = [e for e in entry_df['key'] if e not in set(payout_df['key'])]
        close = difflib.get_close_matches(k, pool, n=1, cutoff=0.8)
        if close:
            payout_df.loc[payout_df['key'] == k, 'key'] = close[0]
        else:
            name = payout_df.loc[payout_df['key'] == k, 'horse_name'].iloc[0]
            print(f"  WARNING race {race_no}: result '{name}' matches no entry")
    merged = entry_df.merge(payout_df.drop(columns=['horse_name']), on='key', how='left').drop(columns=['key'])
    merged['running'] = merged['finish'].notna().astype(int) if len(payout_df) else merged['running']
    return merged


def main():
    parser = argparse.ArgumentParser(description='Downloads results and morning lines for a given racetrack and date.')
    parser.add_argument('--track', type=str, help='The name of the racetrack')
    parser.add_argument('--date', type=str,help='The date of the race in the format YYYY-MM-DD')
    args = parser.parse_args()

    if args.track:
        print(f"Track: {args.track}")
    if args.date:
        print(f"Date: {args.date}")

    # Get the results
    url = f"https://entries.horseracingnation.com/entries-results/{args.track}/{args.date}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    entry_tables = soup.find_all('table', {'class': 'table-entries'})
    payout_tables = soup.find_all('table', {'class': 'table-payouts'})
    also_rans = soup.find_all('div', {'class': 'race-also-rans'})
    exotic_payout_tables = soup.find_all('table', {'class': 'table-exotic-payouts'})
    race_info = soup.find_all('div', {'class': 'race-distance'})
    race_restrictions = soup.find_all('div', {'class': 'race-restrictions'})
    race_purses = soup.find_all('div', {'class': 'race-purse'})
    
    n_entry = len(entry_tables)
    n_payout = len(payout_tables)
    n_also_ran = len(also_rans)
    n_exotic = len(exotic_payout_tables)
    import sys
    if n_entry == 0:
        print(f"ERROR: No entry tables at {url}")
        print("Check track name spelling and that the date had racing.")
        sys.exit(1)

    if n_payout == 0:
        print(f"Found {n_entry} entry tables but ZERO payout tables.")
        print(f"HRN has not posted any results for {args.track}/{args.date} yet.")
        print("Possible causes:")
        print("  - Card hasn't run yet (check post times)")
        print("  - HRN's data feed is lagging (common on big stakes days)")
        print("  - Equibase hasn't pushed chart data through")
        print(f"Try Equibase directly or check back in a few hours.")
        sys.exit(1)

    if n_payout < n_entry:
        print(f"PARTIAL: {n_entry} races, {n_payout} have results.")
        missing_idx = [i+1 for i in range(n_entry)
                    if i >= n_payout]  # naive — see note below
        print(f"Missing results for the LATER races, likely races: {missing_idx}")
        print("Either wait for results to post, or pass --partial to process what's available.")
        # if not args.partial:
        #     sys.exit(1)

    assert(len(entry_tables) == len(payout_tables))
    assert(len(entry_tables) == len(also_rans))
    assert(len(entry_tables) == len(exotic_payout_tables))
    assert(len(entry_tables) == len(race_info))
    assert(len(entry_tables) == len(race_restrictions))
    assert(len(entry_tables) == len(race_purses))

    entries = []
    meta = []

    for i in tqdm(range(len(entry_tables))):
        race_no = i+1

        entry_table = entry_tables[i]
        payout_table = payout_tables[i]
        also_ran = also_rans[i]
        exotic_payout_table = exotic_payout_tables[i]
        info = race_info[i]
        restrictions = race_restrictions[i]
        purse = race_purses[i]

        entry_df = parse_entries(entry_table)
        payout_df = create_payout_table(payout_table, also_ran)
        entry_df = mark_starters(entry_df, payout_df, race_no)

        entry_df['track'] = args.track
        entry_df['date'] = args.date
        entry_df['race'] = race_no
        meta_row = parse_meta(info, restrictions, purse)
        meta_row['track'] = args.track
        meta_row['date'] = args.date
        meta_row['race'] = race_no
        exotic_data = format_exotics(exotic_payout_table, race_no)
        meta_row = pd.concat([meta_row, exotic_data.set_index(meta_row.index)], axis=1)
        meta.append(meta_row)
        entries.append(entry_df)

    entries = pd.concat(entries, axis=0).reset_index(drop=True)
    meta = pd.concat(meta, axis=0).reset_index(drop=True)
    
    # Use DATA_PATH from .env so the script works no matter where it's run from.
    # Previously this used relative paths "data/entries" / "data/meta", which
    # only worked when run from the repo root.
    entries_dir = DATA_PATH / 'entries'
    meta_dir = DATA_PATH / 'meta'
    entries_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    entries_path = entries_dir / f'{args.track}_{args.date}.csv'
    meta_path = meta_dir / f'{args.track}_{args.date}.csv'

    entries.to_csv(entries_path, index=False)
    meta.to_csv(meta_path, index=False)
    print(f"  → entries: {entries_path}")
    print(f"  → meta:    {meta_path}")

    # Summary
    n_races = entries['race'].nunique()
    n_horses = len(entries)
    n_with_finish = entries['finish'].notna().sum()
    n_running = (entries['running'] == 1).sum()
    print(f"\n{n_races} races, {n_horses} horses ({n_running} running)")
    print(f"Finish data: {n_with_finish}/{n_running} running horses have finishes")
    
    # Warn about any remaining gaps
    missing = (entries['running'] == 1) & entries['finish'].isna()
    if missing.sum() > 0:
        print(f"WARNING: {missing.sum()} running horses still missing finish data")
        
if __name__ == '__main__':
    main()