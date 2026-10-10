import os
import re
import requests
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup

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
    cols = row.find_all('td')
    raw_name = cols[0].text.strip()
    horse_name = clean_horse_name(raw_name)
    beyer = extract_beyer(raw_name)
    win_paid = cols[2].text.strip()
    place_paid = cols[3].text.strip()
    show_paid = cols[4].text.strip()
    if win_paid == '-':
        win_paid = 0
    if place_paid == '-':
        place_paid = 0
    if show_paid == '-':
        show_paid = 0
    
    return [finish, horse_name, win_paid, place_paid, show_paid, beyer]

def create_payout_table(table, ar):

    rows = table.find_all('tr')
    
    # Check if we actually have payout data (race might not have been run yet)
    if len(rows) < 4:
        return pd.DataFrame(columns=['finish','horse_name','win_paid','place_paid','show_paid','actual_beyer'])
    
    win_row = rows[1]
    place_row = rows[2]
    show_row = rows[3]
    payout_data = [extract_payouts(win_row,1),extract_payouts(place_row,2),extract_payouts(show_row,3)]
    if len(rows)>=5:
        fourth_row = rows[4]
        payout_data.append(extract_payouts(fourth_row,4))
    
    # Parse also-rans
    ar_text = ar.text.strip().replace('Also rans: ','')
    if ar_text and ar_text.lower() != 'also rans:':
        ars = ar_text.split(', ')
        ar_start = len(rows)
        for i, raw_name in enumerate(ars):
            raw_name = raw_name.strip()
            if not raw_name:
                continue
            horse = clean_horse_name(raw_name)
            beyer = extract_beyer(raw_name)
            payout_data.append([i+ar_start, horse, 0, 0, 0, beyer])
    
    payout_data = pd.DataFrame(payout_data, columns=['finish','horse_name','win_paid','place_paid','show_paid','actual_beyer'])
    
    return payout_data

def format_exotics(exotic_table, race):
    if len(exotic_table.find_all('tr')) < 2:
        return pd.DataFrame([[np.nan, np.nan, np.nan]], columns=['exacta_finish','exacta_$2_payout','exacta_total_pool'])
    exotic_payout_data = []
    exotic_rows = exotic_table.find_all('tr')
    for row in exotic_rows:
        cols = row.find_all('td')
        if len(cols) < 3:
            continue
        exotic_payout_data.append([col.text.strip() for col in cols])
        
    exotic_payout_data = pd.DataFrame(exotic_payout_data, columns=['pool','finish','$2_payout','total_pool'])
    exotic_payout_data['race'] = race
    ## sometimes duplicate pools, I think when carryover
    exotic_payout_data = exotic_payout_data.drop_duplicates(subset=['pool'], keep='first').reset_index(drop=True)
    exotic_payout_data = exotic_payout_data.pivot(index='race', columns=['pool'], values=['finish','$2_payout','total_pool'])
    exotic_payout_data.columns = ['_'.join([col[1].lower(), col[0]]) for col in list(exotic_payout_data)]
    return exotic_payout_data
    

def parse_entries(table):
    table_data = []
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        
        # Scratch detection - check both old method (row class) and new (scratch column)
        row_classes = row.get('class', [])
        if 'scratched' in row_classes:
            running = 0
            program_number = str(cols[0]['data-label'].split()[-1])
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
        
        table_data.append([running, program_number, horse, sire, trainer, jockey, morning_line_odds, prior_beyer])
    
    return pd.DataFrame(table_data, columns=['running','horse','horse_name','sire','trainer','jockey','ml_odds','prior_beyer'])


def assign_missing_finishes(entry_df):
    """
    For horses that are marked running=1 but have no finish position
    (not in payout table or also-rans), assign finish positions after
    the last known finisher.
    
    This handles the case where the also-rans list is incomplete.
    """
    if 'finish' not in entry_df.columns:
        return entry_df
    
    running_mask = entry_df['running'] == 1
    has_finish = entry_df['finish'].notna()
    missing = running_mask & ~has_finish
    
    if missing.sum() == 0:
        return entry_df
    
    # Assign finishes starting after the last known finish
    max_finish = entry_df.loc[has_finish, 'finish'].max() if has_finish.sum() > 0 else 0
    if pd.isna(max_finish):
        max_finish = 0
    
    n_missing = missing.sum()
    entry_df.loc[missing, 'finish'] = range(int(max_finish) + 1, int(max_finish) + 1 + n_missing)
    entry_df.loc[missing, 'win_paid'] = 0
    entry_df.loc[missing, 'place_paid'] = 0
    entry_df.loc[missing, 'show_paid'] = 0
    
    return entry_df


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
        entry_df = entry_df.merge(payout_df, on='horse_name', how='left')
        
        # Fill in missing finishes for horses not in payout table or also-rans
        entry_df = assign_missing_finishes(entry_df)
        
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
    
    # Ensure output directories exist
    os.makedirs('data/entries', exist_ok=True)
    os.makedirs('data/meta', exist_ok=True)
    
    entries.to_csv(f'data/entries/{args.track}_{args.date}.csv', index=False)
    meta.to_csv(f'data/meta/{args.track}_{args.date}.csv', index=False)

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