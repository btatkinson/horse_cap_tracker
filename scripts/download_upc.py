import os
import requests
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = Path(os.environ['DATA_PATH'])

### parsing functions ###
def parse_meta(info, restriction, purse):
    info = info.text.split(',\n')
    info = [x.strip() for x in info]
    race_dist, race_track, race_type = info
    restriction = restriction.text.strip()
    purse = purse.text.replace('Purse: ','').strip()
    return pd.DataFrame([[race_dist, race_track, race_type, restriction, purse]], columns=['Distance','Surface','Type','Restriction','Purse'])


def parse_entries(table):
    table_data = []
    rows = table.find_all('tr')
    
    for row in rows:
        cols = row.find_all('td')
        
        # Skip rows that don't have enough columns (like header rows)
        if len(cols) < 6:
            continue
            
        # Check if horse is scratched
        if len(row.get('class', [])) > 0:
            if 'scratched' in row['class']:
                running = 0
                program_number = str(cols[0]['data-label'].split()[-1])
            else:
                # Handle other classes if needed
                running = 1
                program_number = str(cols[0].find('img')['alt'])
        else:
            running = 1
            program_number = str(cols[0].find('img')['alt'])
        
        # Extract horse name from the link text
        horse_element = cols[2].find('h4')
        if horse_element:
            horse_link = horse_element.find('a')
            if horse_link:
                horse_name = horse_link.text.strip()
            else:
                # Scratched horses might not have a link, just text
                horse_name = horse_element.text.strip()
                
                running = 0  # If there's no link, assume scratched
        else:
            horse_name = ''
            running = 0  # If no horse element, assume scratched
        
        # Extract other fields
        sire = cols[2].find('p').text.strip() if cols[2].find('p') else ''
        trainer_p = cols[3].find('p')
        trainer = trainer_p.text.strip() if trainer_p else ''
        jockey_ps = cols[3].find_all('p')
        jockey = jockey_ps[-1].text.strip() if jockey_ps else ''
        ml_odds_p = cols[5].find('p')
        morning_line_odds = ml_odds_p.text.strip() if ml_odds_p else ''
        
        # Append with 'horse' as program_number and 'horse_name' as the actual name
        table_data.append([
            running, 
            program_number,  # This goes in 'horse' column
            horse_name,      # This goes in 'horse_name' column
            sire, 
            trainer, 
            jockey, 
            morning_line_odds
        ])
    
    return pd.DataFrame(
        table_data, 
        columns=['running', 'horse', 'horse_name', 'sire', 'trainer', 'jockey', 'ml_odds']
    )

def main():
    parser = argparse.ArgumentParser(description='Downloads results and morning lines for a given racetrack and date.')
    parser.add_argument('--track', type=str, required=True, help='The name of the racetrack')
    parser.add_argument('--date', type=str, default=None,
                        help='Date to fetch (YYYY-MM-DD). Defaults to today.')
    args = parser.parse_args()

    if args.date:
        date = args.date
    else:
        date = datetime.strftime(datetime.today(), '%Y-%m-%d')

    print(f"Track: {args.track}  Date: {date}")
    # Get the entries
    url = f"https://entries.horseracingnation.com/entries-results/{args.track}/{date}"
    print(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    entry_tables = soup.find_all('table', {'class': 'table-entries'})

    race_info = soup.find_all('div', {'class': 'race-distance'})
    race_restrictions = soup.find_all('div', {'class': 'race-restrictions'})
    race_purses = soup.find_all('div', {'class': 'race-purse'})

    assert(len(entry_tables) == len(race_info))
    assert(len(entry_tables) == len(race_restrictions))
    assert(len(entry_tables) == len(race_purses))

    entries = []
    meta = []

    for i in tqdm(range(len(entry_tables))):
        race_no = i+1

        entry_table = entry_tables[i]
        info = race_info[i]
        restrictions = race_restrictions[i]
        purse = race_purses[i]

        entry_df = parse_entries(entry_table)
        entry_df['track'] = args.track
        entry_df['date'] = date
        entry_df['race'] = race_no
        meta_row = parse_meta(info, restrictions, purse)
        meta_row['track'] = args.track
        meta_row['date'] = date
        meta_row['race'] = race_no
        meta.append(meta_row)
        entries.append(entry_df)
        
    entries = pd.concat(entries, axis=0).reset_index(drop=True)
    meta = pd.concat(meta, axis=0).reset_index(drop=True)

    upc_dir = DATA_PATH / 'upc_entries'
    meta_dir = DATA_PATH / 'upc_meta'
    upc_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    entries.to_csv(upc_dir / f'{args.track}.csv', index=False)
    meta.to_csv(meta_dir / f'{args.track}.csv', index=False)

    print("Today's Entries: ", entries.head())
    print("\nMeta: ", meta.head())

if __name__ == '__main__':
    main()