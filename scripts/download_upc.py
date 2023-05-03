import os
import requests
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime

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
        if len(row['class']) > 0:
            if row['class'][0] == 'scratched':
                running = 0
                program_number = int(cols[0]['data-label'].split()[-1])
            else:
                raise ValueError()
        else:
            running = 1
            program_number = int(cols[0].find('img')['alt'])
        horse = cols[3].find('h4').text.strip()
        sire = cols[3].find('p').text.strip()
        trainer = cols[4].find('p').text.strip()
        jockey = cols[4].find_all('p')[-1].text.strip()
        morning_line_odds = cols[6].find('p').text.strip()
        table_data.append([running, program_number, horse, sire, trainer, jockey, morning_line_odds])
    
    return pd.DataFrame(table_data, columns=['running','horse','horse_name','sire','trainer','jockey','ml_odds'])


def main():
    parser = argparse.ArgumentParser(description='Downloads results and morning lines for a given racetrack and date.')
    parser.add_argument('--track', type=str, help='The name of the racetrack')
    args = parser.parse_args()

    if args.track:
        print(f"Track: {args.track}")

    date = datetime.strftime(datetime.today(),'%Y-%m-%d')
    # Get the entries
    url = f"https://entries.horseracingnation.com/entries-results/{args.track}/{date}"
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
    entries.to_csv(f'../data/upc_entries/{args.track}.csv', index=False)
    meta.to_csv(f'../data/upc_meta/{args.track}.csv', index=False)

    print("Today's Entries: ", entries.head())
    print("\nMeta: ", meta.head())

if __name__ == '__main__':
    main()

