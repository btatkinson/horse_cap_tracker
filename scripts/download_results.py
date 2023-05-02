

import os
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

    for i in tqdm(range(len(entry_tables))):
        entry_table = entry_tables[i]
        payout_table = payout_tables[i]
        also_ran = also_rans[i]
        exotic_payout_table = exotic_payout_tables[i]
        info = race_info[i]
        
        


if __name__ == '__main__':
    main()


