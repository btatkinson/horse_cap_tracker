"""
diagnose_scratch_source.py

Hypothesis: predict_v2 drops legitimate runners during walk-forward
because the upc_entries/{track}.csv file (a per-track snapshot of
pre-race scratches + ML odds) is stale or was captured from a different
race day than the holdout. The post-race entries/*.csv files are the
ground truth for what actually ran.

This script tests the hypothesis on the specific race we have evidence
for: 2026-04-12 / keeneland race 2, where horse 4 appeared in the raw
ai_combined.csv (LLM scrape) and in the post-race entries data with
running=1, but is missing from walk_forward_summary.csv.

Compares per-horse running flags between:
  - upc_entries/keeneland.csv  (the snapshot predict_v2 used)
  - entries/*.csv              (post-race ground truth)

If upc_entries says running=0 (or horse missing entirely) for a horse
that entries says ran (running=1, finish>0), the hypothesis is confirmed.

Usage:
    python diagnose_scratch_source.py
"""

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '..' / '.env')
DATA_PATH = Path(os.environ['DATA_PATH'])

UPC_DIR = DATA_PATH / 'upc_entries'
ENTRIES_DIR = DATA_PATH / 'entries'

TARGET_TRACK = 'keeneland'
TARGET_DATE = '2026-04-12'
TARGET_RACE = 2
TARGET_HORSE = 4


def load_upc(track):
    """Load upc_entries/{track}.csv. Per-track file, not per-date."""
    candidates = [
        UPC_DIR / f'{track}.csv',
        UPC_DIR / f'{track.replace("-", "_")}.csv',
    ]
    for p in candidates:
        if p.exists():
            print(f"  Found {p}")
            df = pd.read_csv(p)
            # Mirror predict_v2's normalization
            if 'horse' in df.columns and 'horse_number' not in df.columns:
                df['horse_number'] = pd.to_numeric(df['horse'], errors='coerce')
            if 'race' in df.columns:
                df['race'] = pd.to_numeric(df['race'], errors='coerce')
            return df, p
    return None, None


def load_entries_for(date_str, track):
    """Load post-race entries CSVs and filter to one card. Returns the
    filtered df."""
    if not ENTRIES_DIR.exists():
        return None
    frames = []
    for f in ENTRIES_DIR.glob('*.csv'):
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None
    entries = pd.concat(frames, ignore_index=True)

    # Normalize keys
    aliases = {
        'gulfstream-park': 'gulfstream', 'gulfstream park': 'gulfstream',
        'churchill-downs': 'churchill', 'churchill downs': 'churchill',
    }
    if 'track' in entries.columns:
        entries['track'] = entries['track'].astype(str).str.strip().str.lower()
        entries['track'] = entries['track'].replace(aliases)
    if 'horse' in entries.columns and 'horse_number' not in entries.columns:
        entries['horse_number'] = pd.to_numeric(entries['horse'], errors='coerce')
    target = aliases.get(track.lower(), track.lower())
    return entries[
        (entries.get('date') == date_str)
        & (entries.get('track') == target)
    ].copy()


def main():
    print(f"Investigating: {TARGET_DATE} / {TARGET_TRACK} race {TARGET_RACE}")
    print(f"Suspect horse: #{TARGET_HORSE}")
    print()

    # ── upc_entries ─────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"Source A: upc_entries/{TARGET_TRACK}.csv (predict_v2's scratch source)")
    print(f"{'='*72}")
    upc, upc_path = load_upc(TARGET_TRACK)
    if upc is None:
        print(f"  upc_entries file NOT FOUND. Cannot compare.")
        print(f"  Note: this file is per-track, no date in the name. It")
        print(f"  may have been overwritten by a later card or deleted.")
        return

    # Show all columns so we know what we're dealing with
    print(f"\n  Columns: {list(upc.columns)}")
    print(f"  Total rows: {len(upc)}")
    if 'race' in upc.columns:
        print(f"  Races present: {sorted(upc['race'].dropna().unique())}")

    # Filter to target race
    if 'race' in upc.columns:
        race_upc = upc[upc['race'] == TARGET_RACE].copy()
        print(f"\n  Race {TARGET_RACE} rows in upc_entries: {len(race_upc)}")
        if not race_upc.empty:
            cols_show = [c for c in
                         ('horse_number', 'horse', 'running', 'ml_odds', 'name')
                         if c in race_upc.columns]
            print(race_upc[cols_show].to_string(index=False))

            # Specifically: is the suspect horse running?
            suspect = race_upc[race_upc['horse_number'] == TARGET_HORSE]
            print(f"\n  Suspect horse #{TARGET_HORSE} in upc_entries:")
            if suspect.empty:
                print(f"    *** NOT IN upc_entries — predict_v2 dropped it as 'late scratch / AI-only'")
                print(f"    This is the bug: the horse exists in entries (it ran) but")
                print(f"    upc_entries doesn't have it, so predict_v2 treats it as a")
                print(f"    scratch and renormalizes the LLM probs across only the horses")
                print(f"    that DID make it into upc_entries.")
            else:
                running = suspect.iloc[0].get('running', None)
                print(f"    running={running}")
                if running == 0:
                    print(f"    *** running=0 here. predict_v2 drops on this flag.")
                else:
                    print(f"    running ≠ 0 here. predict_v2 should NOT have dropped this horse.")
                    print(f"    Hypothesis disconfirmed for this row.")

    # ── post-race entries ────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Source B: entries/*.csv for {TARGET_DATE} / {TARGET_TRACK}")
    print(f"  (post-race ground truth)")
    print(f"{'='*72}")
    ent = load_entries_for(TARGET_DATE, TARGET_TRACK)
    if ent is None or ent.empty:
        print(f"  No entries data found for that card.")
    else:
        race_ent = ent[ent['race'] == TARGET_RACE].copy()
        print(f"\n  Race {TARGET_RACE} rows in entries: {len(race_ent)}")
        cols_show = [c for c in
                     ('horse_number', 'horse', 'running', 'finish', 'name')
                     if c in race_ent.columns]
        if cols_show:
            print(race_ent[cols_show].sort_values('horse_number').to_string(index=False))

        suspect = race_ent[race_ent['horse_number'] == TARGET_HORSE]
        print(f"\n  Suspect horse #{TARGET_HORSE} in entries:")
        if suspect.empty:
            print(f"    NOT IN entries either — horse may not have existed.")
            print(f"    Hypothesis weakened.")
        else:
            row = suspect.iloc[0]
            print(f"    running={row.get('running', '?')}, "
                  f"finish={row.get('finish', '?')}")
            if row.get('running') == 1 and row.get('finish', 0) > 0:
                print(f"    *** Confirmed: this horse RAN and FINISHED.")
                print(f"    If upc_entries dropped it, we have our bug.")

    # ── compare horse counts side by side ─────────────────────────
    print(f"\n{'='*72}")
    print(f"Side-by-side: which horses are in each source for race {TARGET_RACE}?")
    print(f"{'='*72}")
    if upc is not None and 'race' in upc.columns:
        race_upc = upc[upc['race'] == TARGET_RACE]
        upc_horses = set(race_upc['horse_number'].dropna().astype(int).tolist())
    else:
        upc_horses = set()
    if ent is not None and not ent.empty:
        race_ent = ent[ent['race'] == TARGET_RACE]
        ent_horses = set(race_ent['horse_number'].dropna().astype(int).tolist())
        ent_running = set(
            race_ent[race_ent.get('running', 1) == 1]['horse_number']
            .dropna().astype(int).tolist()
        )
    else:
        ent_horses = set()
        ent_running = set()

    print(f"  Horses in upc_entries (race {TARGET_RACE}):       "
          f"{sorted(upc_horses)}")
    print(f"  Horses in entries (race {TARGET_RACE}, all):      "
          f"{sorted(ent_horses)}")
    print(f"  Horses in entries (race {TARGET_RACE}, running):  "
          f"{sorted(ent_running)}")
    diff = ent_running - upc_horses
    print(f"  In entries(running) but NOT in upc:           {sorted(diff)}")
    if diff:
        print(f"\n  → These horses ran but predict_v2 never saw them as runners.")
        print(f"    Their LLM probability mass got redistributed to the")
        print(f"    horses that DID survive predict_v2's filter, inflating")
        print(f"    everyone else by mass(missing horses) / sum(remaining).")
    else:
        print(f"\n  → upc_entries and entries agree for this race.")
        print(f"    Hypothesis disconfirmed for this race specifically.")
        print(f"    The 0.38→0.49 inflation must come from somewhere else.")

    # ── Bonus: does the upc_entries file even cover the right card? ─
    print(f"\n{'='*72}")
    print(f"Sanity: is upc_entries/{TARGET_TRACK}.csv still pointing at "
          f"{TARGET_DATE}'s card?")
    print(f"{'='*72}")
    if upc_path is not None:
        mtime = pd.Timestamp(upc_path.stat().st_mtime, unit='s')
        print(f"  File mtime: {mtime}")
        print(f"  Race day:   {TARGET_DATE}")
        print(f"  → If mtime is much later than the race day, this file was")
        print(f"    overwritten by a later card. Walk-forward re-running")
        print(f"    predict_v2 on a historical date would have read THIS file,")
        print(f"    not whatever existed on race day. That's a per-track-file")
        print(f"    architecture issue: scratch state isn't versioned by date.")
    if 'date' in upc.columns:
        print(f"\n  upc_entries 'date' column: {upc['date'].unique()}")
    else:
        print(f"  upc_entries has no 'date' column — the file represents")
        print(f"  a single moment in time without saying which day.")


if __name__ == '__main__':
    main()