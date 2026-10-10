"""
diagnose_meta_lookup.py — quick standalone diagnostic for the meta
file lookup. No imports from backtest_kfold; just looks at what's
on disk vs what backtest_kfold thinks the track name is.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '..' / '.env')
DATA_PATH = Path(os.environ['DATA_PATH'])
META_DIR = DATA_PATH / 'meta'

print(f"DATA_PATH: {DATA_PATH}")
print(f"META_DIR:  {META_DIR}")
print(f"META_DIR exists: {META_DIR.exists()}")
print()

# 1. What files actually exist?
print("All meta files matching 'churchill*' on disk:")
if META_DIR.exists():
    matches = sorted(META_DIR.glob('churchill*'))
    for p in matches:
        print(f"  {p.name!r}  ({p.stat().st_size} bytes)")
    if not matches:
        print("  (none)")
print()

print("All meta files matching '*2026-04-26*' on disk:")
if META_DIR.exists():
    matches = sorted(META_DIR.glob('*2026-04-26*'))
    for p in matches:
        print(f"  {p.name!r}")
    if not matches:
        print("  (none)")
print()

# 2. Build the same candidate list the backtest builds, check each
def _meta_path_candidates(track, date):
    REVERSE_ALIASES = {
        'churchill': ['churchill-downs', 'churchill downs'],
        'gulfstream': ['gulfstream-park', 'gulfstream park'],
    }
    candidates = [META_DIR / f'{track}_{date}.csv']
    for variant in REVERSE_ALIASES.get(track, []):
        candidates.append(META_DIR / f'{variant}_{date}.csv')
    return candidates

print("Candidate paths checked for ('churchill', '2026-04-26'):")
for p in _meta_path_candidates('churchill', '2026-04-26'):
    print(f"  exists={p.exists()}  {p}")
print()

print("Candidate paths checked for ('churchill', '2026-04-25'):")
for p in _meta_path_candidates('churchill', '2026-04-25'):
    print(f"  exists={p.exists()}  {p}")
print()

# 3. What does the predictions df actually contain for churchill?
# Try both walk_forward and cv sources; show distinct track values
import pandas as pd

wf_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
cv_path = DATA_PATH / 'regression_predictions.csv'

for label, path in [('walk_forward', wf_path), ('cv', cv_path)]:
    print(f"{label} source: {path}")
    if not path.exists():
        print(f"  (file does not exist)")
        continue
    df = pd.read_csv(path)
    if 'track' not in df.columns:
        print(f"  (no 'track' column)")
        continue
    # Find distinct values
    tracks = df['track'].dropna().unique()
    print(f"  Distinct track values ({len(tracks)}):")
    for t in sorted(tracks):
        n = (df['track'] == t).sum()
        # Show repr to surface any whitespace/casing weirdness
        print(f"    {t!r:<30}  ({n} rows)")
    print()

print("If the candidate paths show exists=False but you see the file in")
print("the disk listing above, paste BOTH of those outputs to me — there's")
print("likely a hidden character or case mismatch we need to handle.")