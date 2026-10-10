"""
diagnose_blend.py

Quick diagnostic: is predict_v2's ensemble_prob actually different from
ml_prob_norm? The backtest_v5 component table showed identical numbers
across all sweep fractions, which suggests the blend is silently broken
upstream. This script answers definitively by examining the columns
directly rather than inferring from quantile cutoffs.

Run after backtest_walk_forward.py has populated walk_forward_summary.csv.

Usage:
    python diagnose_blend.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '..' / '.env')
DATA_PATH = Path(os.environ['DATA_PATH'])

WF_PATH = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'


def main():
    if not WF_PATH.exists():
        raise SystemExit(f"{WF_PATH} not found. Run backtest_walk_forward.py first.")

    df = pd.read_csv(WF_PATH)
    print(f"Loaded {len(df)} rows from {WF_PATH.name}\n")

    cols_needed = ['ensemble_prob', 'ml_prob_norm', 'ml_win_prob_norm']
    have = [c for c in cols_needed if c in df.columns]
    print(f"Columns present: {have}")

    # predict_v2 writes 'ml_win_prob_norm' (its native column name);
    # backtest_kfold's loader copies it to 'ml_prob_norm'. The CSV
    # straight from disk has the predict_v2 names — check both.
    ml_col = None
    for candidate in ['ml_prob_norm', 'ml_win_prob_norm']:
        if candidate in df.columns:
            ml_col = candidate
            break
    if ml_col is None:
        raise SystemExit("Neither ml_prob_norm nor ml_win_prob_norm in df.")
    if 'ensemble_prob' not in df.columns:
        raise SystemExit("ensemble_prob not in df.")

    print(f"Comparing 'ensemble_prob' vs '{ml_col}'\n")

    a = df['ensemble_prob']
    b = df[ml_col]

    # 1. Hard equality
    both_present = a.notna() & b.notna()
    n_both = both_present.sum()
    n_equal = (a[both_present] == b[both_present]).sum()
    print(f"Both columns present:        {n_both}")
    print(f"Exactly equal:               {n_equal}  ({n_equal/n_both:.1%})")
    print(f"  → If 100%: blend is silently passing ML through unchanged.")
    print(f"  → If 0%: blend is fully active.")
    print(f"  → Mixed: blend works on some rows, not others.\n")

    # 2. Diff distribution
    diff = (a - b)[both_present]
    abs_diff = diff.abs()
    print(f"Absolute diff (|ensemble_prob − {ml_col}|):")
    print(f"  mean:   {abs_diff.mean():.6f}")
    print(f"  median: {abs_diff.median():.6f}")
    print(f"  max:    {abs_diff.max():.6f}")
    print(f"  std:    {abs_diff.std():.6f}\n")

    # 3. Distribution of signed diffs — does the blend systematically
    # shift up or down? Or is it symmetric noise?
    print(f"Signed diff (ensemble_prob − {ml_col}):")
    print(f"  mean:   {diff.mean():+.6f}")
    print(f"  q10:    {diff.quantile(0.10):+.6f}")
    print(f"  q50:    {diff.quantile(0.50):+.6f}")
    print(f"  q90:    {diff.quantile(0.90):+.6f}\n")

    # 4. Pearson correlation — even if values differ, do they rank
    # together? Identical rankings = same picks at every quantile,
    # which is what we saw in the backtest sweep.
    rho_pearson = a[both_present].corr(b[both_present])
    print(f"Pearson correlation:         {rho_pearson:.6f}")
    print(f"  → If 1.000: linearly identical (e.g. blend = ML × constant).")
    print(f"  → If close to 1.0 but not 1.0: blend is mostly ML, slight perturbation.\n")

    # 5. Spearman (rank) — this is the one that actually matters for
    # backtest_v5's quantile-based picks. If rho_spearman == 1.0, the
    # blend ranks horses identically to ML, so the sweep cutoffs land
    # on the same horses regardless of which column you use.
    from scipy.stats import spearmanr
    rho_spearman, _ = spearmanr(a[both_present], b[both_present])
    print(f"Spearman rank correlation:   {rho_spearman:.6f}")
    print(f"  → If 1.000: blend ranks horses identically to ML — even if")
    print(f"    values differ, picks at every quantile will be the same.")
    print(f"  → < 1.0: blend changes the ranking, picks should differ.\n")

    # 6. Same-pick rate per race — most operationally meaningful test.
    # For each race, does ensemble_prob's top-pick match ml_prob_norm's
    # top-pick? If yes, the blend never changes which horse you'd back.
    print(f"Per-race top-pick agreement:")
    if 'race' in df.columns and 'date' in df.columns and 'track' in df.columns:
        same_top = 0
        n_races = 0
        for _, race_df in df.groupby(['date', 'track', 'race']):
            valid = race_df.dropna(subset=['ensemble_prob', ml_col])
            if len(valid) < 2:
                continue
            n_races += 1
            top_e = valid.loc[valid['ensemble_prob'].idxmax(), 'horse_number']
            top_m = valid.loc[valid[ml_col].idxmax(), 'horse_number']
            if top_e == top_m:
                same_top += 1
        if n_races > 0:
            print(f"  {same_top}/{n_races} races ({same_top/n_races:.1%}) "
                  f"have the same top horse under both columns.")
            print(f"  → 100% means the blend never overrules ML for the favorite.")
            print(f"  → < 100% means the blend is doing real work somewhere.\n")
    else:
        print(f"  (Cannot compute — missing date/track/race columns)\n")

    # 7. A few example rows where they differ most, to eyeball
    nonzero_diff = abs_diff[abs_diff > 1e-9]
    if len(nonzero_diff) > 0:
        # Pick rows with the largest absolute differences
        top_diff_idx = nonzero_diff.nlargest(min(10, len(nonzero_diff))).index
        cols_show = ['date', 'track', 'race', 'horse_number',
                     'ensemble_prob', ml_col]
        cols_show = [c for c in cols_show if c in df.columns]
        print(f"Top {min(10, len(nonzero_diff))} rows by absolute diff:")
        print(df.loc[top_diff_idx, cols_show].to_string(index=False))
    else:
        print(f"No rows have a measurable diff (all within 1e-9). "
              f"Columns are effectively identical.")


if __name__ == '__main__':
    main()