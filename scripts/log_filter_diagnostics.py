"""
prob_calibration_diagnostic.py

Three-way calibration comparison for win probability estimators:

  1. ensemble_prob       — the blended logistic prob used in v2/v3 hit-prob
  2. ridge_value / 18.3  — ridge interpretation B from Diagnostic 3
                           (already shown to calibrate at 4.2% mean abs err)
  3. ml_prob = 1/ml_decimal — the morning-line baseline

For each: bin horses into deciles by predicted prob, compute actual
win rate per decile. The estimator with lowest mean absolute calibration
error is the one v3 should use for hit-prob estimation.

Why this matters: I previously recommended switching v3 from
ensemble_prob to ridge_prob based only on ridge calibrating well. But
'A calibrates well' doesn't imply 'A calibrates better than B' if B
also calibrates well. This script tests the comparison directly.

Notes on column naming:
  - backtest_kfold maps walk_forward's 'ensemble_prob' to 'cv_prob'
    internally. Diagnostic 1 used 'cv_prob' for the LOG edge ranking.
    For calibration we want whatever the OPTIMIZER uses at prediction
    time — which is 'ensemble_prob' from predictions.csv. We test
    both names if both exist; if they differ that's its own finding.

Usage:
    python prob_calibration_diagnostic.py
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from backtest_kfold import (
    load_cv_predictions,
    compute_ev_per_2,
    apply_ridge_race_norm,
    compute_raw_calibration,
    compute_bet_flags,
    RIDGE_RACE_TOTAL,
    RIDGE_FLOOR,
    WIN_TAKEOUT,
)


def evaluate_calibration(df, prob_col, label, n_bins=10):
    """
    Bin horses into n_bins deciles by prob_col, compute actual win rate
    per bin, return (per-bin table, mean abs calibration error, n_used).

    Uses pd.qcut with duplicates='drop' which can collapse bins when
    many rows share the same predicted prob. We report how many bins
    we actually got — a model whose predicted probs are heavily
    quantized (e.g., always 0.10 for many horses) will produce fewer
    bins, which itself is a finding.
    """
    valid = df.dropna(subset=[prob_col, 'won']).copy()
    valid = valid[(valid[prob_col] > 0) & (valid[prob_col] < 1)]
    if len(valid) == 0:
        return None, None, 0

    # qcut into deciles, dropping duplicate edges
    try:
        valid['bin'] = pd.qcut(valid[prob_col], q=n_bins, labels=False,
                               duplicates='drop')
    except ValueError:
        # All identical values
        return None, None, len(valid)

    bins_used = valid['bin'].nunique()
    rows = []
    total_weighted_err = 0.0
    total_n = 0
    for b in sorted(valid['bin'].dropna().unique()):
        sub = valid[valid['bin'] == b]
        n = len(sub)
        pred = sub[prob_col].mean()
        actual = sub['won'].mean()
        diff = actual - pred
        rows.append({'bin': int(b) + 1, 'n': n, 'pred': pred,
                     'actual': actual, 'diff': diff})
        total_weighted_err += abs(diff) * n
        total_n += n

    mae = total_weighted_err / total_n if total_n > 0 else None
    return rows, mae, len(valid)


def print_calibration_table(label, rows, mae, n_used, n_bins_target=10):
    print()
    print(f"  {label}")
    if rows is None:
        print(f"    No valid data.")
        return
    bins_actual = len(rows)
    if bins_actual < n_bins_target:
        print(f"    NOTE: only {bins_actual} bins (target {n_bins_target}). "
              f"Predicted probs are quantized — many horses share identical values.")
    print(f"    {'bin':>4} {'n':>5} {'pred':>7} {'actual':>7} {'diff':>7}")
    print(f"    {'-'*4} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    for r in rows:
        print(f"    {r['bin']:>4} {r['n']:>5} "
              f"{r['pred']:>6.1%} {r['actual']:>6.1%} {r['diff']:>+6.1%}")
    print(f"    → Mean abs calibration error: {mae:.1%}  (n={n_used})")


def main():
    print("Loading predictions (same pipeline as other diagnostics)...")
    df = load_cv_predictions(source=None)
    df = compute_ev_per_2(df, takeout=WIN_TAKEOUT)

    if 'ridge_value' in df.columns:
        df = apply_ridge_race_norm(df, 'ridge_value', RIDGE_RACE_TOTAL, RIDGE_FLOOR)

    raw_mean, raw_std = compute_raw_calibration(df, 'raw_ridge_value')
    df = compute_bet_flags(df, raw_mean, raw_std)

    # Build the three probability estimators
    df = df.copy()

    # ml_prob from ml_decimal
    if 'ml_decimal' in df.columns:
        df['ml_prob'] = 1.0 / df['ml_decimal']
    elif 'ml_win_prob_norm' in df.columns:
        df['ml_prob'] = df['ml_win_prob_norm']
    else:
        df['ml_prob'] = np.nan

    # ridge-implied prob (interpretation B from Diag 3)
    if 'ridge_value' in df.columns:
        df['ridge_prob'] = df['ridge_value'] / RIDGE_RACE_TOTAL
    else:
        df['ridge_prob'] = np.nan

    # ensemble_prob: try a few candidate column names since the column
    # is renamed across the pipeline
    ens_col = None
    for cand in ['ensemble_prob', 'cv_prob', 'cv_prob_logistic']:
        if cand in df.columns and df[cand].notna().any():
            ens_col = cand
            break

    print()
    print("="*78)
    print("CALIBRATION COMPARISON — three probability estimators")
    print("="*78)
    print("  Bin horses into 10 deciles by predicted prob, compute actual")
    print("  win rate per decile. Lower mean abs calibration error = better.")
    print()

    estimators = []
    if ens_col is not None:
        print(f"  Using '{ens_col}' for ensemble_prob calibration.")
        estimators.append(('ensemble_prob (blended logistic)', ens_col))
    else:
        print(f"  WARNING: no ensemble_prob/cv_prob column found. Skipping that estimator.")

    estimators.append(('ridge_prob (ridge_value / 18.3)', 'ridge_prob'))
    estimators.append(('ml_prob (1 / ml_decimal)', 'ml_prob'))

    summary = []
    for label, col in estimators:
        rows, mae, n_used = evaluate_calibration(df, col, label, n_bins=10)
        print_calibration_table(label, rows, mae, n_used)
        if mae is not None:
            summary.append((label, mae, n_used))

    print()
    print("="*78)
    print("SUMMARY")
    print("="*78)
    print(f"  {'estimator':<40} {'MAE':>7} {'n':>5}")
    print(f"  {'-'*40} {'-'*7} {'-'*5}")
    for label, mae, n in sorted(summary, key=lambda x: x[1]):
        print(f"  {label:<40} {mae:>6.1%} {n:>5}")

    print()
    print("INTERPRETATION")
    print("─"*78)
    print("""
  Lowest MAE = use this for hit-prob estimation in v3 / future optimizers.

  Things to watch for in the table above:
    - If ml_prob's MAE is low, the morning line is shockingly accurate.
      That doesn't kill your models — they still need to BEAT ml_prob,
      not just match it — but it's a humbling baseline.
    - If ensemble_prob's MAE is much higher than ridge_prob's, that's
      direct support for switching v3 to ridge_prob.
    - If ensemble_prob's MAE is similar to ridge_prob's, both are usable.
      In that case the better choice depends on other factors:
        * Which has more data points (predictions per card)?
        * Which is computationally cheaper?
        * Which was used in the calibration sample? (a model fit on
          THIS data may overfit by definition — out-of-sample is harder)
    - If a column has fewer than 10 bins, its probs are quantized.
      Common cause: ML decimals only take ~20 distinct values, so
      ml_prob will have ~20 buckets. Not a problem for calibration
      check, just noted for context.

  The decile that consistently miscalibrates ACROSS estimators is also
  informative: usually D10 (top decile) over-predicts because true win
  prob caps around 35% even for huge favorites in horse racing. If
  multiple estimators all over-predict at D10, capping at ~0.35 is a
  general fix not specific to ridge.
""")


if __name__ == '__main__':
    main()