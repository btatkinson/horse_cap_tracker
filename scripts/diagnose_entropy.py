"""
diagnose_entropy.py

Compare today's per-race entropy distribution against the backtest's
distribution that the v3 thresholds were calibrated on.

Why this exists: predict_v3 reported 11/12 races landing in the
LONGSHOT-zone for the 2026-04-30 Churchill card. By construction
(q40 = bottom 40%) only ~40% of *backtest* races should land there.
This script tells you whether today is genuinely a peaky-card outlier
OR whether the live ensemble has drifted relative to the backtest
sample (e.g. because more experts with higher weights are being
applied at race time than during walk_forward generation).

Usage:
    python diagnose_entropy.py --date 2026-04-30 --track churchill-downs

Outputs:
  1. Backtest entropy distribution (per-race), plus q40/q60 markers.
  2. Today's per-race entropy values, marked on the same axis.
  3. A summary of where today's races fall vs the backtest distribution.

If today's mean is far below the backtest mean and most races land
under q40, the calibration is stale — re-run backtest_walk_forward
+ backtest_kfold to refresh thresholds against current pipeline
output.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

DATA_PATH = Path(os.environ.get('DATA_PATH', '.'))


def compute_race_entropy(probs):
    """Same function as predict_v3.compute_race_entropy. Normalized H/log(N)."""
    arr = np.asarray(probs, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = np.clip(arr, 0.0, None)
    if len(arr) < 2:
        return np.nan
    total = arr.sum()
    if total <= 0:
        return np.nan
    p = arr / total
    p = p[p > 0]
    if len(p) < 2:
        return np.nan
    return -np.sum(p * np.log(p)) / np.log(len(p))


def load_backtest_entropies():
    """Load walk_forward_summary, compute per-race race_entropy_norm.
    Returns a Series of per-race entropy values (one row per race)."""
    wf_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
    if not wf_path.exists():
        print(f"ERROR: {wf_path} not found.")
        print("       Run backtest_walk_forward.py first.")
        sys.exit(1)

    df = pd.read_csv(wf_path)
    if 'ensemble_prob' not in df.columns:
        print(f"ERROR: walk_forward_summary.csv missing 'ensemble_prob' column.")
        sys.exit(1)

    keys = ['date', 'track', 'race']
    missing = [k for k in keys if k not in df.columns]
    if missing:
        print(f"ERROR: walk_forward_summary.csv missing columns: {missing}")
        sys.exit(1)

    ent = df.groupby(keys)['ensemble_prob'].apply(compute_race_entropy)
    ent = ent.dropna()
    return df, ent


def load_today_entropies(date_str, track):
    """Load today's predictions.csv (written by predict_v3 / predict_v2),
    compute per-race entropy. Returns dict {race: entropy}."""
    pred_path = DATA_PATH / 'predictions' / date_str / track / 'predictions.csv'
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found.")
        print(f"       Run: python predict_v3.py --date {date_str} --track {track}")
        sys.exit(1)
    df = pd.read_csv(pred_path)
    if 'ensemble_prob' not in df.columns:
        print(f"ERROR: today's predictions.csv missing 'ensemble_prob'.")
        sys.exit(1)
    out = {}
    for race, race_df in df.groupby('race'):
        out[int(race)] = compute_race_entropy(race_df['ensemble_prob'])
    return out


def load_thresholds():
    """Load v3_thresholds.json. Returns dict or None."""
    path = DATA_PATH / 'predictions' / 'v3_thresholds.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def render_histogram(values, n_bins=24, width=60, mark_at=None):
    """ASCII histogram. mark_at is a dict {label: value} drawn under the bars."""
    values = np.asarray(values)
    lo, hi = float(values.min()), float(values.max())
    if hi == lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    max_count = counts.max() if counts.max() > 0 else 1

    print(f"  range: [{lo:.3f}, {hi:.3f}]  n={len(values)}  "
          f"mean={values.mean():.3f}  median={np.median(values):.3f}  "
          f"std={values.std():.3f}")
    print()
    for i, c in enumerate(counts):
        bar_len = int(round(width * c / max_count))
        bar = '█' * bar_len
        print(f"  {edges[i]:.3f}–{edges[i+1]:.3f}  {bar} {c}")

    if mark_at:
        print()
        print(f"  {'value':>7}   marker")
        for label, val in sorted(mark_at.items(), key=lambda x: x[1]):
            # Find which bin
            bin_idx = min(int((val - lo) / (hi - lo) * n_bins), n_bins - 1)
            pos = int(round(width * bin_idx / n_bins))
            print(f"  {val:>7.3f}   {' ' * pos}↑ {label}")


def main():
    ap = argparse.ArgumentParser(
        description='Compare today\'s entropy distribution vs the backtest\'s')
    ap.add_argument('--date', required=True)
    ap.add_argument('--track', required=True)
    args = ap.parse_args()

    # --- Load backtest distribution ---
    print("=" * 78)
    print("BACKTEST entropy distribution (per-race race_entropy_norm)")
    print("Source: walk_forward_summary.csv → ensemble_prob → H/log(N)")
    print("=" * 78)
    wf_df, bt_ent = load_backtest_entropies()
    bt_q40 = float(bt_ent.quantile(0.40))
    bt_q60 = float(bt_ent.quantile(0.60))
    bt_q50 = float(bt_ent.quantile(0.50))
    render_histogram(bt_ent.values, mark_at={
        'q40': bt_q40, 'q50 (median)': bt_q50, 'q60': bt_q60,
    })
    print()
    print(f"  Backtest: q40={bt_q40:.4f}, q60={bt_q60:.4f}")

    # --- Compare with v3_thresholds.json ---
    th = load_thresholds()
    if th and 'thresholds' in th:
        file_q40 = th['thresholds'].get('race_entropy_norm_q40')
        file_q60 = th['thresholds'].get('race_entropy_norm_q60')
        if file_q40 is not None and file_q60 is not None:
            print(f"  v3_thresholds.json: q40={file_q40:.4f}, q60={file_q60:.4f}")
            drift_q40 = abs(file_q40 - bt_q40)
            drift_q60 = abs(file_q60 - bt_q60)
            if drift_q40 > 0.01 or drift_q60 > 0.01:
                print(f"  WARNING: thresholds in JSON drift from current "
                      f"walk_forward by Δq40={drift_q40:.4f} Δq60={drift_q60:.4f}")
                print(f"           (re-run backtest_kfold.py to refresh)")
            else:
                print(f"  ✓ JSON thresholds match current walk_forward distribution")
            n_races_in_file = th.get('n_races_in_sample')
            if n_races_in_file != len(bt_ent):
                print(f"  Note: JSON computed on n_races={n_races_in_file}, "
                      f"current walk_forward has {len(bt_ent)} races")
        else:
            print("  v3_thresholds.json missing race_entropy_norm_q40/q60 keys")
    else:
        print("  v3_thresholds.json not found or malformed")

    # --- Today's distribution ---
    print()
    print("=" * 78)
    print(f"TODAY'S races: {args.date} / {args.track}")
    print("=" * 78)
    today = load_today_entropies(args.date, args.track)
    today_vals = np.array(list(today.values()))

    print(f"  n={len(today_vals)}  mean={today_vals.mean():.3f}  "
          f"median={np.median(today_vals):.3f}  "
          f"min={today_vals.min():.3f}  max={today_vals.max():.3f}")
    print()
    use_q40 = file_q40 if (th and file_q40 is not None) else bt_q40
    use_q60 = file_q60 if (th and file_q60 is not None) else bt_q60

    print(f"  {'race':>4}  {'entropy':>7}  {'pct vs bt':>10}  zone")
    print(f"  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*15}")
    for race in sorted(today):
        ent = today[race]
        pct = (bt_ent < ent).mean() * 100
        if ent >= use_q60:
            zone = 'CHALK'
        elif ent <= use_q40:
            zone = 'LONGSHOT'
        else:
            zone = 'no-zone'
        print(f"  {race:>4}  {ent:>7.3f}  {pct:>9.1f}%  {zone}")

    # --- Verdict ---
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    n_lo = sum(1 for e in today_vals if e <= use_q40)
    n_hi = sum(1 for e in today_vals if e >= use_q60)
    n_mid = len(today_vals) - n_lo - n_hi
    pct_today_below_bt_median = (today_vals < bt_q50).mean() * 100
    today_median = float(np.median(today_vals))

    print(f"  Today: {n_lo} LONGSHOT-zone, {n_mid} no-zone, {n_hi} CHALK-zone "
          f"(out of {len(today_vals)})")
    print(f"  Today's median entropy: {today_median:.3f}")
    print(f"  Backtest median entropy: {bt_q50:.3f}")
    print(f"  Today's median falls at the {(bt_ent < today_median).mean()*100:.0f}th "
          f"percentile of the backtest distribution.")
    print()

    if pct_today_below_bt_median > 75:
        print("  >>> Today's card is meaningfully PEAKIER than the typical")
        print("      backtest race. Possible causes (most → least likely):")
        print()
        print("      1. Stronger expert weighting now than at walk_forward time.")
        print("         Today has lots of high-rated experts (1600+) shifting")
        print("         logits aggressively. If walk_forward_summary.csv was")
        print("         built before those ratings stabilized, the historical")
        print("         entropy distribution is inflated relative to today's.")
        print("         FIX: re-run backtest_walk_forward.py to regenerate")
        print("         walk_forward_summary with current expert weights, then")
        print("         re-run backtest_kfold.py to refresh thresholds.")
        print()
        print("      2. Genuinely peaky day (small fields, clear favorites).")
        print("         No fix needed — the gate is correctly identifying that")
        print("         most races today have a strong model favorite.")
        print()
        print("      3. Bug in expert adjustment renormalization (less likely")
        print("         given the live and historical paths use the same code).")
    elif n_lo + n_hi == 0:
        print("  >>> Today's card sits entirely in the mid-band (no-zone). No v3")
        print("      bets will fire today regardless of model triggers.")
    else:
        print("  >>> Today's distribution looks roughly typical relative to the")
        print(f"      backtest. {n_lo}/{len(today_vals)} below q40, "
              f"{n_hi}/{len(today_vals)} above q60.")


if __name__ == '__main__':
    main()