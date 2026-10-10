"""
backtest_v4_corner.py

Backtest predict_v4_corner's rule against historical out-of-sample
predictions in walk_forward_summary.csv.

The rule (mirrors predict_v4_corner.py):
    1. Per-race normalize raw_value (raw_value / race_mean), since
       predict_v2 doesn't do this and pooled z-scoring it as-is would
       systematically favor heavy favorites.
    2. Z-score the three values (ev_per_2, ridge_value, raw_value_pernorm)
       using pooled mean/std from the calibration sample.
    3. Compute corner_score = (z1+z2+z3) * sqrt(z1²+z2²+z3²) per horse.
    4. Bet flat $2 (win only) on horses whose score >= the
       (1 - bet_fraction) quantile of historical scores.

Why per-race normalize raw_value
--------------------------------
Heavy favorites tend to dominate the corner score at restrictive bet
fractions if you z-score raw_value directly. The reason: predict_v2's
raw-ridge model is NOT per-race normalized — its absolute level is
governed by input prob features, so high-probability favorites have
systematically higher raw_value than longshots regardless of price.
Pooled z-scoring then surfaces favorites at the top.

Per-race normalization (divide by race mean) puts each race's raw_value
on a comparable [around-1.0] scale before z-scoring. ridge_value is
already per-race normalized by predict_v2 (sums to RIDGE_RACE_TOTAL),
and ev_per_2 is value-based by construction (break-even = $0), so this
fix targets only the misbehaving column.

What this script reports:
    - Overall ROI, hit rate, avg odds on bets fired, vs. random-bettor
      baseline of -takeout per dollar.
    - Sweep over restrictive bet_fraction values (0.05–0.33) so you
      can see how rule sensitivity changes the answer in the regime
      where this rule is most likely to work.
    - Always uses k-fold cross-validation by default (5 folds). Each
      fold calibrates from out-of-fold dates and tests on in-fold
      dates. --in-sample disables this and uses one global calibration.
    - Sweep table shows: n_bets across all folds, hit rate, ROI,
      vs-random edge, the prospective full-data cutoff, and the
      min-max cutoff range across folds (wide range = unstable).

Limitations
-----------
- Uses win bets only ($2 flat). Does not simulate v2's tiered W/WP/WPS
  bet-type selection. This isolates the v4 selection rule from the
  bet-sizing rule.
- Reads payout / won columns from entries via load_cv_predictions's
  existing merge logic. Races where this merge fails are dropped
  upstream by filter_broken_races.
- K-fold mode (default) gives an honest out-of-sample estimate but
  bet counts depend on per-fold cutoffs which vary slightly. The
  "fold_range" column in --sweep output shows that variation.

Usage
-----
    python backtest_v4_corner.py
    python backtest_v4_corner.py --bet-fraction 0.10
    python backtest_v4_corner.py --sweep
    python backtest_v4_corner.py --sweep --folds 10
    python backtest_v4_corner.py --in-sample --sweep    # opt out of CV
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])

# Reuse the existing data-loading + entries-merge logic. backtest_kfold's
# load_cv_predictions handles auto-detection between walk_forward_summary
# and regression_predictions, merges in payout / won / ml_decimal from
# entries when missing, and filters broken races.
from backtest_kfold import load_cv_predictions, WIN_TAKEOUT


# =============================================================================
# CONFIG
# =============================================================================

VALUE_COLS = ['ev_per_2', 'ridge_value', 'raw_value_pernorm']
FLAT_BET = 2.0

# Default sweep values for --sweep mode. Zoomed on the lower range
# where restrictive cutoffs tend to live; finer granularity in the
# 0.05–0.34 band to spot inflection points.
DEFAULT_SWEEP = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.33]


# =============================================================================
# PER-RACE NORMALIZATION
# =============================================================================
# raw_value (raw-ridge model) is NOT per-race normalized by predict_v2 —
# its scale across races is governed by the input prob features, so heavy
# favorites get systematically higher raw_value than longshots regardless
# of price. Pooled z-scoring then surfaces favorites at the top of the
# corner score.
#
# ridge_value IS per-race normalized by predict_v2 (sums to RIDGE_RACE_TOTAL
# per race), so each horse's value is already on a comparable scale across
# races. ev_per_2 is value-based by construction (break-even = $0).
#
# To put raw_value on the same footing, divide each horse's raw_value by
# the per-race mean of raw_value. Result: each race has mean 1.0, so the
# score reflects "above/below this race's average" rather than absolute
# raw_value level. Then z-score this normalized version pooled across races.

def add_per_race_normalized_raw(df):
    """Add raw_value_pernorm column = raw_value / race_mean(raw_value).

    Groups by (date, track, race) — the canonical race identifier in
    walk_forward_summary. Races with mean ≤ 0 (degenerate, shouldn't
    happen) get NaN to avoid division blow-ups.
    """
    if 'raw_value' not in df.columns:
        raise RuntimeError("Cannot per-race normalize: raw_value column missing")
    group_cols = ['date', 'track', 'race']
    race_mean = df.groupby(group_cols)['raw_value'].transform('mean')
    df = df.copy()
    df['raw_value_pernorm'] = np.where(race_mean > 0,
                                        df['raw_value'] / race_mean,
                                        np.nan)
    return df


# =============================================================================
# SCORING (mirrors predict_v4_corner._compute_corner_score)
# =============================================================================

def compute_corner_score(z_array):
    """dot([z1,z2,z3], [1,1,1]) * sqrt(z1²+z2²+z3²) per row.

    Sign-preserving — negative when most z's are negative. The all-positive
    octant naturally produces the highest scores without an explicit filter.
    """
    dot = z_array.sum(axis=1)
    mag = np.sqrt((z_array ** 2).sum(axis=1))
    return dot * mag


def add_z_and_score(df, means, stds):
    """Add z_log, z_ridge, z_raw, corner_score columns to df.

    means, stds: pd.Series indexed by VALUE_COLS. Note that the third
    column is raw_value_pernorm (per-race normalized), not raw_value —
    see add_per_race_normalized_raw above for why.

    Returns: df with the four new columns.
    """
    df = df.copy()
    df['z_log']   = (df['ev_per_2']           - means['ev_per_2'])           / stds['ev_per_2']
    df['z_ridge'] = (df['ridge_value']        - means['ridge_value'])        / stds['ridge_value']
    df['z_raw']   = (df['raw_value_pernorm']  - means['raw_value_pernorm'])  / stds['raw_value_pernorm']
    z_arr = df[['z_log', 'z_ridge', 'z_raw']].values
    df['corner_score'] = compute_corner_score(z_arr)
    return df


def calibrate(df):
    """Compute pooled mean/std and return (means, stds, scores).

    `scores` is the array of corner scores under this calibration —
    used downstream to compute the percentile cutoff.

    Drops rows with NaN in any value column from the calibration.
    """
    sample = df[VALUE_COLS].dropna()
    means = sample.mean()
    stds = sample.std()
    z = ((sample - means) / stds).values
    scores = compute_corner_score(z)
    return means, stds, scores


# =============================================================================
# BET SIMULATION
# =============================================================================

def simulate_bets(scored_df, cutoff):
    """Apply the rule to scored_df and compute per-bet outcomes.

    Args:
        scored_df: DataFrame with corner_score, won, win_paid_num populated.
        cutoff: scalar threshold; horses with corner_score >= cutoff are bet.

    Returns:
        bets_df: rows = bets fired, with bet_amount, payout, profit columns.
                 Empty DataFrame if no bets qualify.
    """
    qualifies = (scored_df['corner_score'].notna()) & \
                (scored_df['corner_score'] >= cutoff)
    bets = scored_df[qualifies].copy()
    if bets.empty:
        return bets

    # Flat $2 win bet. Payout = win_paid_num if won else 0. The win_paid_num
    # column from entries is the $2-bet payout (e.g. $5.40 for a 3/2 winner),
    # so it's directly comparable to FLAT_BET=2.
    bets['bet_amount'] = FLAT_BET
    bets['payout'] = np.where(bets['won'] == 1,
                              bets['win_paid_num'].fillna(0.0),
                              0.0)
    bets['profit'] = bets['payout'] - bets['bet_amount']
    return bets


def summarize(bets_df, label, takeout=WIN_TAKEOUT):
    """Compute headline metrics and return as a dict."""
    if bets_df.empty:
        return {
            'label': label,
            'n_bets': 0,
            'n_wins': 0,
            'hit_rate': float('nan'),
            'total_bet': 0.0,
            'total_payout': 0.0,
            'total_profit': 0.0,
            'roi_pct': float('nan'),
            'edge_vs_random_pp': float('nan'),
            'avg_winning_payout': float('nan'),
            'avg_z_log': float('nan'),
            'avg_z_ridge': float('nan'),
            'avg_z_raw': float('nan'),
            'avg_ml_decimal': float('nan'),
        }

    n_bets = len(bets_df)
    n_wins = int(bets_df['won'].sum())
    total_bet = float(bets_df['bet_amount'].sum())
    total_payout = float(bets_df['payout'].sum())
    total_profit = float(bets_df['profit'].sum())
    roi_pct = (total_profit / total_bet) * 100.0
    avg_win = float(bets_df.loc[bets_df['won'] == 1, 'payout'].mean()) \
              if n_wins else float('nan')

    # A random bettor at the same takeout loses takeout × 100 percent.
    edge_vs_random_pp = roi_pct - (-takeout * 100.0)

    # Per-axis z-score means on bets — diagnostic for which axis the rule
    # is leaning on. Symmetric rule with independent equal-variance axes
    # would have means roughly equal. Skewed means = one axis carries.
    avg_z_log   = float(bets_df['z_log'].mean())   if 'z_log'   in bets_df.columns else float('nan')
    avg_z_ridge = float(bets_df['z_ridge'].mean()) if 'z_ridge' in bets_df.columns else float('nan')
    avg_z_raw   = float(bets_df['z_raw'].mean())   if 'z_raw'   in bets_df.columns else float('nan')

    # Avg ML decimal — diagnostic for "are these favorites or longshots?"
    # A 4/5 favorite has ml_decimal=1.8; a 5/1 horse has 6.0; a 20/1 horse 21.0.
    if 'ml_decimal' in bets_df.columns:
        ml_vals = bets_df['ml_decimal'].dropna()
        avg_ml = float(ml_vals.mean()) if len(ml_vals) else float('nan')
    else:
        avg_ml = float('nan')

    return {
        'label': label,
        'n_bets': n_bets,
        'n_wins': n_wins,
        'hit_rate': n_wins / n_bets,
        'total_bet': total_bet,
        'total_payout': total_payout,
        'total_profit': total_profit,
        'roi_pct': roi_pct,
        'edge_vs_random_pp': edge_vs_random_pp,
        'avg_winning_payout': avg_win,
        'avg_z_log': avg_z_log,
        'avg_z_ridge': avg_z_ridge,
        'avg_z_raw': avg_z_raw,
        'avg_ml_decimal': avg_ml,
    }


# =============================================================================
# IN-SAMPLE BACKTEST
# =============================================================================

def backtest_in_sample(df, bet_fraction):
    """Single-shot in-sample evaluation at a given bet_fraction.

    Calibrate z-score stats and percentile cutoff on the full df,
    then bet against the same df. Slightly optimistic (small leakage
    from using same sample for cutoff and test) but matches what
    predict_v4_corner does prospectively.
    """
    means, stds, scores = calibrate(df)
    cutoff = float(np.quantile(scores, 1.0 - bet_fraction))
    scored = add_z_and_score(df, means, stds)
    bets = simulate_bets(scored, cutoff)
    metrics = summarize(bets, label=f'in-sample @ bf={bet_fraction:.2f}')
    metrics['cutoff'] = cutoff
    metrics['n_octant_bets'] = int(((bets[['z_log','z_ridge','z_raw']] > 0)
                                    .all(axis=1)).sum()) if not bets.empty else 0
    return metrics, bets


# =============================================================================
# K-FOLD BACKTEST (out-of-sample)
# =============================================================================

def backtest_kfold(df, bet_fraction, n_folds=5):
    """Temporal k-fold. For each fold:
        - calibrate (means, stds, cutoff) from rows in OTHER folds
        - bet on rows in THIS fold using that calibration
    Concatenate all in-fold bets and report aggregate metrics.

    Folds are assigned by date ordering so the calibration always comes
    from a different time slice than the test bets.

    Also computes (separately) the cutoff that would be used PROSPECTIVELY
    if all available data were used as the calibration sample. This is
    the number you'd actually plug into predict_v4_corner's threshold
    file. It's not used for any of the bet decisions in this backtest —
    it's reported alongside the per-fold range so you can see how stable
    the cutoff is.
    """
    if 'date' not in df.columns:
        raise RuntimeError("k-fold requires a 'date' column to slice by.")

    dates = sorted(df['date'].unique())
    if len(dates) < n_folds:
        raise RuntimeError(
            f"Only {len(dates)} unique dates in sample — can't do "
            f"{n_folds}-fold. Reduce --folds or run more dates."
        )

    # Compute the prospective cutoff (full-data calibration). This is what
    # predict_v4_corner would persist in v4_thresholds.json if you ran it
    # against this same walk_forward sample.
    _, _, all_scores = calibrate(df)
    prospective_cutoff = float(np.quantile(all_scores, 1.0 - bet_fraction))

    # Split dates into n_folds roughly equal chunks (temporal order preserved).
    fold_size = len(dates) // n_folds
    fold_dates = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(dates)
        fold_dates.append(set(dates[start:end]))

    all_bets = []
    fold_metrics = []
    fold_cutoffs = []
    for i, fold in enumerate(fold_dates):
        in_fold = df[df['date'].isin(fold)]
        out_fold = df[~df['date'].isin(fold)]
        if out_fold.empty or in_fold.empty:
            print(f"  Fold {i+1}: skipped (empty in/out partition)")
            continue
        means, stds, scores = calibrate(out_fold)
        cutoff = float(np.quantile(scores, 1.0 - bet_fraction))
        fold_cutoffs.append(cutoff)
        scored = add_z_and_score(in_fold, means, stds)
        bets = simulate_bets(scored, cutoff)
        m = summarize(bets, label=f'fold {i+1}')
        m['cutoff'] = cutoff
        m['n_dates_in_fold'] = len(fold)
        m['n_dates_out_fold'] = len(dates) - len(fold)
        fold_metrics.append(m)
        if not bets.empty:
            all_bets.append(bets)

    pooled_bets = pd.concat(all_bets) if all_bets else pd.DataFrame()
    pooled = summarize(pooled_bets, label=f'k-fold pooled @ bf={bet_fraction:.2f}')
    pooled['n_folds'] = n_folds
    pooled['cutoff_prospective'] = prospective_cutoff  # full-data cutoff
    pooled['cutoff_fold_min'] = min(fold_cutoffs) if fold_cutoffs else float('nan')
    pooled['cutoff_fold_max'] = max(fold_cutoffs) if fold_cutoffs else float('nan')
    pooled['cutoff_fold_median'] = float(np.median(fold_cutoffs)) if fold_cutoffs else float('nan')
    return pooled, fold_metrics, pooled_bets


# =============================================================================
# REPORT
# =============================================================================

def print_metrics_row(m, takeout=WIN_TAKEOUT):
    """Single-line metrics row for sweep / k-fold reports."""
    if m['n_bets'] == 0:
        print(f"  {m['label']:30s}  no bets fired")
        return
    print(f"  {m['label']:30s}  "
          f"n={m['n_bets']:>4d}  "
          f"wins={m['n_wins']:>3d}  "
          f"hit={m['hit_rate']*100:>5.1f}%  "
          f"ROI={m['roi_pct']:>+6.2f}%  "
          f"vs-random={m['edge_vs_random_pp']:>+6.2f}pp  "
          f"avg-win=${m['avg_winning_payout']:.2f}")


def print_full_report(metrics, takeout=WIN_TAKEOUT, bets_df=None):
    """Detailed report for a single backtest result."""
    print(f"\n{'='*72}")
    print(f"  {metrics['label']}")
    print(f"{'='*72}")
    if metrics['n_bets'] == 0:
        print("  Zero bets fired. Either the sample is empty or the cutoff is")
        print("  set too high for any horse to clear. Lower --bet-fraction or")
        print("  check that walk_forward_summary.csv has the three value cols.")
        return
    print(f"  Bets:                {metrics['n_bets']:,d}")
    print(f"  Wins:                {metrics['n_wins']:,d}  "
          f"({metrics['hit_rate']*100:.1f}%)")
    print(f"  Total bet:           ${metrics['total_bet']:,.2f}")
    print(f"  Total payout:        ${metrics['total_payout']:,.2f}")
    print(f"  Total profit:        ${metrics['total_profit']:+,.2f}")
    print(f"  ROI:                 {metrics['roi_pct']:+.2f}%")
    print(f"  Random-bettor ROI:   {-takeout*100:.2f}%  "
          f"(loses takeout per $1)")
    print(f"  Edge vs random:      {metrics['edge_vs_random_pp']:+.2f} pp")
    print(f"  Avg winning payout:  ${metrics['avg_winning_payout']:.2f}")
    if 'cutoff' in metrics:
        print(f"  Score cutoff used:   {metrics['cutoff']:.3f}")
    if 'n_octant_bets' in metrics:
        pct = metrics['n_octant_bets'] / metrics['n_bets'] * 100
        print(f"  Bets in +octant:     {metrics['n_octant_bets']:,d}  "
              f"({pct:.1f}% — informational, not a rule constraint)")

    # Profitability sanity: are we net positive after takeout?
    if metrics['edge_vs_random_pp'] > 0:
        print(f"\n  ✓ Beats the random-bettor baseline by "
              f"{metrics['edge_vs_random_pp']:.2f} pp.")
        if metrics['roi_pct'] > 0:
            print(f"  ✓ Net positive ROI — beats takeout entirely.")
        else:
            print(f"  ✗ Still net negative ROI ({metrics['roi_pct']:+.2f}%) — "
                  f"cuts into takeout but doesn't overcome it.")
    else:
        print(f"\n  ✗ Underperforms the random-bettor baseline. "
              f"This rule, at this bet_fraction, doesn't work on this sample.")


def print_sweep_report(sweep_results, takeout=WIN_TAKEOUT, mode_label='k-fold'):
    """Compact table comparing bet_fraction values."""
    print(f"\n{'='*120}")
    print(f"  Sweep over bet_fraction ({mode_label})")
    print(f"{'='*120}")
    # Two-row header to fit the wider table.
    print(f"  {'bet_frac':>8s}  {'n_bets':>6s}  {'hit%':>5s}  "
          f"{'ROI%':>6s}  {'vs-rnd':>7s}  "
          f"{'z̄_log':>6s}  {'z̄_rdg':>6s}  {'z̄_raw':>6s}  "
          f"{'avg_ml':>6s}  {'cutoff':>7s}  {'fold_range':>15s}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*7}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*15}")
    for bf, m in sweep_results:
        prosp = m.get('cutoff_prospective', m.get('cutoff', float('nan')))
        fmin = m.get('cutoff_fold_min', float('nan'))
        fmax = m.get('cutoff_fold_max', float('nan'))
        if not np.isnan(fmin):
            range_str = f"{fmin:.2f}-{fmax:.2f}"
        else:
            range_str = "─"
        if m['n_bets'] == 0:
            print(f"  {bf:>8.3f}  {'0':>6s}  ─      ─       ─        "
                  f"─       ─       ─       ─       "
                  f"{prosp:>7.3f}  {range_str:>15s}")
            continue
        # Helper to format possibly-NaN floats compactly
        def fmt(x, fmt_str='{:>+6.2f}'):
            return fmt_str.format(x) if not np.isnan(x) else '   ─  '
        ml_str = f"{m['avg_ml_decimal']:6.2f}" if not np.isnan(m['avg_ml_decimal']) else '   ─  '
        print(f"  {bf:>8.3f}  "
              f"{m['n_bets']:>6d}  "
              f"{m['hit_rate']*100:>5.1f}  "
              f"{m['roi_pct']:>+6.2f}  "
              f"{m['edge_vs_random_pp']:>+7.2f}  "
              f"{fmt(m['avg_z_log'])}  "
              f"{fmt(m['avg_z_ridge'])}  "
              f"{fmt(m['avg_z_raw'])}  "
              f"{ml_str:>6s}  "
              f"{prosp:>7.3f}  "
              f"{range_str:>15s}")
    print(f"\n  Random-bettor baseline ROI: {-takeout*100:.2f}%  "
          f"(any vs-rnd > 0 = beating the house edge by that many pp)")
    print(f"\n  z̄_log / z̄_rdg / z̄_raw: mean z-score on each axis, averaged")
    print(f"    across bets. Symmetric rule with independent axes would")
    print(f"    have all three roughly equal. One axis dominating means")
    print(f"    that model is pulling more weight on bet selection.")
    print(f"  avg_ml: mean ML decimal odds of bets. 1.8=4/5 favorite, "
          f"3.0=2/1, 5.0=4/1, 11.0=10/1. Low values = mostly favorites.")
    if mode_label.startswith('k-fold'):
        print(f"  cutoff column = full-data calibration (what you'd use prospectively).")
        print(f"  fold_range = min-max cutoff across folds; wide range = unstable cutoff.")
        print(f"  n_bets = total bets across all folds combined.")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Backtest predict_v4_corner against historical predictions. '
                    'Always uses k-fold cross-validation by default.'
    )
    parser.add_argument('--bet-fraction', type=float, default=0.10,
                        help='Top fraction of historical scores to bet. '
                             'Default 0.10 (top 10%%). Use --sweep to '
                             'compare multiple values.')
    parser.add_argument('--sweep', action='store_true',
                        help=f'Sweep over bet_fraction values '
                             f'({DEFAULT_SWEEP}) and print a comparison table.')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of temporal cross-validation folds. '
                             'Default 5.')
    parser.add_argument('--in-sample', action='store_true',
                        help='Use in-sample calibration (no cross-validation). '
                             'Faster but slightly optimistic. Default is k-fold.')
    parser.add_argument('--takeout', type=float, default=WIN_TAKEOUT,
                        help=f'Track takeout for the random-bettor baseline. '
                             f'Default {WIN_TAKEOUT}.')
    parser.add_argument('--source', default=None, choices=[None, 'walk_forward', 'cv'],
                        help='Which prediction file to load. Default auto-detect '
                             '(prefers walk_forward_summary.csv).')
    args = parser.parse_args()

    if not (0.0 < args.bet_fraction < 1.0):
        raise SystemExit(f"--bet-fraction must be in (0,1), got {args.bet_fraction}")
    if args.folds < 2:
        raise SystemExit(f"--folds must be >= 2, got {args.folds}")

    df = load_cv_predictions(source=args.source)

    # Validate the three value columns exist (load_cv_predictions normalizes
    # 'raw_value' → 'raw_ridge_value' for the v3 path, but also keeps the
    # original; we want the predict_v2 names ev_per_2/ridge_value/raw_value).
    if 'raw_value' not in df.columns and 'raw_ridge_value' in df.columns:
        df['raw_value'] = df['raw_ridge_value']
    raw_input_cols = ['ev_per_2', 'ridge_value', 'raw_value']
    missing = [c for c in raw_input_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Required value columns missing from prediction source: {missing}. "
            f"Available cols include: {sorted(df.columns)[:20]}..."
        )

    # Drop rows missing essentials for backtest (won, win_paid_num).
    n0 = len(df)
    df = df.dropna(subset=['won', 'win_paid_num'])
    if len(df) < n0:
        print(f"  Dropped {n0 - len(df):,d} rows missing won/win_paid_num "
              f"({len(df):,d} remaining).")

    # Per-race normalize raw_value (predict_v2 doesn't do this; ridge_value
    # is already per-race normalized; ev_per_2 is value-based by construction).
    # Result is raw_value_pernorm column, used downstream as the third axis.
    df = add_per_race_normalized_raw(df)
    n_pernorm_nan = df['raw_value_pernorm'].isna().sum()
    if n_pernorm_nan:
        print(f"  Note: {n_pernorm_nan:,d} rows have NaN raw_value_pernorm "
              f"(degenerate races with mean ≤ 0); excluded from scoring.")

    if args.in_sample:
        # In-sample mode (single calibration on all data).
        if args.sweep:
            print(f"\nIn-sample sweep over bet_fraction:")
            sweep_results = []
            for bf in DEFAULT_SWEEP:
                metrics, _ = backtest_in_sample(df, bf)
                sweep_results.append((bf, metrics))
            print_sweep_report(sweep_results, takeout=args.takeout,
                               mode_label='in-sample')
        else:
            metrics, bets = backtest_in_sample(df, args.bet_fraction)
            print_full_report(metrics, takeout=args.takeout, bets_df=bets)
    else:
        # K-fold mode (out-of-sample). The default.
        if args.sweep:
            print(f"\nK-fold sweep over bet_fraction (n_folds={args.folds}):")
            sweep_results = []
            for bf in DEFAULT_SWEEP:
                pooled, _, _ = backtest_kfold(df, bf, n_folds=args.folds)
                sweep_results.append((bf, pooled))
            print_sweep_report(sweep_results, takeout=args.takeout,
                               mode_label=f'k-fold n={args.folds}')
        else:
            pooled, fold_metrics, _ = backtest_kfold(
                df, args.bet_fraction, n_folds=args.folds
            )
            print(f"\nPer-fold results (out-of-sample, n_folds={args.folds}, "
                  f"bf={args.bet_fraction:.2f}):")
            for m in fold_metrics:
                print_metrics_row(m, takeout=args.takeout)
            print_full_report(pooled, takeout=args.takeout)


if __name__ == '__main__':
    main()