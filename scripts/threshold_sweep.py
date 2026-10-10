"""
threshold_sweep.py

For each of the three bet triggers (logistic, ridge, raw-ridge), sort
historical bets by edge strength and show cumulative ROI as the threshold
tightens. Answers: "where does the signal actually live?"

Usage:
    python threshold_sweep.py                  # all three triggers
    python threshold_sweep.py --trigger raw    # just raw-ridge

Reads:
    {DATA_PATH}/predictions/walk_forward_summary.csv

This is a READ-ONLY diagnostic — it doesn't modify predict_v2 thresholds.
Use the output to decide what to change, then update the constants in
predict_v2.py manually and re-run the backtest.
"""

import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
SUMMARY_PATH = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'

WIN_TAKEOUT = 0.17

# Current production thresholds (for reference in output)
CURRENT_THRESHOLDS = {
    'logistic': 0.05,    # LOGISTIC_EDGE_THRESHOLD: $ per $2 EV
    'ridge_mult': 1.05,  # RIDGE_EDGE_MULTIPLIER: × per-race break-even
    'raw_k': 1.0,        # RAW_EDGE_K: σ above cv_mean
}


def _filter_valid_races(df):
    """Drop pending races + broken races (same as backtest_kfold)."""
    if 'won' not in df.columns:
        return df

    winners_per_race = df.groupby(['date', 'track', 'race'])['won'].sum()
    good = winners_per_race[winners_per_race == 1].index

    bad = winners_per_race[winners_per_race != 1]
    if len(bad) == 0:
        return df

    race_key = df.set_index(['date', 'track', 'race']).index
    return df[race_key.isin(good)].copy()


def _load():
    """Load walk-forward summary, attach won column, filter broken races."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Run backtest_walk_forward.py first.")

    df = pd.read_csv(SUMMARY_PATH)
    # If 'won' isn't there, we can't do anything
    if 'won' not in df.columns:
        raise ValueError(
            "walk_forward_summary.csv has no 'won' column. Check that "
            "backtest_walk_forward merges entries results.")

    before = len(df)
    df = _filter_valid_races(df)
    after = len(df)
    if before != after:
        print(f"Filtered {before - after} rows from broken/pending races "
              f"({before} → {after} horses)")

    return df


def _compute_roi_curve(df, edge_col, label, current_threshold,
                      higher_is_better=True):
    """
    Sort bets by edge strength (descending if higher_is_better). Compute
    cumulative ROI as we include the top-N bets. Returns a DataFrame of
    diagnostic rows at meaningful quantile cutoffs.

    Bet payout model: flat $2 WIN bet per horse. Winners pay the
    ml_odds × $1 = net of (ml_decimal - 1) × $2 winnings on $2 cost; losers
    pay $0 returning -$2. This matches the LOG-top / All predict_v2 bets
    line of backtest_kfold but without the WPS layering (since we're
    diagnosing the trigger, not the payout structure).
    """
    if edge_col not in df.columns:
        print(f"  {label}: {edge_col} column missing, skipping")
        return None

    # Filter to rows where the edge value is defined AND the bet is
    # currently POSITIVE (above the signal's break-even). We're asking
    # "among bets the model thinks have edge, how does ROI degrade as we
    # relax the edge requirement?"
    work = df[df[edge_col].notna()].copy()
    if len(work) == 0:
        print(f"  {label}: no rows with {edge_col} defined")
        return None

    # Sort by edge descending (strongest edge first)
    work = work.sort_values(edge_col, ascending=not higher_is_better)

    # Compute per-bet payout assuming flat $2 win bet
    # Need ml_decimal to compute payouts. If not present, use ml_odds_decimal,
    # ml_prob, etc. walk_forward_summary has ml_prob_norm. Convert back:
    # ml_prob * ml_decimal ≈ 1 (for non-overround form). Approximate:
    #   ml_decimal = 1 / ml_prob (simple form)
    # Better: use ml_odds if it's a string like "5/2"
    ml_dec = None
    if 'ml_decimal' in work.columns:
        ml_dec = work['ml_decimal'].values
    elif 'ml_prob' in work.columns:
        # Approximation; actual decimal odds include overround
        with np.errstate(divide='ignore'):
            ml_dec = np.where(work['ml_prob'] > 0, 1.0 / work['ml_prob'], np.nan)
    else:
        print(f"  {label}: no ml_decimal or ml_prob, can't compute payouts")
        return None

    won = work['won'].values.astype(float)
    cost = np.full(len(work), 2.0)
    # WIN bet payout: $2 returns $2 × ml_decimal on a win (gross), else $0
    payout = np.where(won == 1, 2.0 * ml_dec, 0.0)

    # Cumulative ROI as a function of N (top-N bets included)
    cum_cost = np.cumsum(cost)
    cum_payout = np.cumsum(payout)
    cum_roi = (cum_payout - cum_cost) / cum_cost
    cum_wins = np.cumsum(won)

    n = len(work)

    # Report at meaningful cutoffs: threshold values at each decile + current
    # production threshold
    cutoffs = []
    for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                 0.70, 1.00]:
        k = max(1, int(n * frac))
        cutoffs.append({
            'cutoff': f"top {int(frac*100)}%",
            'n_bets': k,
            'threshold': work[edge_col].iloc[k-1],
            'roi_pct': cum_roi[k-1] * 100,
            'win_pct': (cum_wins[k-1] / k) * 100,
            'payout': cum_payout[k-1],
            'cost': cum_cost[k-1],
        })

    # Also show the current production threshold for comparison
    if higher_is_better:
        at_threshold = int((work[edge_col] >= current_threshold).sum())
    else:
        at_threshold = int((work[edge_col] <= current_threshold).sum())

    if at_threshold > 0:
        k = at_threshold
        cutoffs.append({
            'cutoff': f"current ({current_threshold})",
            'n_bets': k,
            'threshold': current_threshold,
            'roi_pct': cum_roi[k-1] * 100,
            'win_pct': (cum_wins[k-1] / k) * 100,
            'payout': cum_payout[k-1],
            'cost': cum_cost[k-1],
        })

    return pd.DataFrame(cutoffs), cum_roi, cum_wins, work[edge_col].values


def _print_curve(cutoffs_df, label, edge_col):
    print(f"\n{'='*76}")
    print(f"  {label}  (sorted by {edge_col}, strongest edge first)")
    print(f"{'='*76}")
    print(f"  {'cutoff':<22s}{'n':>5s}{'threshold':>12s}{'ROI':>8s}{'win%':>7s}"
          f"{'cost':>7s}{'payout':>8s}")
    print(f"  {'-'*22} {'-'*4} {'-'*11} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
    for _, r in cutoffs_df.iterrows():
        marker = ''
        if 'current' in r['cutoff']:
            marker = ' ← current'
        print(f"  {r['cutoff']:<22s}{r['n_bets']:>5d}"
              f"{r['threshold']:>12.3f}{r['roi_pct']:>+7.1f}%"
              f"{r['win_pct']:>+6.1f}%"
              f"{r['cost']:>6.0f} {r['payout']:>7.0f}{marker}")


def _find_peak(cum_roi, edge_values, min_bets=20):
    """Find the threshold giving peak ROI with at least min_bets."""
    if len(cum_roi) < min_bets:
        return None
    # Peak ROI over window where N >= min_bets
    peak_idx = np.argmax(cum_roi[min_bets-1:]) + (min_bets - 1)
    return {
        'n_bets': peak_idx + 1,
        'threshold': edge_values[peak_idx],
        'roi_pct': cum_roi[peak_idx] * 100,
    }


def main():
    parser = argparse.ArgumentParser(description='Bet threshold sweep diagnostic')
    parser.add_argument('--trigger', choices=['logistic', 'ridge', 'raw', 'all'],
                        default='all', help='Which trigger to sweep')
    args = parser.parse_args()

    df = _load()
    print(f"Loaded {len(df)} horses from walk_forward_summary.csv")

    # Per-trigger edge columns and break-even reference values
    specs = {
        'logistic': {
            'col': 'ev_per_2',
            'label': 'LOGISTIC trigger (ev_per_2)',
            'threshold': CURRENT_THRESHOLDS['logistic'],
            'breakeven': 0.0,  # ev_per_2 > 0 means positive EV
        },
    }

    # Ridge: edge = ridge_value - break_even, where break_even varies by race
    # A normalized metric: ridge_value / break_even
    if 'ridge_value' in df.columns:
        # break_even per race ~= RIDGE_RACE_TOTAL / n_horses / (1-takeout)
        # Computing per-row gives us a ridge_ratio that's directly comparable
        # across races
        RIDGE_RACE_TOTAL = 18.3
        take_gross = 1.0 / (1.0 - WIN_TAKEOUT)
        field_size = df.groupby(['date', 'track', 'race'])['horse_number'].transform('count')
        ridge_be = (RIDGE_RACE_TOTAL / field_size) * take_gross
        df['ridge_ratio'] = df['ridge_value'] / ridge_be
        specs['ridge'] = {
            'col': 'ridge_ratio',
            'label': 'RIDGE trigger (ridge_value / break_even)',
            'threshold': CURRENT_THRESHOLDS['ridge_mult'],
            'breakeven': 1.0,
        }

    # Raw: edge = (raw_value - raw_cv_mean) / raw_cv_std
    # This is the σ above mean that predict_v2's k-threshold measures
    if ('raw_value' in df.columns or 'raw_ridge_value' in df.columns):
        raw_col = 'raw_value' if 'raw_value' in df.columns else 'raw_ridge_value'
        if 'raw_cv_mean' in df.columns and 'raw_cv_std' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                df['raw_k'] = (df[raw_col] - df['raw_cv_mean']) / df['raw_cv_std']
            specs['raw'] = {
                'col': 'raw_k',
                'label': 'RAW trigger ((raw_value - μ) / σ)',
                'threshold': CURRENT_THRESHOLDS['raw_k'],
                'breakeven': 0.0,
            }
        else:
            print("  (raw_cv_mean/raw_cv_std columns missing; skipping raw sweep)")

    # Run the requested sweeps
    triggers_to_run = list(specs.keys()) if args.trigger == 'all' else [args.trigger]
    for t in triggers_to_run:
        if t not in specs:
            print(f"  SKIP: {t} trigger not available in this summary")
            continue
        spec = specs[t]
        result = _compute_roi_curve(df, spec['col'], spec['label'],
                                    spec['threshold'])
        if result is None:
            continue
        cutoffs_df, cum_roi, cum_wins, edges = result
        _print_curve(cutoffs_df, spec['label'], spec['col'])

        peak = _find_peak(cum_roi, edges, min_bets=20)
        if peak:
            print(f"\n  Peak ROI: {peak['roi_pct']:+.1f}% "
                  f"at threshold {peak['threshold']:.3f} "
                  f"(n={peak['n_bets']})")

        # Break-even reference
        n_be = int(np.sum(edges >= spec['breakeven']))
        if n_be > 0:
            be_roi = cum_roi[n_be - 1] * 100
            print(f"  At break-even ({spec['col']} >= {spec['breakeven']}): "
                  f"ROI={be_roi:+.1f}% (n={n_be})")

    print(f"\n{'='*76}")
    print(f"HOW TO READ")
    print(f"{'='*76}")
    print(f"  • Top-N% rows show cumulative ROI if you bet only the top-N% by edge")
    print(f"  • 'current' row shows where the production threshold lands today")
    print(f"  • Peak ROI shows the BEST possible cutoff in-sample (overfit warning)")
    print(f"  • If peak is MUCH better than current, tightening pays off")
    print(f"  • If peak N is tiny (<30), it's likely noise — ignore")
    print(f"  • If ROI is monotonically increasing as N shrinks, the edge is real")


if __name__ == '__main__':
    main()