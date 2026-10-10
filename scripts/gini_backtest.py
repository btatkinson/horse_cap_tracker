"""
gini_backtest.py

Diagnostic: does race-level gini coefficient (of ridge_value or
ensemble_prob across all horses in the race) discriminate good vs bad
single-race bets within each model's BET trigger?

The hypothesis (from Blake's intuition + earlier diagnostics):
  - HIGH gini = race has a strong-opinion outlier. If that outlier is a
    longshot, ridge over-predicts (D10 calibration gap). If it's heavy
    chalk, public has crushed the price. Either way bad.
  - LOW gini = race is muddled, model has no real opinion. "Agreement"
    on a coin-flip race doesn't mean much.
  - MODERATE gini = clear contender structure but not extreme. The
    sweet spot for trigger-based bets.

For each of LOG / RDG / RAW triggers, this script:
  1. Filters to horses where that model's BET flag fires
     (no floor-veto cross-checks — pure single-model trigger)
  2. Bins those horses into terciles by their RACE's gini coefficient
  3. Reports W-only ROI per tercile (W bets are where strongest edge
     has been found; pass --all-bet-types for full bet-type ROI)
  4. Does this for two gini measures: ridge_value gini AND
     ensemble_prob gini, to see if the signal is ridge-specific.

Reuses backtest_kfold's loading + signal preparation pipeline, so the
data here matches what backtest_kfold sees.

Caveats:
  - Sample sizes per tercile per bet type are small (~20-50 W bets per
    cell expected). Expect noisy ROIs. Look for consistency across
    models, not single-cell wins.
  - Gini is computed race-wide (every horse in the race). Field size
    affects gini, so larger fields tend to have higher gini values
    even at the same "concentration" structure. Tercile cuts are
    relative to the bet population, so this normalizes somewhat.
  - Negative ridge values shifted to non-negative before computing gini
    (gini formula requires non-negative inputs).

Usage:
    python gini_backtest.py
    python gini_backtest.py --source walk_forward
    python gini_backtest.py --save data/gini_diagnostic.csv
    python gini_backtest.py --all-bet-types
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])

# Pull in everything we need from backtest_kfold so loading & signal
# preparation match exactly. If the backtest evolves, this picks up
# the changes automatically.
sys.path.insert(0, str(Path(__file__).parent))
from backtest_kfold import (
    load_cv_predictions,
    compute_ev_per_2,
    apply_ridge_race_norm,
    compute_raw_calibration,
    WIN_TAKEOUT,
    LOGISTIC_EDGE_THRESHOLD,
    RIDGE_RACE_TOTAL,
    RIDGE_FLOOR,
    RIDGE_EDGE_MULTIPLIER,
    RAW_EDGE_K,
    ML_WIN_ONLY_MAX,
    ML_WP_MAX,
)


# =============================================================================
# 2. GINI COMPUTATION
# =============================================================================

def gini(values):
    """
    Standard gini coefficient on non-negative values. 0 = perfect
    equality, 1 = max inequality (one value gets all the mass).

    For a race of n horses with values v_1 <= v_2 <= ... <= v_n:
        gini = (2 * sum(i * v_i) - (n+1) * sum(v_i)) / (n * sum(v_i))

    Returns NaN if all values zero (race-level no-information case)
    or if input is empty.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan

    # Shift to non-negative if needed (ridge_value can go negative when
    # below break-even). Shift uses min, preserving relative spread.
    if arr.min() < 0:
        arr = arr - arr.min()

    if arr.sum() == 0:
        return np.nan  # all zeros — no inequality information

    arr = np.sort(arr)
    n = len(arr)
    cumsum = np.cumsum(arr)
    # Equivalent to (2 * sum(i * v_i) / (n * sum) ) - (n+1)/n
    return (2 * np.sum(np.arange(1, n + 1) * arr) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def add_race_gini(df, value_col, gini_col):
    """Add a column gini_col with race-level gini of value_col, broadcast
    to every row in that race."""
    if value_col not in df.columns:
        df[gini_col] = np.nan
        return df

    race_gini = df.groupby(['date', 'track', 'race'])[value_col].apply(gini)
    race_gini.name = gini_col
    df = df.merge(race_gini.reset_index(), on=['date', 'track', 'race'], how='left')
    return df


# =============================================================================
# 3. PER-MODEL TRIGGER FLAGS (no floor veto — pure trigger)
# =============================================================================

def compute_pure_triggers(df):
    """
    For each model, mark BET-trigger fires per horse. NO floor veto
    applied — we want each model's BET zone independently. This is
    different from backtest_kfold's any_bet logic (which AND's the
    edge trigger with no-floor-veto across active models).
    """
    df = df.copy()
    takeout_gross = 1.0 / max(1e-6, 1.0 - WIN_TAKEOUT)

    df['log_bet_pure'] = False
    df['rdg_bet_pure'] = False
    df['raw_bet_pure'] = False

    has_log = 'ev_per_2' in df.columns
    has_rdg = 'ridge_value_norm' in df.columns
    has_raw = 'raw_ridge_value' in df.columns

    if has_log:
        df['log_bet_pure'] = df['ev_per_2'].fillna(-np.inf) >= LOGISTIC_EDGE_THRESHOLD

    if has_rdg:
        # Per-race break-even — same as backtest_kfold
        for (_, _, _), race_df in df.groupby(['date', 'track', 'race']):
            n = len(race_df)
            if n == 0:
                continue
            ridge_be = (RIDGE_RACE_TOTAL / n) * takeout_gross
            ridge_edge = ridge_be * RIDGE_EDGE_MULTIPLIER
            mask = race_df.index
            df.loc[mask, 'rdg_bet_pure'] = (
                df.loc[mask, 'ridge_value_norm'].fillna(-np.inf) >= ridge_edge
            )

    if has_raw:
        # Per-row calibration if available, else card mean (matches backtest_kfold)
        if 'raw_cv_mean' in df.columns and df['raw_cv_mean'].notna().any():
            mean_s = df['raw_cv_mean']
            std_s = df['raw_cv_std']
        else:
            print("  WARNING: no per-row raw calibration; using card-pooled stats")
            mean_s = df.groupby(['date', 'track'])['raw_ridge_value'].transform('mean')
            std_s = df.groupby(['date', 'track'])['raw_ridge_value'].transform('std')
        raw_edge = mean_s + RAW_EDGE_K * std_s
        df['raw_bet_pure'] = df['raw_ridge_value'].fillna(-np.inf) >= raw_edge

    return df


# =============================================================================
# 4. BET EVALUATION (mirrors backtest_kfold simulate_bet)
# =============================================================================

def evaluate_subset(picks_df, label, bet_type=None, takeout=WIN_TAKEOUT):
    """
    Evaluate a subset of bets, returning n / hits / cost / payout / ROI.

    bet_type:
      None  → use predict_v2's ML-based rule per row (W/WP/WPS)
      'W'   → force win-only ($2 each, hit = finish 1)
      'WP'  → force win+place ($4 each, hit = finish 1 or 2)
      'WPS' → force win+place+show ($6 each, hit = finish 1, 2, or 3)

    The forced modes let us slice already-filtered subsets cleanly.
    """
    if picks_df.empty:
        return {
            'label': label, 'n_bets': 0, 'hits': 0,
            'cost': 0.0, 'payout': 0.0, 'roi': 0.0, 'win_rate': 0.0,
        }

    cost_total = 0.0
    payout_total = 0.0
    hits = 0
    for _, row in picks_df.iterrows():
        # Determine bet type
        if bet_type is None:
            ml_dec = row.get('ml_decimal', np.nan)
            if pd.isna(ml_dec):
                bt = 'WPS'  # conservative default, matches predict_v2
            elif ml_dec <= ML_WIN_ONLY_MAX:
                bt = 'W'
            elif ml_dec <= ML_WP_MAX:
                bt = 'WP'
            else:
                bt = 'WPS'
        else:
            bt = bet_type

        wp = row.get('win_paid_num', 0) or 0
        pp = row.get('place_paid_num', 0) or 0
        sp = row.get('show_paid_num', 0) or 0
        if pd.isna(wp): wp = 0
        if pd.isna(pp): pp = 0
        if pd.isna(sp): sp = 0
        finish = row.get('finish', np.nan)

        if bt == 'W':
            cost_total += 2
            payout_total += float(wp)
            if finish == 1:
                hits += 1
        elif bt == 'WP':
            cost_total += 4
            payout_total += float(wp) + float(pp)
            if finish in (1, 2):
                hits += 1
        else:  # WPS
            cost_total += 6
            payout_total += float(wp) + float(pp) + float(sp)
            if finish in (1, 2, 3):
                hits += 1

    return {
        'label': label, 'n_bets': len(picks_df), 'hits': hits,
        'cost': cost_total, 'payout': payout_total,
        'roi': (payout_total - cost_total) / cost_total if cost_total > 0 else 0.0,
        'win_rate': hits / len(picks_df),
    }


# =============================================================================
# 5. MAIN ANALYSIS — per-model trigger × gini terciles
# =============================================================================

def assign_terciles(df, value_col, n_bins=3):
    """Assign df rows to terciles 0/1/2 by value_col. Returns Series."""
    if df[value_col].notna().sum() < n_bins:
        return pd.Series([np.nan] * len(df), index=df.index)
    try:
        return pd.qcut(df[value_col], q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        return pd.Series([np.nan] * len(df), index=df.index)


def analyze_model_trigger(df, trigger_col, gini_col, model_label,
                          gini_label, bet_type_filter='W'):
    """
    For one (model trigger × gini measure × bet_type) combination, slice
    into terciles, report ROI per tercile.

    bet_type_filter:
      'W'   → only ML ≤ 4.5, evaluate as W bets ($2)
      'WP'  → only 4.5 < ML ≤ 9.0, evaluate as WP bets ($4)
      'WPS' → only ML > 9.0, evaluate as WPS bets ($6)
      'ALL' → no ML filter, evaluate per-row using predict_v2's rules

    Returns list of result rows for later aggregation.
    """
    triggered = df[df[trigger_col]].copy()
    if triggered.empty:
        print(f"\n  {model_label} × {gini_label} × {bet_type_filter}: no triggered bets")
        return []

    # Bet-type filter on ML decimal
    if bet_type_filter == 'W':
        subset = triggered[triggered['ml_decimal'].notna() &
                           (triggered['ml_decimal'] <= ML_WIN_ONLY_MAX)]
        force_bt = 'W'
    elif bet_type_filter == 'WP':
        subset = triggered[triggered['ml_decimal'].notna() &
                           (triggered['ml_decimal'] > ML_WIN_ONLY_MAX) &
                           (triggered['ml_decimal'] <= ML_WP_MAX)]
        force_bt = 'WP'
    elif bet_type_filter == 'WPS':
        subset = triggered[triggered['ml_decimal'].notna() &
                           (triggered['ml_decimal'] > ML_WP_MAX)]
        force_bt = 'WPS'
    elif bet_type_filter == 'ALL':
        subset = triggered
        force_bt = None  # use per-row ML rule
    else:
        raise ValueError(f"Unknown bet_type_filter: {bet_type_filter}")

    if subset.empty:
        print(f"\n  {model_label} × {gini_label} × {bet_type_filter}: "
              f"no bets after filter")
        return []

    # Tercile binning calibrated to this filtered subset
    subset['_gini_tercile'] = assign_terciles(subset, gini_col)
    if subset['_gini_tercile'].isna().all():
        print(f"\n  {model_label} × {gini_label} × {bet_type_filter}: "
              f"couldn't form terciles (n={len(subset)})")
        return []

    print(f"\n  {model_label} triggered, {bet_type_filter} bets, by {gini_label}:")
    print(f"    {'tercile':<10} {'gini range':<14} {'n':>4} {'hits':>5} "
          f"{'win%':>6} {'cost':>7} {'payout':>8} {'ROI':>8}")
    print(f"    {'-'*10} {'-'*14} {'-'*4} {'-'*5} {'-'*6} {'-'*7} "
          f"{'-'*8} {'-'*8}")

    rows = []
    for t in sorted(subset['_gini_tercile'].dropna().unique()):
        cell = subset[subset['_gini_tercile'] == t]
        gmin, gmax = cell[gini_col].min(), cell[gini_col].max()
        m = evaluate_subset(cell, f"T{int(t)+1}", bet_type=force_bt)
        tname = ['LOW', 'MID', 'HIGH'][int(t)]
        print(f"    T{int(t)+1} {tname:<7} "
              f"[{gmin:.2f}-{gmax:.2f}] "
              f"{m['n_bets']:>4} {m['hits']:>5} "
              f"{m['win_rate']:>5.1%} "
              f"${m['cost']:>5.0f} ${m['payout']:>6.0f} "
              f"{m['roi']:>+7.1%}")
        rows.append({
            'model': model_label, 'gini_measure': gini_label,
            'bet_type': bet_type_filter,
            'tercile': int(t), 'tercile_name': tname,
            'gini_min': gmin, 'gini_max': gmax,
            **m,
        })

    return rows


def plot_ml_gini_interaction(df, gini_col, gini_label, save_dir,
                              n_gini_bins=5):
    """
    For each model trigger, plot ROI vs gini with separate lines for
    different ML buckets. The unified hypothesis predicts:
      - Low ML (chalk) → ROI decreasing with gini
      - High ML (longshot) → ROI increasing with gini
    The plot makes the interaction visible at a glance.

    ML buckets:
      ≤ 2.5  : heavy chalk (W bet)
      2.5-4.5: medium chalk (W bet)
      4.5-9.0: middle (WP bet)
      > 9.0  : longshot (WPS bet)

    Each bucket gets binned by gini into n_gini_bins bins (default 5),
    ROI computed per bin with bootstrap CI.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  WARN: matplotlib not available, skipping plot")
        return None

    ml_buckets = [
        ('ML ≤ 2.5',   lambda x: x <= 2.5,                'W'),
        ('ML 2.5-4.5', lambda x: (x > 2.5) & (x <= 4.5),  'W'),
        ('ML 4.5-9.0', lambda x: (x > 4.5) & (x <= 9.0),  'WP'),
        ('ML > 9.0',   lambda x: x > 9.0,                 'WPS'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    triggers = [
        ('log_bet_pure', 'LOG'),
        ('rdg_bet_pure', 'RDG'),
        ('raw_bet_pure', 'RAW'),
    ]

    for ax, (trigger_col, model_label) in zip(axes, triggers):
        triggered = df[df[trigger_col]].copy()

        for ml_label, ml_filter, bt in ml_buckets:
            sub = triggered[
                triggered['ml_decimal'].notna() &
                triggered['ml_decimal'].apply(ml_filter)
            ].copy()

            if len(sub) < n_gini_bins * 2:
                # Not enough data to bin meaningfully — skip silently
                continue

            # Bin by gini value (qcut so equal-count bins)
            try:
                sub['_g_bin'] = pd.qcut(sub[gini_col], q=n_gini_bins,
                                        labels=False, duplicates='drop')
            except ValueError:
                continue

            xs, ys, ns = [], [], []
            for b in sorted(sub['_g_bin'].dropna().unique()):
                cell = sub[sub['_g_bin'] == b]
                m = evaluate_subset(cell, '', bet_type=bt)
                if m['cost'] == 0:
                    continue
                xs.append(cell[gini_col].mean())
                ys.append(m['roi'])
                ns.append(m['n_bets'])

            if not xs:
                continue

            ax.plot(xs, ys, marker='o', label=f"{ml_label} (n={sum(ns)})")
            # Annotate each point with its sample size
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(f'{n}', (x, y), textcoords='offset points',
                           xytext=(4, 4), fontsize=7, alpha=0.6)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(gini_label)
        ax.set_title(f'{model_label} triggered')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('ROI')
    fig.suptitle(
        f'Bet ROI vs {gini_label}, split by ML bucket\n'
        f'(numbers above points = bet count in that bin)',
        fontsize=11
    )
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f'gini_ml_interaction_{gini_label}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def consistency_summary(all_rows):
    """
    Cross-model consistency check: for each gini_measure, do the three
    models show the same direction of pattern across terciles?

    Heuristic: rank tercile ROIs within each (model, gini_measure)
    group. If terciles consistently rank the same way across models,
    the signal is robust. If they differ wildly, gini interacts with
    each model differently (less compelling).
    """
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    print()
    print("=" * 78)
    print("CROSS-MODEL CONSISTENCY")
    print("=" * 78)
    print("  For each gini measure, which tercile is best-ROI per model?")
    print("  If all three models agree on the best tercile, the signal is")
    print("  about gini itself, not just one model's quirks.")

    for gini_label in df['gini_measure'].unique():
        sub = df[df['gini_measure'] == gini_label]
        print(f"\n  {gini_label}:")
        for model in sub['model'].unique():
            msub = sub[sub['model'] == model]
            if msub.empty:
                continue
            best_idx = msub['roi'].idxmax()
            worst_idx = msub['roi'].idxmin()
            best = msub.loc[best_idx]
            worst = msub.loc[worst_idx]
            print(f"    {model:<22} best=T{best['tercile']+1} ({best['tercile_name']}) "
                  f"@{best['roi']:+.1%}  worst=T{worst['tercile']+1} ({worst['tercile_name']}) "
                  f"@{worst['roi']:+.1%}")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Gini coefficient diagnostic')
    parser.add_argument('--source', type=str, default=None,
                        choices=['walk_forward', 'cv', None],
                        help="Which predictions to backtest. Same flag as "
                             "backtest_kfold. Default: auto-detect "
                             "(prefers walk_forward).")
    parser.add_argument('--save', type=str, default=None,
                        help='Save per-cell results to CSV')
    parser.add_argument('--all-bet-types', action='store_true',
                        help='Run stratified W/WP/WPS analysis (default: '
                             'W-only). When set, runs each bet type as its '
                             'own population so we can test the unified '
                             'hypothesis: low-gini for chalks, high-gini '
                             'for longshots.')
    args = parser.parse_args()

    # Same loading pipeline as backtest_kfold so data prep is consistent.
    # load_cv_predictions handles auto-detection between walk_forward and
    # cv sources, normalizes column names, merges won/finish from entries,
    # merges payout columns, etc.
    df = load_cv_predictions(source=args.source)

    # Compute ev_per_2 — left as-is if walk_forward already wrote it
    df = compute_ev_per_2(df, takeout=WIN_TAKEOUT)

    # Per-race ridge normalization → creates ridge_value_norm column
    if 'ridge_value' in df.columns:
        df = apply_ridge_race_norm(df, 'ridge_value',
                                   race_total=RIDGE_RACE_TOTAL,
                                   floor=RIDGE_FLOOR)

    # Raw-ridge calibration. If per-row stamps exist (walk_forward path),
    # compute_pure_triggers will use them; the pooled values are a fallback.
    if 'raw_ridge_value' in df.columns:
        raw_mean, raw_std = compute_raw_calibration(df, 'raw_ridge_value')
    else:
        raw_mean, raw_std = np.nan, np.nan

    # Compute race-level gini for ridge_value_norm and ensemble_prob
    print("\n  Computing race-level gini coefficients...")
    df = add_race_gini(df, 'ridge_value_norm', 'ridge_gini')
    # ensemble_prob lives as cv_prob after backtest_kfold's normalization
    # for cv source; for walk_forward source it's both ensemble_prob (raw)
    # AND cv_prob (alias). Use ensemble_prob if present, fall back to cv_prob.
    prob_col = 'ensemble_prob' if 'ensemble_prob' in df.columns else 'cv_prob'
    df = add_race_gini(df, prob_col, 'prob_gini')
    n_races = df.groupby(['date', 'track', 'race']).ngroups
    rg_valid = df.groupby(['date', 'track', 'race'])['ridge_gini'].first().notna().sum()
    pg_valid = df.groupby(['date', 'track', 'race'])['prob_gini'].first().notna().sum()
    print(f"    Total races: {n_races}")
    print(f"    Races with ridge_gini: {rg_valid}")
    print(f"    Races with prob_gini ({prob_col}): {pg_valid}")

    # Compute pure per-model triggers (no floor veto)
    print("\n  Computing pure per-model BET triggers (no floor veto)...")
    df = compute_pure_triggers(df)
    for col in ('log_bet_pure', 'rdg_bet_pure', 'raw_bet_pure'):
        n_trig = df[col].sum()
        if 'ml_decimal' in df.columns:
            n_w = (df[col] & (df['ml_decimal'] <= ML_WIN_ONLY_MAX)).sum()
            print(f"    {col}: {n_trig} fires ({n_w} W-only)")
        else:
            print(f"    {col}: {n_trig} fires")

    # Per-model × per-gini-measure × per-bet-type analysis
    print()
    print("=" * 78)
    print("GINI TERCILE ANALYSIS — STRATIFIED BY BET TYPE")
    print("=" * 78)
    print("  Each model's BET trigger fires on its own (no floor-veto cross-check).")
    print("  Triggered horses split into 3 buckets by their RACE's gini coefficient.")
    print("  Tercile cuts are calibrated to each (model × gini × bet_type) subset.")
    print()
    print("  Hypothesis being tested (Blake's unified rule):")
    print("    W bets  (chalk):    LOW gini = best (model picks an unhyped chalk)")
    print("    WPS bets (longshots): HIGH gini = best (model picks a horse")
    print("                          ignored because of an obvious favorite)")
    print("    WP bets (middle):   ambiguous — gradient between the two")

    bet_filters = ['W', 'WP', 'WPS'] if args.all_bet_types else ['W']
    all_rows = []
    for bet_filter in bet_filters:
        print()
        print(f"{'-'*78}")
        print(f"BET TYPE: {bet_filter}")
        print(f"{'-'*78}")
        for trigger_col, model_label in [
            ('log_bet_pure', 'LOG (cv_prob+EV)'),
            ('rdg_bet_pure', 'RDG (ridge_value)'),
            ('raw_bet_pure', 'RAW (raw_ridge)'),
        ]:
            for gini_col, gini_label in [
                ('ridge_gini', 'ridge_gini'),
                ('prob_gini',  'prob_gini'),
            ]:
                rows = analyze_model_trigger(
                    df, trigger_col, gini_col, model_label, gini_label,
                    bet_type_filter=bet_filter,
                )
                all_rows.extend(rows)

    # Cross-model consistency check
    consistency_summary(all_rows)

    # Generate plots: ML-bin × gini interaction
    print()
    print("=" * 78)
    print("PLOTS — ML-bin × gini interaction")
    print("=" * 78)
    plot_dir = DATA_PATH / 'gini_diagnostic'
    for gini_col, gini_label in [('ridge_gini', 'ridge_gini'),
                                  ('prob_gini', 'prob_gini')]:
        path = plot_ml_gini_interaction(df, gini_col, gini_label, plot_dir)
        if path:
            print(f"  Saved: {path}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\nSaved per-cell results to {out}")


if __name__ == '__main__':
    main()