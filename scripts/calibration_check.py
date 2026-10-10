"""
calibration_check.py

For each probability source (cv_prob, ensemble_prob, ml_win_prob_norm),
bin horses by predicted probability and report:
  - n_horses in bin
  - mean predicted probability
  - actual win rate in bin
  - delta (actual - predicted)

A well-calibrated source has |delta| close to zero in every bin.
Systematic over-confidence (predicting too high) shows negative delta
across the upper bins. That's the smoking gun for "model says 30% but
reality is 10%" pathology.

Two views:
  1. ALL horses across the dataset
  2. BET-TRIGGERED horses only (the actual selection set)

If a source is calibrated globally but miscalibrated on triggered bets,
the trigger logic is selecting *anti-edge* — picks within a bin that are
worse than the bin average. That would explain "favorite triggers lose
at extraordinary rates" without requiring the whole model to be broken.

Usage:
    python calibration_check.py
    python calibration_check.py --source cv     # use regression_predictions.csv
    python calibration_check.py --bin-edges 0,0.05,0.10,0.15,0.20,0.30,0.50,1.0
"""

import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
ENTRIES_DIR = DATA_PATH / 'entries'

# Default bins skewed toward the favorites end since that's where most
# probability mass lives and where calibration matters most for betting.
DEFAULT_BIN_EDGES = [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 1.0]


def _parse_payout(val):
    if pd.isna(val):
        return None
    s = str(val).strip().replace('$', '').replace(',', '')
    if s == '' or s.lower() in ('nan', 'none', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_ml_odds(odds_str):
    """Returns implied probability."""
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '':
            return np.nan
        s = str(odds_str).strip()
        if '/' in s:
            num, den = map(float, s.split('/'))
            return den / (num + den)
        val = float(s)
        return 1 / (val + 1) if val >= 1 else val
    except (ValueError, ZeroDivisionError):
        return np.nan


def load_predictions(source='auto'):
    """Load predictions; auto-prefers walk_forward."""
    wf_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
    cv_path = DATA_PATH / 'regression_predictions.csv'

    if source == 'auto':
        path = wf_path if wf_path.exists() else cv_path
    elif source == 'walk_forward':
        path = wf_path
    elif source == 'cv':
        path = cv_path
    else:
        raise ValueError(f"source must be 'auto', 'walk_forward', or 'cv'")

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}")
    return df, path


def merge_outcomes(df):
    """Merge `won` and bet-trigger flags from entries if missing."""
    if 'won' not in df.columns or 'ml_win_prob_norm' not in df.columns:
        # Need to merge from entries
        if not ENTRIES_DIR.exists():
            print("WARNING: entries dir missing; cannot merge outcomes")
            return df

        entry_dfs = []
        for f in ENTRIES_DIR.glob('*.csv'):
            try:
                entry_dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if not entry_dfs:
            return df

        entries = pd.concat(entry_dfs, ignore_index=True)

        # Normalize keys
        for col in ('date', 'track'):
            if col in entries.columns:
                entries[col] = entries[col].astype(str).str.strip().str.lower()
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        entries['race'] = pd.to_numeric(entries['race'], errors='coerce')
        entries['horse_number'] = pd.to_numeric(
            entries.get('horse_number', entries.get('horse')), errors='coerce')
        df['race'] = pd.to_numeric(df['race'], errors='coerce')
        df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')

        if 'finish' in entries.columns:
            entries['_finish'] = pd.to_numeric(entries['finish'], errors='coerce')
            entries['won'] = (entries['_finish'] == 1).astype(int)

        if 'ml_odds' in entries.columns:
            entries['ml_win_prob'] = entries['ml_odds'].apply(_parse_ml_odds)

        bring_cols = ['date', 'track', 'race', 'horse_number']
        for c in ('won', '_finish', 'ml_win_prob'):
            if c in entries.columns and c not in df.columns:
                bring_cols.append(c)

        df = df.merge(
            entries[bring_cols].drop_duplicates(
                ['date', 'track', 'race', 'horse_number']),
            on=['date', 'track', 'race', 'horse_number'],
            how='left',
        )

        if 'won' in df.columns:
            df['won'] = df['won'].fillna(0).astype(int)

        # Per-race ML normalization (matches predict_v2)
        if 'ml_win_prob' in df.columns and 'ml_win_prob_norm' not in df.columns:
            df['ml_win_prob_norm'] = np.nan
            for _, race_df in df.groupby(['date', 'track', 'race']):
                idx = race_df.index
                raw = race_df['ml_win_prob'].values
                valid = ~np.isnan(raw)
                if valid.any():
                    s = raw[valid].sum()
                    if s > 0:
                        df.loc[idx[valid], 'ml_win_prob_norm'] = raw[valid] / s

    return df


def calibration_table(df, prob_col, bin_edges, label_col='won', subset_label='ALL'):
    """
    Bin horses by predicted probability and compare to actual win rate.

    Returns a DataFrame with one row per bin.
    """
    work = df[[prob_col, label_col]].copy().dropna()
    if work.empty:
        return None

    work['bin'] = pd.cut(work[prob_col], bins=bin_edges, include_lowest=True)
    rows = []
    for bin_label, g in work.groupby('bin', observed=True):
        n = len(g)
        if n == 0:
            continue
        mean_pred = g[prob_col].mean()
        actual = g[label_col].mean()
        delta = actual - mean_pred
        # Standard error on the actual rate (binomial)
        se = np.sqrt(actual * (1 - actual) / n) if n > 0 else np.nan
        rows.append({
            'bin': str(bin_label),
            'n': n,
            'mean_pred': mean_pred,
            'actual': actual,
            'delta': delta,
            'se': se,
            'sigma': delta / se if (se and se > 0) else np.nan,
        })
    return pd.DataFrame(rows)


def print_calibration_table(table, title):
    """Pretty-print one calibration table."""
    print(f"\n  {title}")
    print(f"    {'bin':<22s} {'n':>5s} {'pred':>7s} {'actual':>7s} "
          f"{'delta':>8s} {'sigma':>7s}")
    print(f"    {'-'*22} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    if table is None or table.empty:
        print(f"    (no data)")
        return
    for _, r in table.iterrows():
        # Sigma > 2 is a 95% confidence indicator that the bin is miscalibrated
        sig_marker = ''
        if not np.isnan(r['sigma']):
            if abs(r['sigma']) >= 3:
                sig_marker = ' ***'
            elif abs(r['sigma']) >= 2:
                sig_marker = ' **'
            elif abs(r['sigma']) >= 1:
                sig_marker = ' *'
        print(f"    {r['bin']:<22s} {r['n']:>5d} "
              f"{r['mean_pred']:>6.1%} {r['actual']:>6.1%} "
              f"{r['delta']:>+7.1%} {r['sigma']:>+6.2f}{sig_marker}")


def overall_summary(table, label):
    """One-line summary: weighted mean delta across bins, by horse-count."""
    if table is None or table.empty:
        return
    total_n = table['n'].sum()
    if total_n == 0:
        return
    weighted_pred = (table['mean_pred'] * table['n']).sum() / total_n
    weighted_actual = (table['actual'] * table['n']).sum() / total_n
    overall_delta = weighted_actual - weighted_pred
    # Aggregate sigma: total deviation in win counts vs expected
    expected_wins = (table['mean_pred'] * table['n']).sum()
    actual_wins = (table['actual'] * table['n']).sum()
    se_total = np.sqrt(sum(
        n * p * (1 - p) for n, p in zip(table['n'], table['mean_pred'])
    ))
    sigma = (actual_wins - expected_wins) / se_total if se_total > 0 else np.nan
    print(f"  {label}: avg pred {weighted_pred:.1%}, "
          f"avg actual {weighted_actual:.1%}, "
          f"delta {overall_delta:+.1%} ({sigma:+.1f}σ over {total_n} horses)")


def _derive_bet_flags(df, win_takeout=0.17,
                      logistic_edge=0.05,
                      ridge_race_total=18.3,
                      ridge_edge_mult=1.05,
                      raw_edge_k=1.0):
    """
    Minimal re-derivation of any_bet from raw signals — matches
    backtest_kfold.compute_bet_flags. Used so VIEW 2 (bet-triggered
    horses) works without requiring backtest_kfold to have been run.

    Honors per-row raw_cv_mean/std stamps when present (walk_forward
    source); otherwise pools across the loaded data.
    """
    df = df.copy()
    takeout_gross = 1.0 / max(1e-6, 1.0 - win_takeout)

    # LOG bet trigger: ev_per_2 >= edge threshold (uses pre-computed ev_per_2
    # from predict_v2 if present, otherwise computes from cv_prob/ml_prob).
    if 'ev_per_2' not in df.columns:
        if 'cv_prob' in df.columns and 'ml_win_prob_norm' in df.columns:
            ml = df['ml_win_prob_norm']
            df['ev_per_2'] = np.where(
                ml > 0,
                2 * ((df['cv_prob'] / ml) * (1 - win_takeout) - 1),
                np.nan,
            )
        else:
            df['ev_per_2'] = np.nan

    df['log_bet'] = df['ev_per_2'].fillna(-np.inf) >= logistic_edge

    # RDG bet trigger: per-race ridge_value >= ridge_be * edge_multiplier
    df['rdg_bet'] = False
    if 'ridge_value' in df.columns:
        for _, race_df in df.groupby(['date', 'track', 'race']):
            n = len(race_df)
            if n == 0:
                continue
            ridge_be = (ridge_race_total / n) * takeout_gross
            ridge_edge = ridge_be * ridge_edge_mult
            df.loc[race_df.index, 'rdg_bet'] = (
                race_df['ridge_value'].fillna(-np.inf) >= ridge_edge)

    # RAW bet trigger: raw_value >= raw_mean + K * raw_std
    df['raw_bet'] = False
    raw_col = None
    for cand in ('raw_value', 'raw_ridge_value'):
        if cand in df.columns:
            raw_col = cand
            break
    if raw_col is not None:
        if 'raw_cv_mean' in df.columns and 'raw_cv_std' in df.columns and \
                df['raw_cv_mean'].notna().any():
            # Per-row stamps — use them
            mean = df['raw_cv_mean'].fillna(df[raw_col].mean())
            std = df['raw_cv_std'].fillna(df[raw_col].std())
        else:
            # Pooled fallback
            m = df[raw_col].mean()
            s = df[raw_col].std()
            mean = pd.Series(m, index=df.index)
            std = pd.Series(s, index=df.index)
        threshold = mean + raw_edge_k * std
        df['raw_bet'] = df[raw_col].fillna(-np.inf) >= threshold

    df['any_bet'] = df['log_bet'] | df['rdg_bet'] | df['raw_bet']
    return df


def main():
    parser = argparse.ArgumentParser(description='Calibration diagnostic')
    parser.add_argument('--source', type=str, default='auto',
                       choices=['auto', 'walk_forward', 'cv'])
    parser.add_argument('--bin-edges', type=str, default=None,
                       help='Comma-separated bin edges, e.g. "0,0.05,0.10,..."')
    args = parser.parse_args()

    if args.bin_edges:
        bin_edges = [float(x) for x in args.bin_edges.split(',')]
    else:
        bin_edges = DEFAULT_BIN_EDGES

    df, _ = load_predictions(args.source)
    df = merge_outcomes(df)

    if 'won' not in df.columns:
        print("ERROR: no 'won' column available — cannot check calibration")
        return

    # Drop broken races before calibration — same logic as backtest_kfold.
    # A race where the winner isn't in predictions contributes only losses
    # (won=0) to calibration, which biases the "actual" win rate downward
    # and makes every probability bin look over-predicted.
    #
    # Three categories:
    #   - Pending (today's card, no finish data) → dropped silently
    #   - Broken (finish exists but winner missing from predictions)
    #   - Multi-winner (duplicate finish=1, data bug)
    winners_per_race = df.groupby(['date', 'track', 'race'])['won'].sum()
    good = winners_per_race[winners_per_race == 1].index
    zero_winner = winners_per_race[winners_per_race == 0].index
    multi_winner = winners_per_race[winners_per_race > 1].index.tolist()

    if 'finish' in df.columns:
        finish_present = df.groupby(
            ['date', 'track', 'race'])['finish'].apply(
            lambda s: s.notna().any())
    else:
        finish_present = pd.Series(True, index=winners_per_race.index)

    pending = [r for r in zero_winner if not finish_present.get(r, True)]
    broken = [r for r in zero_winner if finish_present.get(r, True)]

    total_bad = len(pending) + len(broken) + len(multi_winner)
    if total_bad > 0:
        race_key = df.set_index(['date', 'track', 'race']).index
        n_before = len(df)
        df = df[race_key.isin(good)].copy()
        msgs = []
        if pending:
            msgs.append(f"{len(pending)} pending (no results yet)")
        if broken:
            msgs.append(f"{len(broken)} broken (winner not in predictions)")
        if multi_winner:
            msgs.append(f"{len(multi_winner)} multi-winner")
        print(f"  Filtered {total_bad} races "
              f"({', '.join(msgs)}): {n_before} → {len(df)} horses")

    print(f"\n{'='*72}")
    print(f"CALIBRATION DIAGNOSTIC")
    print(f"{'='*72}")
    print(f"  Bin edges: {bin_edges}")
    print(f"  Significance markers: * = 1σ, ** = 2σ, *** = 3σ")
    print(f"  Negative delta = model OVER-predicts (predicts higher than actual)")
    print(f"  Positive delta = model UNDER-predicts")

    # Probability sources to check
    sources = []
    if 'cv_prob' in df.columns:
        sources.append(('cv_prob', 'Logistic CV (regression_analysis output)'))
    if 'ensemble_prob' in df.columns:
        sources.append(('ensemble_prob', 'Ensemble LLM+ML+regression (predict_v2)'))
    if 'ml_win_prob_norm' in df.columns:
        sources.append(('ml_win_prob_norm', 'Morning Line (market baseline)'))

    print(f"\n{'─'*72}")
    print(f"VIEW 1: ALL HORSES")
    print(f"{'─'*72}")
    for col, label in sources:
        table = calibration_table(df, col, bin_edges)
        print_calibration_table(table, label)
        overall_summary(table, label)

    # Now check on triggered bets specifically
    # Derive bet flags inline if not present (matches backtest_kfold logic)
    has_flags = ('any_bet' in df.columns or
                 all(c in df.columns for c in ['log_bet', 'rdg_bet', 'raw_bet']))
    if not has_flags:
        df = _derive_bet_flags(df)
        has_flags = 'any_bet' in df.columns

    if has_flags:
        if 'any_bet' not in df.columns:
            df['any_bet'] = (df.get('log_bet', False) |
                            df.get('rdg_bet', False) |
                            df.get('raw_bet', False))
        bet_df = df[df['any_bet']]
        if len(bet_df) > 0:
            print(f"\n{'─'*72}")
            print(f"VIEW 2: BET-TRIGGERED HORSES ONLY ({len(bet_df)} bets)")
            print(f"  Are the horses we bet on calibrated, or are they ANTI-selected?")
            print(f"{'─'*72}")
            for col, label in sources:
                table = calibration_table(bet_df, col, bin_edges)
                print_calibration_table(table, label)
                overall_summary(table, label)
        else:
            print("\n(No triggered bets in dataset for VIEW 2)")
    else:
        print(f"\n(Could not derive bet flags — VIEW 2 skipped)")

    # Interpretation guide
    print(f"\n{'='*72}")
    print(f"HOW TO INTERPRET")
    print(f"{'='*72}")
    print(f"  • If ALL HORSES is well-calibrated but BET-TRIGGERED is not:")
    print(f"    The model is fine; the BET selection logic is anti-edge.")
    print(f"    Fix: rethink the trigger criteria (edge_threshold, etc).")
    print(f"  • If ALL HORSES is also miscalibrated:")
    print(f"    The probability model itself is biased. Fix the model.")
    print(f"  • If ml_win_prob_norm is well-calibrated but model probs aren't:")
    print(f"    The market knows something your model doesn't.")
    print(f"  • If even ml_win_prob_norm is miscalibrated:")
    print(f"    Likely a data/parsing issue (or a small/biased sample).")


if __name__ == '__main__':
    main()