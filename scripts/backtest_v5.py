"""
backtest_v5.py

Threshold sweep for predict_v5's EV-ratio bet rule. For each candidate
bet_fraction, computes the resulting value_ratio cutoff and reports
n_bets, win rate, ROI, and dollar PnL.

Pipe:
    walk_forward_summary.csv  →  apply payout_model.json  →
    compute value_ratio = ensemble_prob × predicted_payout / 2  →
    for each bet_fraction f, cutoff = q(1-f) of historical ratios  →
    bet every horse with ratio ≥ cutoff, simulate_bet, aggregate ROI

Two evaluation modes:

  In-sample (default): cutoff is set on the SAME data we evaluate on.
    Fast and useful for seeing the shape of the threshold→ROI curve,
    but the absolute ROI is optimistic — you're picking the threshold
    that worked for this exact dataset.

  Honest (--honest): leave-one-card-out cutoff. For each card, use the
    OTHER cards' history to set the cutoff, then bet that card with it.
    Genuinely out-of-sample. Slower but the ROI numbers you can trust.

Usage:
    python backtest_v5.py
    python backtest_v5.py --takeout 0.17
    python backtest_v5.py --honest
    python backtest_v5.py --fractions 0.05,0.10,0.15,0.20,0.25,0.50

Reads:
    {DATA_PATH}/predictions/walk_forward_summary.csv
    {DATA_PATH}/payout_model.json

Pre-requisites:
    1. Run backtest_walk_forward.py to populate walk_forward_summary.csv
    2. Run regression_analysis.py to fit payout_model.json (the 4th model;
       the patched regression_analysis.py does this automatically)

What "ROI" means here
---------------------
  ROI = (total_payout - total_cost) / total_cost
  takeout_cut_pct = ROI - (-takeout)

A random bettor in a 17%-takeout pool achieves ROI = -0.17 in the limit.
ROI > -0.17 means we cut into the takeout. ROI > 0 means we beat the
house entirely. predict_v5's bet rule fires WPS by default (matching
v2), so cost = $6 per bet, payout = win_paid + place_paid + show_paid.
"""

import os, argparse, warnings, sys
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])

# Reuse the existing loader + simulation helpers from backtest_kfold.
# Kept as imports rather than duplicated so any future change to the
# bet-type logic in backtest_kfold flows through to v5 automatically.
sys.path.insert(0, str(Path(__file__).parent))
from backtest_kfold import (
    load_cv_predictions,
    simulate_bet,
    evaluate_system,
    WIN_TAKEOUT,
)
from payout_model import load_payout_model, predict_payout


PAYOUT_MODEL_PATH = DATA_PATH / 'payout_model.json'

# Default sweep grid. Spans the meaningful range from "very selective"
# (top 5% of horses) through "all in" (bet everything with valid model
# output). Including 1.0 as the floor lets you see baseline ROI for
# "bet every horse" — the takeout-only loss line.
DEFAULT_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]


def filter_cards_without_results(df):
    """Drop cards where results never landed in entries/.

    load_cv_predictions zero-fills payout columns when the entries merge
    misses (line ~457 of backtest_kfold). The result is that any card
    whose entries CSV hasn't been backfilled yet looks like a card where
    every horse lost and paid nothing — biasing ROI strictly downward
    by treating real wins as losses.

    Heuristic: every actual race has exactly one winner who pays at
    least $2.10. So a card whose entire win_paid_num column is 0 is a
    card with missing results, not a card where every horse lost (the
    latter is impossible). Drop those cards entirely with a loud print
    so you know what got excluded.

    Returns (filtered_df, list_of_skipped_cards).
    """
    if 'win_paid_num' not in df.columns:
        return df, []

    skipped = []
    keep_idx = []
    for (date, track), card_df in df.groupby(['date', 'track']):
        # A card with any positive payout has at least one race's results.
        # Cards with zero across every horse are missing results entirely.
        # Note: in theory you could have a card where the merge partially
        # landed (some races recovered, others not). That's much rarer than
        # whole-card-missing and would require row-level inference; the
        # whole-card heuristic is safer because it has no false positives
        # — every real card MUST have positive payouts somewhere.
        max_payout = card_df['win_paid_num'].max()
        if max_payout > 0:
            keep_idx.extend(card_df.index)
        else:
            skipped.append((date, track, len(card_df)))

    if skipped:
        print(f"\n  ⚠ Dropped {len(skipped)} card(s) with no results data "
              f"(missing entries → zero-filled by loader):")
        for date, track, n_horses in skipped:
            print(f"      {date} / {track}  ({n_horses} horses)")
        print(f"    These cards bias ROI downward — every horse looks like "
              f"a $6→$0 loss. Backfill the entries CSV and re-run for "
              f"complete numbers.")

    return df.loc[keep_idx].copy(), skipped


def add_value_ratio(df, payout_model, bet_size=2.0):
    """Apply payout model to df and compute value_ratio per row.

    Returns df with new columns:
        predicted_payout: model's expected win_paid (in $)
        value_ratio:      ensemble_prob × predicted_payout / bet_size

    Uses ensemble_prob (predict_v2's blended P(win)) — the same column
    predict_v5's calibration uses. Walk_forward_summary's ensemble_prob
    is genuinely OOS for the logistic since backtest_walk_forward
    retrained per holdout.
    """
    if 'ensemble_prob' not in df.columns:
        # backtest_kfold's load_cv_predictions copies ensemble_prob → cv_prob,
        # so this only fires if neither was present (unlikely but defensive).
        if 'cv_prob' in df.columns:
            df = df.copy()
            df['ensemble_prob'] = df['cv_prob']
        else:
            raise RuntimeError(
                "df has neither ensemble_prob nor cv_prob — can't compute "
                "value_ratio without a P(win) signal."
            )

    df = predict_payout(df, payout_model)
    df['value_ratio'] = (df['ensemble_prob'] * df['predicted_payout']) / bet_size
    return df


def sweep_in_sample(df, fractions, takeout=WIN_TAKEOUT):
    """In-sample cutoff sweep — cutoff is set from the same df we evaluate.

    Returns a list of dicts, one per fraction, with cutoff + ROI metrics.
    """
    valid_ratios = df['value_ratio'].dropna()
    n_total = len(valid_ratios)
    if n_total < 100:
        raise RuntimeError(
            f"Only {n_total} rows have valid value_ratio. "
            f"Need 100+ for meaningful threshold sweep."
        )

    results = []
    for frac in fractions:
        if frac >= 1.0:
            # "Bet everything" — no cutoff. Useful baseline showing
            # the takeout-only loss line for context.
            cutoff = float(valid_ratios.min()) - 1.0
        else:
            cutoff = float(np.quantile(valid_ratios, 1.0 - frac))

        picks = df[df['value_ratio'].notna() & (df['value_ratio'] >= cutoff)]
        metrics = evaluate_system(picks, label=f'frac={frac:.2f}', takeout=takeout)
        metrics['fraction'] = frac
        metrics['cutoff'] = cutoff
        metrics['mean_ratio_picked']    = float(picks['value_ratio'].mean()) if not picks.empty else float('nan')
        metrics['mean_ev_dollars']      = float(picks['value_ratio'].mean() * 2 - 2) if not picks.empty else float('nan')
        metrics['mean_pred_payout']     = float(picks['predicted_payout'].mean()) if not picks.empty else float('nan')
        results.append(metrics)
    return results


def sweep_honest(df, fractions, takeout=WIN_TAKEOUT):
    """Leave-one-card-out cutoff sweep.

    For each (date, track) card: use the OTHER cards' value_ratio
    distribution to compute the cutoff, then evaluate that card's
    picks against it. Aggregate across all cards.

    This is the honest version — the cutoff for each card is set
    without using that card's data. Slower than in-sample but the
    ROI numbers you can trust.
    """
    if not all(c in df.columns for c in ('date', 'track')):
        raise RuntimeError(
            "Honest mode needs 'date' and 'track' columns — "
            "walk_forward_summary.csv should have them. Re-run "
            "backtest_walk_forward.py if missing."
        )

    cards = df.groupby(['date', 'track']).ngroups
    if cards < 5:
        raise RuntimeError(
            f"Only {cards} unique cards in df. Honest leave-one-out "
            f"needs at least 5 cards to be meaningful. Use the default "
            f"in-sample sweep instead."
        )

    valid_mask = df['value_ratio'].notna()

    # Pre-build per-card index lists so groupby isn't called inside the
    # quantile loop. With ~50 cards × 10 fractions, the inner loop runs
    # 500 times; precomputing is meaningfully faster.
    card_groups = list(df.groupby(['date', 'track']).groups.items())

    results = []
    for frac in fractions:
        # Aggregate picks across all cards into one bet log
        all_picks = []

        for (date, track), idx in card_groups:
            # Heldout card's rows
            this_card = df.loc[idx]
            # Other cards' value_ratios — the calibration sample for THIS card
            others_mask = valid_mask & ~df.index.isin(idx)
            others = df.loc[others_mask, 'value_ratio']
            if len(others) < 50:
                # Not enough other-card data to set a stable cutoff for this
                # card. Skip — including it would be noisy and could distort
                # the aggregate ROI. The skipped cards just don't contribute
                # bets to this fraction's tally.
                continue

            if frac >= 1.0:
                cutoff = float(others.min()) - 1.0
            else:
                cutoff = float(np.quantile(others, 1.0 - frac))

            picks = this_card[
                this_card['value_ratio'].notna() &
                (this_card['value_ratio'] >= cutoff)
            ]
            if not picks.empty:
                all_picks.append(picks)

        if all_picks:
            combined = pd.concat(all_picks, ignore_index=True)
            metrics = evaluate_system(
                combined, label=f'frac={frac:.2f}', takeout=takeout
            )
        else:
            metrics = evaluate_system(
                pd.DataFrame(), label=f'frac={frac:.2f}', takeout=takeout
            )

        metrics['fraction'] = frac
        # The cutoff varies per card in honest mode — report the
        # in-sample equivalent for reference, marked as such.
        metrics['cutoff'] = (float(np.quantile(df.loc[valid_mask, 'value_ratio'], 1.0 - frac))
                             if frac < 1.0 else float('nan'))
        metrics['mean_ratio_picked']  = float(combined['value_ratio'].mean()) if all_picks else float('nan')
        metrics['mean_ev_dollars']    = float(combined['value_ratio'].mean() * 2 - 2) if all_picks else float('nan')
        metrics['mean_pred_payout']   = float(combined['predicted_payout'].mean()) if all_picks else float('nan')
        results.append(metrics)

    return results


def print_sweep_table(results, takeout, mode_label):
    """Format the sweep results as an aligned table.

    Highlights the best ROI row and flags low-n bets. n_bets < 30 is
    statistically too thin to draw conclusions from — flagged so users
    don't chase a small-sample fluke.
    """
    print(f"\n{'='*92}")
    print(f"  Threshold sweep ({mode_label})  |  takeout={takeout:.0%}  "
          f"|  random-bettor ROI baseline = {-takeout:.1%}")
    print(f"{'='*92}")
    print(f"  {'frac':>6} {'cutoff':>8} {'n_bets':>7} {'win%':>6} "
          f"{'avg_ratio':>10} {'avg_EV$':>8} "
          f"{'cost':>9} {'payout':>9} {'ROI':>8} {'+/-rand':>9}")
    print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*6} "
          f"{'-'*10} {'-'*8} "
          f"{'-'*9} {'-'*9} {'-'*8} {'-'*9}")

    # Find best ROI for highlighting (only among trustworthy n_bets)
    best_idx = None
    best_roi = -np.inf
    for i, r in enumerate(results):
        if r['n_bets'] >= 30 and not np.isnan(r['roi']) and r['roi'] > best_roi:
            best_roi = r['roi']
            best_idx = i

    for i, r in enumerate(results):
        n = r['n_bets']
        roi = r['roi']
        cut = r.get('cutoff', float('nan'))
        ratio = r.get('mean_ratio_picked', float('nan'))
        ev_d = r.get('mean_ev_dollars', float('nan'))
        wr = r.get('win_rate', float('nan'))

        # Marker: low-n warning, or best ROI star
        marker = ''
        if n == 0:
            marker = ' (no bets)'
        elif n < 30:
            marker = ' ⚠ low n'
        elif i == best_idx:
            marker = ' ← best'

        cut_s   = f"{cut:.3f}"   if not np.isnan(cut)   else '   n/a'
        ratio_s = f"{ratio:.3f}" if not np.isnan(ratio) else '     n/a'
        ev_s    = f"{ev_d:+.2f}" if not np.isnan(ev_d)  else '    n/a'
        wr_s    = f"{wr:.1%}"    if not np.isnan(wr)    else '   n/a'
        roi_s   = f"{roi:+.1%}"  if not np.isnan(roi)   else '    n/a'
        # Takeout-cut: how far above random (negative = worse than random)
        cut_pct = roi - (-takeout) if not np.isnan(roi) else float('nan')
        cut_pct_s = f"{cut_pct:+.1%}" if not np.isnan(cut_pct) else '    n/a'

        print(f"  {r['fraction']:>6.2f} {cut_s:>8} {n:>7d} {wr_s:>6} "
              f"{ratio_s:>10} {ev_s:>8} "
              f"${r['total_cost']:>7.0f} ${r['total_payout']:>7.0f} "
              f"{roi_s:>8} {cut_pct_s:>9}{marker}")

    print()
    if best_idx is not None:
        best = results[best_idx]
        print(f"  Best ROI by trustworthy (n≥30) row: "
              f"frac={best['fraction']:.2f}, ROI={best['roi']:+.1%}, "
              f"n={best['n_bets']}")
        if best['roi'] > 0:
            print(f"  → Beats house: ROI={best['roi']:+.1%} > 0")
        elif best['roi'] > -takeout:
            cut_pct = best['roi'] - (-takeout)
            print(f"  → Cuts into takeout: {cut_pct:+.1%} above random")
        else:
            print(f"  → Worse than random. Either the value model is "
                  f"miscalibrated or the bet-type rule is wrong for this data.")
    else:
        print(f"  No fraction had n_bets≥30 with finite ROI. "
              f"Either all rows excluded by NaN ratio, or threshold range "
              f"is so tight no bets fire. Try wider --fractions.")


def diagnose_value_ratio(df):
    """Print diagnostic summary of the value_ratio distribution.

    Helps identify when something is off (e.g. payout model exploding,
    cutoff blowout) before interpreting ROI numbers.
    """
    valid = df['value_ratio'].dropna()
    if valid.empty:
        print(f"\n  ⚠ No valid value_ratios. Check payout model and "
              f"feature columns.")
        return

    n_total = len(df)
    n_valid = len(valid)
    print(f"\n  value_ratio diagnostics:")
    print(f"    rows total: {n_total}, with valid ratio: {n_valid} ({n_valid/n_total:.1%})")
    print(f"    median: {valid.median():.3f}  mean: {valid.mean():.3f}  "
          f"std: {valid.std():.3f}")
    print(f"    quantiles: q10={valid.quantile(0.1):.3f}  "
          f"q50={valid.quantile(0.5):.3f}  q75={valid.quantile(0.75):.3f}  "
          f"q90={valid.quantile(0.9):.3f}  q99={valid.quantile(0.99):.3f}")
    print(f"    pct with ratio ≥ 1.0 (positive EV): {(valid >= 1.0).mean():.1%}")
    if valid.median() > 5.0:
        print(f"    ⚠ Median ratio is unusually high (>5). Probably a "
              f"normalization issue in predict_payout. value_ratios should "
              f"hover near 1.0 (= break even).")


def decompose_signals(df, fractions, takeout=WIN_TAKEOUT):
    """Run the same threshold sweep ranked by each component independently.

    The value_ratio = ensemble_prob × predicted_payout / 2 has two
    factors that can each be (mis)calibrated. If value_ratio is
    anti-predictive but you don't know which factor is causing it, run
    the sweep on each factor in isolation:

      - ensemble_prob alone: does P(win) rank winners well? Should
        give monotonic win% (more selective → higher win rate).
      - predicted_payout alone: does the payout model rank winners well?
        Sorting by big-payout predictions should produce LONGSHOTS
        (low win rate) but LARGE payouts when they hit. Random ROI
        roughly equal to -takeout is a sign predicted_payout has no
        useful signal beyond "this horse is a longshot, which any
        ML-prob also tells us."
      - ml_prob_norm alone: market baseline. The public's odds turned
        into a probability, normalized per race.
      - value_ratio: the actual production signal, for comparison.

    A healthy 2-model EV setup looks like:
      - ensemble_prob:    monotonic win%, ROI gradually improves with
                          selectivity but doesn't beat takeout
                          (favorites are correctly priced)
      - predicted_payout: low win% at the top (longshots), occasional
                          big hit, ROI roughly -takeout overall
      - value_ratio:      win% AND ROI both improve with selectivity
                          (the multiplication captures genuine edge)

    A broken setup where ranking by individual factors works but
    value_ratio is anti-predictive points to a calibration mismatch —
    the factors are individually correct but their PRODUCT isn't,
    meaning the units don't combine cleanly into expected dollars.
    """
    print(f"\n{'='*92}")
    print(f"  Component diagnostic — sweep each ranking signal separately")
    print(f"{'='*92}")

    # Define the alternative ranking signals to test. Each is a (name,
    # column, ranking_descending) tuple. All these cases use a quantile
    # cutoff on the column itself rather than going through value_ratio.
    candidates = [
        ('ensemble_prob',    'ensemble_prob'),
        ('predicted_payout', 'predicted_payout'),
        ('ml_prob_norm',     'ml_prob_norm') if 'ml_prob_norm' in df.columns else None,
        ('value_ratio',      'value_ratio'),
    ]
    candidates = [c for c in candidates if c is not None]

    # Use a coarser sweep than the main table — 5 points is enough to
    # see the trend without printing 4×10=40 rows.
    diag_fracs = [f for f in fractions if f in (0.05, 0.10, 0.25, 0.50, 1.0)]
    if not diag_fracs:
        diag_fracs = [0.05, 0.10, 0.25, 0.50, 1.0]

    for name, col in candidates:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) < 100:
            print(f"\n  {name}: only {len(valid)} valid rows — skipping")
            continue

        print(f"\n  ── ranked by {name} (top-N% of horses by this signal) ──")
        print(f"     {'frac':>6} {'cutoff':>9} {'n':>6} "
              f"{'win%':>6} {'ROI':>8} {'+/-rand':>9}")
        print(f"     {'-'*6} {'-'*9} {'-'*6} "
              f"{'-'*6} {'-'*8} {'-'*9}")

        for frac in diag_fracs:
            if frac >= 1.0:
                cutoff = float(valid.min()) - 1.0
            else:
                cutoff = float(np.quantile(valid, 1.0 - frac))

            picks = df[df[col].notna() & (df[col] >= cutoff)]
            metrics = evaluate_system(picks, label=name, takeout=takeout)
            n = metrics['n_bets']
            roi = metrics['roi']
            wr = metrics['win_rate']

            wr_s  = f"{wr:.1%}"   if not np.isnan(wr)  else '   n/a'
            roi_s = f"{roi:+.1%}" if not np.isnan(roi) else '    n/a'
            cut_pct = roi - (-takeout) if not np.isnan(roi) else float('nan')
            cut_pct_s = f"{cut_pct:+.1%}" if not np.isnan(cut_pct) else '    n/a'

            tag = ''
            if n < 30:
                tag = '  ⚠ low n'
            print(f"     {frac:>6.2f} {cutoff:>9.3f} {n:>6d} "
                  f"{wr_s:>6} {roi_s:>8} {cut_pct_s:>9}{tag}")

    # Interpretation hints — surface what the four signals' patterns
    # collectively say about where the breakage is.
    print(f"\n  How to read this:")
    print(f"    • ensemble_prob anti-predictive → the LOGISTIC is broken.")
    print(f"      Investigate regression_analysis.py's CV log_loss.")
    print(f"    • predicted_payout anti-predictive (top picks paying LESS")
    print(f"      than bottom picks) → the PAYOUT MODEL has flipped sign.")
    print(f"      Re-check its CV Spearman; should be > 0.")
    print(f"    • Both individually fine but value_ratio still inverted →")
    print(f"      the MULTIPLICATION is the problem. Either P(win) and")
    print(f"      predicted_payout aren't on compatible scales, or they're")
    print(f"      negatively correlated in a way that pulls the product")
    print(f"      ranking toward overestimated longshots.")
    print(f"    • ml_prob_norm sweep is your FLOOR — if your model can't")
    print(f"      beat ranking by the public's morning-line odds, the")
    print(f"      model isn't adding value over the market.")


def main():
    parser = argparse.ArgumentParser(
        description='Threshold sweep + ROI report for predict_v5 EV-ratio rule')
    parser.add_argument('--takeout', type=float, default=WIN_TAKEOUT,
                        help=f'Win-pool takeout (default {WIN_TAKEOUT})')
    parser.add_argument('--fractions', type=str, default=None,
                        help='Comma-separated bet fractions to sweep, '
                             'e.g. "0.10,0.20,0.30". Default sweeps a '
                             'broad grid from 0.05 to 1.00.')
    parser.add_argument('--honest', action='store_true',
                        help='Use leave-one-card-out cutoff calibration. '
                             'Slower but the ROI numbers reflect '
                             'genuinely-out-of-sample threshold choice. '
                             'Recommended for final go/no-go decision.')
    parser.add_argument('--source', choices=['walk_forward', 'cv'],
                        default='walk_forward',
                        help='Which CSV to load. Default walk_forward — '
                             'has the OOS ensemble_prob from predict_v2.')
    parser.add_argument('--keep-missing-results', action='store_true',
                        help='Skip the auto-drop of cards that have no '
                             'results data in entries/. Default behavior '
                             'is to drop them because the loader zero-fills '
                             "missing payouts, biasing ROI downward — every "
                             'unmatched horse looks like a $6→$0 loss. Pass '
                             'this only if you know what you\'re doing.')
    parser.add_argument('--min-prob', type=float, default=None,
                        help='Pre-filter horses with ensemble_prob below '
                             'this threshold before sweeping. E.g. 0.20 '
                             'restricts to horses the logistic rates as '
                             '4/1 or shorter. Useful when the longshot '
                             'tail is dragging down value_ratio ranking. '
                             'Default None (no filter).')
    parser.add_argument('--no-decompose', action='store_true',
                        help='Skip the per-component diagnostic that ranks '
                             'horses by ensemble_prob, predicted_payout, and '
                             'ml_prob_norm separately. Default is to run it '
                             'because it tells you which signal is broken '
                             'when value_ratio looks anti-predictive.')
    args = parser.parse_args()

    # Parse fractions
    if args.fractions:
        try:
            fractions = [float(f.strip()) for f in args.fractions.split(',') if f.strip()]
        except ValueError as e:
            raise SystemExit(f"--fractions parse error: {e}")
        for f in fractions:
            if not (0 < f <= 1.0):
                raise SystemExit(f"Each fraction must be in (0, 1.0], got {f}")
    else:
        fractions = list(DEFAULT_FRACTIONS)

    print(f"Loading predictions...")
    df = load_cv_predictions(source=args.source)

    # backtest_kfold's loader has a fallback bug: when walk_forward_summary
    # has 'ml_win_prob_norm' (predict_v2's name) but not 'ml_prob_norm'
    # (regression's name), it copies cv_prob into ml_prob_norm as a
    # "diagnostic fallback." That makes the ml_prob_norm column identical
    # to ensemble_prob in the loaded df, which silently turns the component
    # diagnostic's "ml_prob_norm sweep" into a duplicate of the
    # "ensemble_prob sweep." Override here with the real ML column from
    # the source CSV so backtest_v5's diagnostics actually compare
    # blend-vs-market.
    if args.source == 'walk_forward':
        # Re-read just the columns we need so we can recover the real
        # ml_win_prob_norm. Cheaper than refactoring load_cv_predictions.
        wf_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
        if wf_path.exists():
            ml_raw = pd.read_csv(
                wf_path,
                usecols=lambda c: c in ('date', 'track', 'race',
                                        'horse_number', 'ml_win_prob_norm')
            )
            if 'ml_win_prob_norm' in ml_raw.columns:
                # Merge on the natural keys. Note: load_cv_predictions
                # normalizes track names (gulfstream-park → gulfstream),
                # so we apply the same normalization here before merging.
                from backtest_kfold import _merge_from_entries  # noqa: F401
                _TRACK_ALIASES = {
                    'gulfstream-park': 'gulfstream',
                    'gulfstream park': 'gulfstream',
                    'churchill-downs': 'churchill',
                    'churchill downs': 'churchill',
                }
                ml_raw['track'] = ml_raw['track'].astype(str).str.strip().str.lower()
                ml_raw['track'] = ml_raw['track'].replace(_TRACK_ALIASES)
                ml_raw['race'] = pd.to_numeric(ml_raw['race'], errors='coerce')
                ml_raw['horse_number'] = pd.to_numeric(
                    ml_raw['horse_number'], errors='coerce'
                )
                # Drop the loader's broken fallback first, then merge real ML
                if 'ml_prob_norm' in df.columns:
                    df = df.drop(columns=['ml_prob_norm'])
                df = df.merge(
                    ml_raw.rename(columns={'ml_win_prob_norm': 'ml_prob_norm'}),
                    on=['date', 'track', 'race', 'horse_number'],
                    how='left',
                )
                n_recovered = df['ml_prob_norm'].notna().sum()
                print(f"  Restored real ml_prob_norm from "
                      f"{wf_path.name}: {n_recovered}/{len(df)} rows.")

    if not args.keep_missing_results:
        df, skipped = filter_cards_without_results(df)
        if not df.empty and len(skipped) > 0:
            print(f"    Continuing with {df.groupby(['date', 'track']).ngroups} "
                  f"complete card(s).")
        if df.empty:
            raise SystemExit(
                "All cards filtered out — no results data found. "
                "Either backfill entries CSVs or pass --keep-missing-results "
                "to score the partial data anyway."
            )

    print(f"\nLoading payout model from {PAYOUT_MODEL_PATH}...")
    payout_model = load_payout_model(PAYOUT_MODEL_PATH)
    print(f"  trained on {payout_model.get('n_winners_train', '?')} winners, "
          f"CV Spearman={payout_model.get('cv_spearman', float('nan')):+.4f}")

    print(f"\nApplying payout model + computing value_ratio...")
    df = add_value_ratio(df, payout_model)

    if args.min_prob is not None:
        # Pre-sweep filter on ensemble_prob. Restricts the pool to horses
        # the logistic considers reasonably likely. Useful when the
        # diagnostic shows that the longshot tail (small ensemble_prob,
        # large predicted_payout) is dragging down value_ratio's ranking
        # quality — the conditional payout model can't distinguish
        # mispriced longshots from correctly-priced longshots, so it
        # blesses all of them, and the multiplication amplifies the
        # noise. Filter cuts out that tail before sweeping.
        before = len(df)
        before_with_ratio = df['value_ratio'].notna().sum()
        df = df[df['ensemble_prob'].notna() &
                (df['ensemble_prob'] >= args.min_prob)].copy()
        after = len(df)
        after_with_ratio = df['value_ratio'].notna().sum()
        print(f"\n  Pre-sweep filter: ensemble_prob >= {args.min_prob:.2f}")
        print(f"    rows: {before} → {after}  ({after/before:.1%} kept)")
        print(f"    rows with valid value_ratio: "
              f"{before_with_ratio} → {after_with_ratio}")
        if after_with_ratio < 100:
            raise SystemExit(
                f"Only {after_with_ratio} rows survive the prob filter — "
                f"too few for meaningful sweep. Lower --min-prob or "
                f"backfill more cards."
            )

    diagnose_value_ratio(df)

    if args.honest:
        print(f"\nRunning HONEST sweep (leave-one-card-out cutoffs)...")
        results = sweep_honest(df, fractions, takeout=args.takeout)
        mode = 'honest / leave-one-card-out'
    else:
        print(f"\nRunning IN-SAMPLE sweep "
              f"(cutoff set from same data being evaluated)...")
        print(f"  ⚠ ROI numbers are optimistic — same data sets cutoff and "
              f"is evaluated. Use --honest for trustworthy figures.")
        results = sweep_in_sample(df, fractions, takeout=args.takeout)
        mode = 'in-sample'

    print_sweep_table(results, takeout=args.takeout, mode_label=mode)

    if not args.no_decompose:
        decompose_signals(df, fractions, takeout=args.takeout)


if __name__ == '__main__':
    main()