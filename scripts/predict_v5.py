"""
predict_v5.py

Two-model EV-ratio bet selector. Replaces v4's three-way corner geometry
with a direct expected-value calculation:

    value_ratio = P(win) * predicted_payout / 2

where:
    P(win)            = predict_v2's ensemble_prob (logistic on AI signals)
    predicted_payout  = Model B's prediction of win_paid CONDITIONAL on
                        winning (trained on winners only — see
                        payout_model.py for rationale)

A horse is BET if its value_ratio meets or exceeds the (1 - bet_fraction)
quantile of historical value_ratios from walk_forward_summary.csv.
With bet_fraction=0.25 (default), that's q75: bet only the top 25% of
horses across the historical sample.

Why ratio rather than EV-per-$2 dollars?
----------------------------------------
ev_per_2 = 2 * value_ratio - 2, so they rank-order identically. The
ratio is bounded near 1.0 = "break even" which makes the cutoff
interpretable. value_ratio = 1.10 means "bet $2 to expect $2.20 back"
i.e. a 10% edge. value_ratio = 1.00 is break-even before takeout.

Why historical cutoff rather than a fixed value_ratio threshold?
----------------------------------------------------------------
A fixed threshold (like "bet anything ≥ 1.05") drifts as model
calibration drifts. The historical cutoff approach matches v4's: anchor
to the backtest distribution, accept that hot/cool model days produce
more/fewer bets than the target rate. Trade-off: when live models run
hot, you bet more horses (and may overrate them); when cool, fewer
(and may pass on real edges).

Drift caveat
------------
Same as v4. If the live models are consistently miscalibrated relative
to backtest, the bet count skews. Re-run with --recompute-thresholds
when walk_forward_summary.csv gets refreshed (new cards added, models
retrained, payout model retrained).

Pre-requisites
--------------
1. walk_forward_summary.csv must exist with the following columns:
       date, track, race, horse_number, ensemble_prob, win_paid_num
   plus all the AI feature columns the payout model expects (claude_prob,
   gemini_prob, ml_prob_norm, etc.). predict_v2 writes its full df per
   card and backtest_walk_forward concatenates them, so any column
   predict_v2 produces will be there.

   ensemble_prob is the post-blend logistic + LLM probability that
   predict_v2 produces. It's already genuinely out-of-sample under the
   walk-forward harness because backtest_walk_forward retrains the
   logistic with the holdout date excluded before running predict_v2.

2. payout_model.json must exist. Generate via regression_analysis.py
   (now also fits Model B), or directly via payout_model.train_payout_model.

Threshold calibration
---------------------
Persisted to v5_thresholds.json:
    {
      "value_ratio_cutoff":         ...,    # (1 - bet_fraction) quantile
      "bet_fraction_calibrated_at": 0.25,
      "n_calibration_rows":         ...,
      "calibration_summary": {
          "median_ratio":  ...,
          "mean_ratio":    ...,
          "p99_ratio":     ...,
      },
      "calibrated_at": "YYYY-MM-DD HH:MM"
    }

Compared to v4
--------------
v4: agreement detector — corner score is high when 3 models all like a
    horse a lot. Gates on "do my models agree?"
v5: value detector — the score IS expected value. Gates on "is this a
    +EV bet?" — the question you actually want to ask.

v4 bets cluster around horses where Claude/Gemini/raw-ridge all converge.
That's good for confidence but doesn't directly target overlays. v5 bets
cluster around horses where the logistic finds them more likely than the
public's odds suggest, scaled by what those odds pay out.

Usage
-----
    python predict_v5.py --date 2026-04-30 --track churchill-downs
    python predict_v5.py --date 2026-04-30 --track churchill-downs --bet-fraction 0.15
    python predict_v5.py --date 2026-04-30 --track churchill-downs --recompute-thresholds
"""

import os
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])

import predict_v2 as p2
from payout_model import load_payout_model, predict_payout

# =============================================================================
# CONFIG
# =============================================================================

# Default fraction of the historical sample to bet, i.e. cutoff = q(1-frac).
# 0.25 = bet anything in the top 25% of historical value_ratio.
DEFAULT_BET_FRACTION = 0.25

# Flat bet size in dollars.
FLAT_BET = 2.0

# Persistence paths
THRESHOLDS_PATH = DATA_PATH / 'predictions' / 'v5_thresholds.json'
PAYOUT_MODEL_PATH = DATA_PATH / 'payout_model.json'
WALK_FORWARD_PATH = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'

# Bet log tag — keeps v5 bets separable from v3/v4 in post-hoc analysis.
GATE_AXIS_TAG = 'ev_ratio_v5'


# =============================================================================
# 1. CALIBRATION — historical value_ratio quantile cutoff
# =============================================================================

def _compute_value_ratio(prob, payout, bet_size=FLAT_BET):
    """value_ratio = prob * payout / bet_size. Vectorized.

    Returns NaN where either input is NaN (excluded from cutoff and
    from live betting). Negative payouts shouldn't happen but get
    passed through; downstream code clamps.
    """
    prob = np.asarray(prob, dtype=float)
    payout = np.asarray(payout, dtype=float)
    return prob * payout / bet_size


def calibrate_thresholds(bet_fraction):
    """Compute and persist the v5 value_ratio cutoff from
    walk_forward_summary.csv.

    Pulls ensemble_prob (predict_v2's blended logistic + LLM probability,
    which is OOS under the walk-forward harness) and derives the payout
    prediction on the fly by applying the saved payout_model.json to
    walk_forward's feature columns.

    Why derive on the fly: backtest_walk_forward.py may not have been
    rerun after the payout model was added. Rather than force a
    backtest rerun (which is slow), apply the production payout model
    to the historical features directly. This ISN'T proper walk-forward
    for the payout model — the payout model was trained on the same
    cards we're now scoring — but for the purpose of setting a quantile
    cutoff on the value_ratio distribution, it's fine. The cutoff just
    needs to reflect what production will produce, not be a clean
    out-of-sample backtest itself.

    If you want a fully clean walk-forward backtest of v5, modify
    backtest_walk_forward.py to (a) refit the payout model excluding
    the held-out date, then (b) score that date with both models.
    That's strictly more correct but slow; the on-the-fly approach
    here is a reasonable shortcut for cutoff calibration.
    """
    if not WALK_FORWARD_PATH.exists():
        raise RuntimeError(
            f"walk_forward_summary.csv not found at {WALK_FORWARD_PATH}. "
            f"Run backtest_walk_forward.py first."
        )
    if not PAYOUT_MODEL_PATH.exists():
        raise RuntimeError(
            f"payout_model.json not found at {PAYOUT_MODEL_PATH}. "
            f"Run regression_analysis.py to fit Model B (or call "
            f"payout_model.train_payout_model directly)."
        )

    df = pd.read_csv(WALK_FORWARD_PATH)
    if 'ensemble_prob' not in df.columns:
        raise RuntimeError(
            f"walk_forward_summary.csv missing 'ensemble_prob' column. "
            f"This is predict_v2's blended P(win) — re-run backtest_walk_forward.py "
            f"so each holdout's predictions.csv gets concatenated into the summary."
        )

    payout_model = load_payout_model(PAYOUT_MODEL_PATH)

    # predict_payout resolves both regression-side ('claude_prob') and
    # predict_v2-side ('claude_win_prob') schemas, and per-race normalizes
    # if the df came from predict_v2. Lets us calibrate from the same CSV
    # that predict_v2 produced without an intermediate transformation step.
    df = predict_payout(df, payout_model)

    # Compute value_ratio on the historical sample. Drop NaN — horses
    # missing inputs aren't relevant for setting the cutoff.
    df['value_ratio'] = _compute_value_ratio(df['ensemble_prob'], df['predicted_payout'])
    sample = df['value_ratio'].dropna()
    n_full = len(sample)
    if n_full < 200:
        raise RuntimeError(
            f"Only {n_full} rows produced a valid value_ratio in "
            f"walk_forward_summary.csv. Need 200+ for a stable quantile. "
            f"Backfill more cards or check for missing AI predictions."
        )

    cutoff = float(np.quantile(sample, 1.0 - bet_fraction))

    payload = {
        'value_ratio_cutoff': cutoff,
        'bet_fraction_calibrated_at': bet_fraction,
        'n_calibration_rows': n_full,
        'calibration_summary': {
            # These are dollar-meaning numbers the user actually wants
            # to see when calibration runs. value_ratio of 1.0 is
            # break-even, 1.10 is a 10% edge, etc.
            'median_ratio':  float(np.median(sample)),
            'mean_ratio':    float(np.mean(sample)),
            'p90_ratio':     float(np.quantile(sample, 0.90)),
            'p99_ratio':     float(np.quantile(sample, 0.99)),
            'pct_above_1.0': float((sample >= 1.0).mean()),
        },
        'payout_model_calibrated_at':
            payout_model.get('calibrated_at', 'unknown'),
        'calibrated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def load_thresholds(bet_fraction, recompute=False):
    """Return calibration payload, recomputing if missing/stale/mismatched."""
    required = {'value_ratio_cutoff', 'bet_fraction_calibrated_at'}
    if recompute or not THRESHOLDS_PATH.exists():
        return calibrate_thresholds(bet_fraction)
    try:
        payload = json.loads(THRESHOLDS_PATH.read_text())
    except json.JSONDecodeError:
        return calibrate_thresholds(bet_fraction)
    if not required.issubset(payload.keys()):
        return calibrate_thresholds(bet_fraction)
    # Cutoff is bet_fraction-dependent; recompute on mismatch.
    if payload['bet_fraction_calibrated_at'] != bet_fraction:
        return calibrate_thresholds(bet_fraction)
    return payload


# =============================================================================
# 2. SCORING — apply payout model + compute value_ratio on live predictions
# =============================================================================

def add_value_ratio_columns(df, payout_model):
    """Add predicted_payout and value_ratio columns to a live df.

    df comes from predict_v2's pipeline post-merge, so it has
    ensemble_prob and the raw feature columns the payout model needs.
    """
    df = predict_payout(df, payout_model)
    df['value_ratio'] = _compute_value_ratio(
        df['ensemble_prob'].values,
        df['predicted_payout'].values,
    )
    # Useful display column. ev_per_2 = 2*ratio - 2, so the two carry
    # the same ranking but ev_per_2 is in dollar terms.
    df['ev_per_2_v5'] = 2.0 * df['value_ratio'] - 2.0
    return df


# =============================================================================
# 3. BET GENERATION — replaces v2's generate_bets via shim
# =============================================================================

def generate_bets_v5(df, calib, payout_model, takeout=None):
    """Build bet records using the historical-cutoff EV-ratio rule.

    Mirrors v2/v4 generate_bets shape: (bet_df, race_summaries dict).
    Bets every horse with value_ratio >= calib['value_ratio_cutoff'].
    NaN ratios (missing inputs or missing payout prediction) are excluded.

    Note: this rule does NOT cap bets at one per race. If two horses in
    the same race both clear the cutoff, both get bet — they're both
    individually +EV (according to the model) so the bet rule treats
    them independently. That's a deliberate departure from "pick the
    favorite of the race" handicapping. If you want one-bet-per-race,
    add a post-filter.
    """
    df = add_value_ratio_columns(df.copy(), payout_model)
    cutoff = calib['value_ratio_cutoff']

    qualifies = df['value_ratio'].notna() & (df['value_ratio'] >= cutoff)

    bets = []
    race_summaries = {}

    for race_num, race_df in df.groupby('race'):
        race_qualifiers = race_df[qualifies.loc[race_df.index]]
        race_top = (float(race_df['value_ratio'].max())
                    if race_df['value_ratio'].notna().any() else float('nan'))

        if race_qualifiers.empty:
            race_summaries[int(race_num)] = {
                'action': 'PASS',
                'top_ratio': race_top,
                'cutoff': cutoff,
            }
            continue

        for _, row in race_qualifiers.iterrows():
            hn = int(row['horse_number'])
            prob = float(row['ensemble_prob']) if pd.notna(row['ensemble_prob']) else np.nan
            ratio = float(row['value_ratio'])
            payout = float(row['predicted_payout']) if pd.notna(row['predicted_payout']) else np.nan
            bet_record = {
                'race': int(race_num),
                'horse_number': hn,
                'bet_type': 'WPS',  # match v2/v4 default
                'ensemble_prob':       round(prob, 4) if pd.notna(prob) else np.nan,
                'ml_prob':             round(row.get('ml_win_prob_norm', np.nan), 4),
                'predicted_payout':    round(payout, 2) if pd.notna(payout) else np.nan,
                'value_ratio':         round(ratio, 4),
                'ev_per_2_v5':         round(float(row['ev_per_2_v5']), 2),
                # Carry v2's existing ev_per_2 too — it's computed from
                # ensemble_prob/ml_prob and is useful for cross-checking
                # whether the payout model is buying us anything beyond
                # the public's odds.
                'ev_per_2_logistic':   round(row.get('ev_per_2', np.nan), 2),
                'ml_odds':             row.get('ml_odds_raw', ''),
                # v5 fields
                'value_ratio_cutoff':  round(cutoff, 4),
                'bet_fraction':        calib['bet_fraction_calibrated_at'],
                # Compatibility / metadata
                'triggered_by':        'EV_RATIO_V5',
                'gate_axis':           GATE_AXIS_TAG,
                'kelly':               0.0,  # not computed (flat bet)
                'bet_amount':          FLAT_BET,
            }
            bets.append(bet_record)

        race_summaries[int(race_num)] = {
            'action': 'BET',
            'n_bet': len(race_qualifiers),
            'horses': [int(h) for h in race_qualifiers['horse_number']],
            'cutoff': cutoff,
            'top_ratio': race_top,
        }

    bet_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    return bet_df, race_summaries


# =============================================================================
# 4. SHIM — plug into predict_v2.run_predictions via monkey-patch
# =============================================================================

# Module-level state, same pattern as v3/v4. See predict_v3.py for why.
_LAST_BET_DF = None
_LAST_FULL_DF = None
_RACE_BANNER_EXTRAS = {}
_CALIB = None
_PAYOUT_MODEL = None


def _generate_bets_shim(df, **kwargs):
    """Adapter for predict_v2.run_predictions."""
    global _LAST_BET_DF, _LAST_FULL_DF

    if _CALIB is None or _PAYOUT_MODEL is None:
        raise RuntimeError(
            "v5 not initialized — run_predictions_v5 must set _CALIB and "
            "_PAYOUT_MODEL before invoking p2.run_predictions."
        )

    takeout = kwargs.get('takeout', getattr(p2, 'WIN_TAKEOUT', None))
    bet_df, race_summaries = generate_bets_v5(
        df, _CALIB, _PAYOUT_MODEL, takeout=takeout
    )

    _LAST_BET_DF = bet_df
    _LAST_FULL_DF = df

    _RACE_BANNER_EXTRAS.clear()
    for race_num, summary in race_summaries.items():
        top    = summary.get('top_ratio', float('nan'))
        cutoff = summary.get('cutoff', float('nan'))
        n_bet  = summary.get('n_bet', 0)
        top_str    = f"{top:.3f}"    if not np.isnan(top)    else "n/a"
        cutoff_str = f"{cutoff:.3f}" if not np.isnan(cutoff) else "n/a"
        _RACE_BANNER_EXTRAS[int(race_num)] = (
            f"v5: top ratio={top_str} (cutoff={cutoff_str}, {n_bet} bet)"
        )

    return bet_df, race_summaries


# =============================================================================
# 5. ENTRY POINT
# =============================================================================

def run_predictions_v5(date_str, track, num_sims=100000, use_regression=False,
                       recompute_thresholds=False,
                       bet_fraction=DEFAULT_BET_FRACTION,
                       skip_tipmeerkat=True):
    """v2's full pipeline with the v5 EV-ratio rule swapped in for bet generation.

    skip_tipmeerkat: forwarded to predict_v2.run_predictions; excludes
        tm_* expert CSVs from the human_pool feature for THIS run.
        Default True because tipmeerkat data is currently bugged.
        Pass False (via --include-tipmeerkat in main) to opt back in.
    """
    global _CALIB, _PAYOUT_MODEL

    if not (0.0 < bet_fraction < 1.0):
        raise ValueError(f"bet_fraction must be in (0, 1), got {bet_fraction}")

    # Load both pieces up front so we fail fast.
    _PAYOUT_MODEL = load_payout_model(PAYOUT_MODEL_PATH)
    _CALIB = load_thresholds(bet_fraction=bet_fraction,
                             recompute=recompute_thresholds)

    print(f"\n{'='*72}")
    print(f"PREDICT V5 — EV-ratio bet selector (P(win) × predicted_payout / 2)")
    print(f"{'='*72}")
    print(f"  Payout model: {PAYOUT_MODEL_PATH.name}")
    print(f"    n winners trained on: {_PAYOUT_MODEL.get('n_winners_train', '?')}")
    print(f"    CV RMSE in $ space:   ${_PAYOUT_MODEL.get('cv_rmse_dollars', float('nan')):.2f}")
    print(f"    CV Spearman (winners): {_PAYOUT_MODEL.get('cv_spearman', float('nan')):+.4f}")
    print(f"    calibrated at:        {_PAYOUT_MODEL.get('calibrated_at', '?')}")
    print(f"  Threshold calibration: {THRESHOLDS_PATH.name}")
    print(f"    n calibration rows:  {_CALIB['n_calibration_rows']}")
    print(f"    median ratio:        {_CALIB['calibration_summary']['median_ratio']:.3f}")
    print(f"    p90 ratio:           {_CALIB['calibration_summary']['p90_ratio']:.3f}")
    print(f"    pct above breakeven: "
          f"{_CALIB['calibration_summary']['pct_above_1.0']:.1%}")
    pct = bet_fraction * 100
    print(f"  Bet rule: value_ratio >= q{100-pct:.0f} of history "
          f"(= {_CALIB['value_ratio_cutoff']:.3f})")
    print(f"            target ~{pct:.0f}% bet rate, flat ${FLAT_BET:.0f} per bet")
    print(f"  Reminder: ratio of 1.0 is break-even on a $2 bet. Cutoff "
          f"{_CALIB['value_ratio_cutoff']:.3f} ≈ "
          f"{(_CALIB['value_ratio_cutoff'] - 1.0) * 100:+.1f}% edge required.")
    print()

    # Monkey-patch v2.generate_bets, run, restore.
    original_generate_bets = p2.generate_bets
    p2.generate_bets = _generate_bets_shim
    try:
        p2.run_predictions(
            date_str, track, num_sims=num_sims,
            use_regression=use_regression,
            race_banner_extras=_RACE_BANNER_EXTRAS,
            suppress_trigger_explanation=True,
            skip_tipmeerkat=skip_tipmeerkat,
        )
    finally:
        p2.generate_bets = original_generate_bets

    # Post-run summary
    if _LAST_BET_DF is not None and not _LAST_BET_DF.empty:
        n = len(_LAST_BET_DF)
        races_with_bet = _LAST_BET_DF['race'].nunique()
        cutoff = _LAST_BET_DF['value_ratio_cutoff'].iloc[0]
        avg_ratio = _LAST_BET_DF['value_ratio'].mean()
        avg_payout = _LAST_BET_DF['predicted_payout'].mean()
        avg_prob = _LAST_BET_DF['ensemble_prob'].mean()
        print(f"\n  v5 EV-ratio summary:")
        print(f"    {n} bet(s) across {races_with_bet} race(s)")
        print(f"    cutoff: {cutoff:.3f}  ({(cutoff-1.0)*100:+.1f}% edge)")
        print(f"    avg ratio on bets:   {avg_ratio:.3f}  "
              f"({(avg_ratio-1.0)*100:+.1f}% edge)")
        print(f"    avg P(win) on bets:  {avg_prob:.1%}")
        print(f"    avg pred payout:     ${avg_payout:.2f}")
    else:
        print(f"\n  v5 EV-ratio: no bets fired on this card.")
        print(f"    No horse cleared the value_ratio cutoff "
              f"({_CALIB['value_ratio_cutoff']:.3f}).")
        print(f"    The day's models may be running cooler than backtest history.")
        print(f"    Options: (1) re-run with --recompute-thresholds if your")
        print(f"    walk_forward_summary.csv has been refreshed, (2) raise")
        print(f"    --bet-fraction to be more permissive (e.g. 0.40), or")
        print(f"    (3) accept the pass — that's what the rule prescribes.")


def main():
    parser = argparse.ArgumentParser(
        description='predict_v5 — EV-ratio bet selector '
                    '(P(win) × predicted_payout / 2 vs historical cutoff)'
    )
    parser.add_argument('--date',  required=True, help='YYYY-MM-DD')
    parser.add_argument('--track', required=True, help='e.g. churchill-downs')
    parser.add_argument('--num-sims', type=int, default=100000)
    parser.add_argument('--use-regression', action='store_true')
    parser.add_argument('--bet-fraction', type=float, default=DEFAULT_BET_FRACTION,
                        help=f'historical-percentile fraction. '
                             f'0.25 = bet horses with value_ratio >= q75 of '
                             f'historical ratios. Default {DEFAULT_BET_FRACTION}.')
    parser.add_argument('--recompute-thresholds', action='store_true',
                        help='Refresh v5_thresholds.json from current '
                             'walk_forward_summary.csv before running')
    parser.add_argument('--include-tipmeerkat', action='store_true',
                        help='Include tipmeerkat (tm_*) experts in the '
                             'human_pool feature. NOT recommended — '
                             'tipmeerkat data is currently bugged. Default '
                             'is to exclude them.')
    args = parser.parse_args()

    # Default is to skip tipmeerkat (its data is bugged); --include-tipmeerkat
    # opts back in. Same pattern as predict_v3.
    skip_tipmeerkat = not args.include_tipmeerkat
    if args.include_tipmeerkat:
        print("!" * 60)
        print("!!  --include-tipmeerkat active: tipmeerkat (tm_*) experts WILL be used.")
        print("!!  tipmeerkat data is currently bugged. Recommend running without it.")
        print("!" * 60)

    run_predictions_v5(
        args.date, args.track,
        num_sims=args.num_sims,
        use_regression=args.use_regression,
        recompute_thresholds=args.recompute_thresholds,
        bet_fraction=args.bet_fraction,
        skip_tipmeerkat=skip_tipmeerkat,
    )


if __name__ == '__main__':
    main()