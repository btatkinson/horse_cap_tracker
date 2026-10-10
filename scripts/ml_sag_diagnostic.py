"""
ml_sag_diagnostic.py

Test the hypothesis: LOG triggers on longshots because ML is structurally
loose on longshots. If true, closing odds on those horses are shorter
than ML, and LOG's "edge" evaporates at post time.

This script runs three checks using your existing walk_forward / CV data.
None require live odds feeds — all derive closing implied odds from
winners' payouts (win_paid_num / 2 ≈ closing decimal odds for winners).

The winners-only caveat is real and the script reports on it explicitly:
if winners drift systematically differently than losers, the diagnostic
under/overstates the ML-sag effect. You can't fix that without actual
tote data, but the direction of the effect is still informative.

Three checks:

  CHECK 1 — ML vs closing, by ML bucket
    For all winners, compare ml_decimal to closing_decimal. If ML-sag is
    real, winners with ML > 9 should have closing_decimal much shorter
    than ml_decimal. Compare 3.5-9.0 bucket as a control.

  CHECK 2 — Are LOG-triggered longshot winners paying less than ML?
    Same idea but restricted to LOG-triggered bets. If LOG is finding
    longshots the crowd also loves, their closing prices should be
    dramatically shorter than ML on average.

  CHECK 3 — Recompute edge using closing odds, see if LOG's "value" remains
    For each LOG-triggered longshot winner, compute what ev_per_2 *would*
    have been using closing implied prob instead of ml_prob. If the bet
    was genuinely value at closing, the edge stays positive. If it was
    ML-sag, the edge collapses or goes negative.

Usage:
    python ml_sag_diagnostic.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

# Reuse the loader from backtest_kfold so we get identical data
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


def derive_closing_decimal(df):
    """
    Recover closing decimal odds from win_paid_num.

    Parimutuel convention: win_paid is paid on a $2 ticket, so
        closing_decimal = win_paid_num / 2
    Only valid for winners (win_paid_num > 0). For non-winners we can't
    observe closing from payouts — we'd need tote data.

    Returns a Series; NaN for non-winners and rows with missing payouts.
    """
    df = df.copy()
    df['closing_decimal'] = np.where(
        (df.get('won', 0) == 1) & (df.get('win_paid_num', 0) > 0),
        df['win_paid_num'] / 2.0,
        np.nan,
    )
    df['closing_prob'] = np.where(
        df['closing_decimal'] > 0,
        1.0 / df['closing_decimal'],
        np.nan,
    )
    return df


def bucket_ml(ml_decimal):
    """Same buckets as backtest_kfold: chalk / lukewarm / mid / longshot."""
    if pd.isna(ml_decimal):
        return 'unknown'
    if ml_decimal <= 3.5:
        return '≤3.5 (deep chalk)'
    if ml_decimal <= 4.5:
        return '3.5-4.5 (lukewarm fave)'
    if ml_decimal <= 9.0:
        return '4.5-9.0 (mid)'
    return '>9.0 (longshot)'


def check_1_ml_vs_closing(df):
    """
    For all winners, show how ml_decimal compares to closing_decimal by
    ML bucket. Expected: longshots have closing << ML (crowd shortens
    them). Mid-range should be closer to ML.
    """
    print("="*78)
    print("CHECK 1 — ML vs closing price, all winners, by ML bucket")
    print("="*78)
    print("  If longshots get bet down, closing_decimal << ml_decimal there.")
    print("  A ratio of 1.0 means ML was accurate at closing.")
    print("  A ratio < 1.0 means the horse's final price was shorter than ML")
    print("  (crowd thought it was more likely to win than the handicapper did).")
    print()

    winners = df[df.get('won', 0) == 1].copy()
    winners = winners[winners['closing_decimal'].notna() & winners['ml_decimal'].notna()]

    if len(winners) == 0:
        print("  No winners with both ML and closing available. Cannot run check.")
        return

    winners['bucket'] = winners['ml_decimal'].apply(bucket_ml)
    winners['ratio'] = winners['closing_decimal'] / winners['ml_decimal']

    bucket_order = ['≤3.5 (deep chalk)', '3.5-4.5 (lukewarm fave)',
                    '4.5-9.0 (mid)', '>9.0 (longshot)']

    print(f"  {'bucket':<26} {'n':>4} {'ML med':>8} {'Close med':>10} "
          f"{'ratio med':>10} {'ratio mean':>11}")
    print(f"  {'-'*26} {'-'*4} {'-'*8} {'-'*10} {'-'*10} {'-'*11}")
    for b in bucket_order:
        sub = winners[winners['bucket'] == b]
        if len(sub) == 0:
            continue
        print(f"  {b:<26} {len(sub):>4} "
              f"{sub['ml_decimal'].median():>8.2f} "
              f"{sub['closing_decimal'].median():>10.2f} "
              f"{sub['ratio'].median():>10.2f} "
              f"{sub['ratio'].mean():>11.2f}")

    print()
    longshot = winners[winners['bucket'] == '>9.0 (longshot)']
    midrange = winners[winners['bucket'] == '4.5-9.0 (mid)']
    if len(longshot) >= 5 and len(midrange) >= 5:
        ls_ratio = longshot['ratio'].median()
        mr_ratio = midrange['ratio'].median()
        print(f"  Longshot winners' median closing/ML ratio: {ls_ratio:.2f}")
        print(f"  Mid-range winners' median closing/ML ratio:  {mr_ratio:.2f}")
        if ls_ratio < mr_ratio - 0.10:
            print(f"  → Longshots are getting bet down more than mid-range.")
            print(f"    Consistent with ML-sag hypothesis.")
        elif ls_ratio > mr_ratio + 0.10:
            print(f"  → Longshots are drifting longer, not shorter. "
                  f"ML-sag hypothesis NOT supported.")
        else:
            print(f"  → Similar drift in both buckets. Hypothesis inconclusive.")

    print()
    print("  CAVEAT: winners only. If losing longshots drift differently")
    print("  (e.g., drift *longer* because they attract no money), this")
    print("  overstates the sag effect. Interpret as directional, not quantitative.")


def check_2_log_longshot_closing(df):
    """
    Restrict to LOG-triggered winners. Do the longshots among them pay
    much less than ML? If yes, LOG is systematically picking crowd
    favorites disguised as longshots.
    """
    print()
    print("="*78)
    print("CHECK 2 — LOG-triggered winners: ML vs closing, by bucket")
    print("="*78)
    print("  If LOG picks longshots the crowd also loves, LOG-triggered")
    print("  longshot *winners* will close much shorter than ML.")
    print()

    if 'log_bet' not in df.columns:
        print("  log_bet column not present. Run compute_bet_flags first.")
        return

    log_winners = df[
        (df.get('won', 0) == 1) &
        (df['log_bet'] == True) &
        df['closing_decimal'].notna() &
        df['ml_decimal'].notna()
    ].copy()

    if len(log_winners) == 0:
        print("  No LOG-triggered winners with closing info. Skipping.")
        return

    log_winners['bucket'] = log_winners['ml_decimal'].apply(bucket_ml)
    log_winners['ratio'] = log_winners['closing_decimal'] / log_winners['ml_decimal']

    bucket_order = ['≤3.5 (deep chalk)', '3.5-4.5 (lukewarm fave)',
                    '4.5-9.0 (mid)', '>9.0 (longshot)']

    print(f"  {'bucket':<26} {'n':>4} {'ML med':>8} {'Close med':>10} "
          f"{'ratio med':>10}")
    print(f"  {'-'*26} {'-'*4} {'-'*8} {'-'*10} {'-'*10}")
    for b in bucket_order:
        sub = log_winners[log_winners['bucket'] == b]
        if len(sub) == 0:
            continue
        print(f"  {b:<26} {len(sub):>4} "
              f"{sub['ml_decimal'].median():>8.2f} "
              f"{sub['closing_decimal'].median():>10.2f} "
              f"{sub['ratio'].median():>10.2f}")

    ls = log_winners[log_winners['bucket'] == '>9.0 (longshot)']
    print()
    if len(ls) >= 3:
        print(f"  LOG-triggered longshot winners (n={len(ls)}):")
        print(f"    Median ML:      {ls['ml_decimal'].median():.1f}")
        print(f"    Median closing: {ls['closing_decimal'].median():.1f}")
        print(f"    Median ratio:   {ls['ratio'].median():.2f}")
        if ls['ratio'].median() < 0.7:
            print(f"  → These horses closed at ~"
                  f"{ls['ratio'].median()*100:.0f}% of their ML price.")
            print(f"    LOG's 'value' was gone by post time.")
    else:
        print(f"  Only {len(ls)} LOG-triggered longshot winners — too few to conclude.")


def check_3_recompute_edge_at_closing(df, takeout=WIN_TAKEOUT):
    """
    For LOG-triggered winners, recompute what ev_per_2 would have been
    using closing_prob instead of ml_prob. If the edge stays positive,
    LOG found real value. If it collapses, LOG was exploiting ML-sag.

        ev_roi_ml      = (cv_prob / ml_prob) * (1 - takeout) - 1
        ev_roi_closing = (cv_prob / closing_prob) * (1 - takeout) - 1
    """
    print()
    print("="*78)
    print("CHECK 3 — Edge re-estimated at closing price (LOG-triggered winners)")
    print("="*78)
    print("  For LOG-triggered winners, recompute ev_per_2 using closing_prob")
    print("  instead of ml_prob. If the real edge is at ML and not at closing,")
    print("  the 'edge at close' column will be much smaller — or negative.")
    print()

    if 'log_bet' not in df.columns:
        print("  log_bet column not present. Skipping.")
        return

    log_winners = df[
        (df.get('won', 0) == 1) &
        (df['log_bet'] == True) &
        df['closing_prob'].notna() &
        df['ml_decimal'].notna() &
        df['cv_prob'].notna()
    ].copy()

    if len(log_winners) == 0:
        print("  No LOG-triggered winners with closing info. Skipping.")
        return

    ml_prob = 1.0 / log_winners['ml_decimal']
    log_winners['ev_at_ml'] = 2 * ((log_winners['cv_prob'] / ml_prob) * (1 - takeout) - 1)
    log_winners['ev_at_closing'] = 2 * (
        (log_winners['cv_prob'] / log_winners['closing_prob']) * (1 - takeout) - 1
    )
    log_winners['ev_collapse'] = log_winners['ev_at_ml'] - log_winners['ev_at_closing']
    log_winners['bucket'] = log_winners['ml_decimal'].apply(bucket_ml)

    bucket_order = ['≤3.5 (deep chalk)', '3.5-4.5 (lukewarm fave)',
                    '4.5-9.0 (mid)', '>9.0 (longshot)']

    print(f"  {'bucket':<26} {'n':>4} {'ev@ML med':>10} "
          f"{'ev@close med':>13} {'collapse':>10}")
    print(f"  {'-'*26} {'-'*4} {'-'*10} {'-'*13} {'-'*10}")
    for b in bucket_order:
        sub = log_winners[log_winners['bucket'] == b]
        if len(sub) == 0:
            continue
        print(f"  {b:<26} {len(sub):>4} "
              f"{sub['ev_at_ml'].median():>+10.2f} "
              f"{sub['ev_at_closing'].median():>+13.2f} "
              f"{sub['ev_collapse'].median():>+10.2f}")

    ls = log_winners[log_winners['bucket'] == '>9.0 (longshot)']
    if len(ls) >= 3:
        print()
        pct_collapsed = (ls['ev_at_closing'] <= 0).mean()
        print(f"  Of {len(ls)} LOG-triggered longshot winners:")
        print(f"    {pct_collapsed*100:.0f}% had ev_at_closing ≤ 0")
        print(f"    (meaning: the edge was an ML artifact, not real value)")


def check_rdg_1_closing_vs_ml(df):
    """
    RDG SECTION — Check 1.

    Different mechanism than LOG. RDG's edge signal is `ridge_value_norm
    vs per-race break-even`, with NO ml in the formula. So we can't ask
    'is RDG fooled by ML' the way we did for LOG. The right question is:
    'does the crowd independently arrive at the same conclusion as RDG?'
    If yes, RDG is a slower version of public consensus and the edge
    evaporates by post time even though RDG never looked at ML.

    For RDG-triggered winners, compare ml_decimal to closing_decimal.
    Bucket by ML for comparability with the LOG checks. If RDG's
    longshot winners close MUCH shorter than ML (like LOG's did),
    that's evidence the crowd finds the same horses RDG does.

    NOTE: this is NOT the same metric as the LOG check. For LOG we
    collapsed an explicit edge formula. For RDG we're asking a separate
    question — 'do RDG picks become market consensus?' — using the same
    underlying ML-vs-closing comparison as a proxy for crowd agreement.
    Don't compare numbers across the LOG and RDG sections directly.
    """
    print()
    print("="*78)
    print("RDG SECTION — Check 1: Do RDG-triggered horses get bet down?")
    print("="*78)
    print("  RDG's edge formula doesn't use ML at all. But if the crowd")
    print("  independently picks the same horses RDG picks, the edge")
    print("  still evaporates by post time. We measure that here as")
    print("  closing_decimal / ml_decimal for RDG-triggered winners.")
    print()
    print("  Reading:")
    print("    ratio ≈ 1.0  → public didn't shorten this horse much")
    print("                   (RDG's pick was NOT public consensus)")
    print("    ratio << 1.0 → public agreed and bet it down")
    print("                   (RDG = slow crowd, edge gone at close)")
    print()

    if 'rdg_bet' not in df.columns:
        print("  rdg_bet column not present. Skipping.")
        return

    rdg_winners = df[
        (df.get('won', 0) == 1) &
        (df['rdg_bet'] == True) &
        df['closing_decimal'].notna() &
        df['ml_decimal'].notna()
    ].copy()

    if len(rdg_winners) == 0:
        print("  No RDG-triggered winners with closing info. Skipping.")
        return

    rdg_winners['bucket'] = rdg_winners['ml_decimal'].apply(bucket_ml)
    rdg_winners['ratio'] = rdg_winners['closing_decimal'] / rdg_winners['ml_decimal']

    bucket_order = ['≤3.5 (deep chalk)', '3.5-4.5 (lukewarm fave)',
                    '4.5-9.0 (mid)', '>9.0 (longshot)']

    print(f"  {'bucket':<26} {'n':>4} {'ML med':>8} {'Close med':>10} "
          f"{'ratio med':>10}")
    print(f"  {'-'*26} {'-'*4} {'-'*8} {'-'*10} {'-'*10}")
    for b in bucket_order:
        sub = rdg_winners[rdg_winners['bucket'] == b]
        if len(sub) == 0:
            continue
        print(f"  {b:<26} {len(sub):>4} "
              f"{sub['ml_decimal'].median():>8.2f} "
              f"{sub['closing_decimal'].median():>10.2f} "
              f"{sub['ratio'].median():>10.2f}")

    print()
    print("  → Compare these ratios to Check 2 (LOG-triggered) ratios.")
    print("    If RDG ratios are CLOSER to 1.0 than LOG ratios in the")
    print("    same bucket, RDG is finding horses the crowd is NOT")
    print("    converging on — i.e., a unique signal. If RDG ratios")
    print("    look the same as LOG ratios, RDG is just rediscovering")
    print("    what the crowd will eventually believe.")


def check_rdg_2_predicted_vs_realized_value(df):
    """
    RDG SECTION — Check 2.

    RDG predicts a value-shaped quantity (ridge_value_norm normalized to
    a per-race total). The question: for winners, how does RDG's
    predicted value compare to the realized winning payout?

    If RDG is well-calibrated against the closing market, predicted
    value tracks realized $win-paid reasonably. If predicted value
    is systematically much higher than realized payout, RDG is finding
    horses the closing market also likes (so the realized payout
    reflects shortened odds, not RDG's predicted opportunity).

    This is a calibration check, not an edge check. It tells us
    whether RDG's number-line means what it claims to mean.
    """
    print()
    print("="*78)
    print("RDG SECTION — Check 2: Predicted value vs realized payout (winners)")
    print("="*78)
    print("  RDG predicts a value-shaped per-horse number. For winners,")
    print("  we compare that predicted value to the realized win_paid.")
    print("  Both are in the same per-$2 ballpark, so if RDG is honest")
    print("  about closing market, the ratio realized/predicted should")
    print("  be near 1.0 on average.")
    print()
    print("  Ratio < 1.0 means: RDG predicted more value than the")
    print("  closing market actually delivered (crowd agreed and shortened).")
    print()

    if 'ridge_value_norm' not in df.columns:
        print("  ridge_value_norm column not present. Skipping.")
        return

    rdg_winners = df[
        (df.get('won', 0) == 1) &
        (df['rdg_bet'] == True) &
        df['ridge_value_norm'].notna() &
        df['win_paid_num'].notna() &
        (df['win_paid_num'] > 0)
    ].copy()

    if len(rdg_winners) == 0:
        print("  No RDG-triggered winners with ridge_value_norm. Skipping.")
        return

    rdg_winners['bucket'] = rdg_winners['ml_decimal'].apply(bucket_ml)
    # Both numbers are in $-on-$2 space conceptually, but ridge_value_norm
    # is a value (price-implied $payout), and win_paid_num is the actual
    # $payout for $2 staked. So the ratio is dimensionless.
    rdg_winners['ratio'] = rdg_winners['win_paid_num'] / rdg_winners['ridge_value_norm']

    bucket_order = ['≤3.5 (deep chalk)', '3.5-4.5 (lukewarm fave)',
                    '4.5-9.0 (mid)', '>9.0 (longshot)']

    print(f"  {'bucket':<26} {'n':>4} {'pred med':>9} {'real med':>9} "
          f"{'ratio med':>10}")
    print(f"  {'-'*26} {'-'*4} {'-'*9} {'-'*9} {'-'*10}")
    for b in bucket_order:
        sub = rdg_winners[rdg_winners['bucket'] == b]
        if len(sub) == 0:
            continue
        print(f"  {b:<26} {len(sub):>4} "
              f"{sub['ridge_value_norm'].median():>9.2f} "
              f"{sub['win_paid_num'].median():>9.2f} "
              f"{sub['ratio'].median():>10.2f}")

    print()
    print("  CAVEAT: same winners-only bias as the LOG checks. Winners")
    print("  are exactly the horses that closed shorter than they should")
    print("  have, so 'realized < predicted' is partially baked in.")
    print("  Treat ratio of ~0.7-0.9 as 'plausibly well-calibrated';")
    print("  ratios <0.5 in any bucket would be a real warning sign.")


def check_rdg_3_log_only_vs_log_plus_rdg_longshots(df):
    """
    RDG SECTION — Check 3 (the actionable one).

    The policy question: if we only bet LOG-triggered longshots when
    RDG ALSO triggers, does the picked-set look better?

    Compares two subsets of LOG-triggered longshot winners:
      A. LOG triggered, RDG did NOT  ('LOG-only longshots')
      B. LOG triggered AND RDG triggered  ('LOG+RDG longshots')

    For each subset, show closing/ML ratio and ev_at_closing. If subset
    B has closing ratios closer to 1.0 (i.e., didn't get bet down as
    hard) and less-negative ev_at_closing, then 'require RDG co-fire
    on longshots' is supported by data.

    If both subsets look equally bad, then RDG agreement doesn't help
    on longshots, and the right policy is just 'no LOG longshots
    period.'
    """
    print()
    print("="*78)
    print("RDG SECTION — Check 3: LOG-only vs LOG+RDG longshot winners")
    print("="*78)
    print("  Direct policy test: does requiring RDG to co-trigger on")
    print("  longshots filter out the bad LOG picks? Compares LOG-only")
    print("  (RDG silent) vs LOG+RDG (both triggered) for longshots.")
    print()

    if 'log_bet' not in df.columns or 'rdg_bet' not in df.columns:
        print("  log_bet or rdg_bet column missing. Skipping.")
        return

    log_winners = df[
        (df.get('won', 0) == 1) &
        (df['log_bet'] == True) &
        df['closing_decimal'].notna() &
        df['ml_decimal'].notna() &
        df['cv_prob'].notna()
    ].copy()

    if len(log_winners) == 0:
        print("  No LOG-triggered winners with closing info. Skipping.")
        return

    log_winners['bucket'] = log_winners['ml_decimal'].apply(bucket_ml)
    log_winners['ratio'] = log_winners['closing_decimal'] / log_winners['ml_decimal']
    log_winners['ev_at_closing'] = 2 * (
        (log_winners['cv_prob'] / log_winners['closing_prob']) * (1 - WIN_TAKEOUT) - 1
    )

    longshots = log_winners[log_winners['bucket'] == '>9.0 (longshot)']
    if len(longshots) == 0:
        print("  No LOG-triggered longshot winners. Skipping.")
        return

    log_only = longshots[longshots['rdg_bet'] != True]
    log_plus_rdg = longshots[longshots['rdg_bet'] == True]

    print(f"  Subset                   n   close/ML med   ev@close med   pct ev≤0")
    print(f"  {'-'*22}  {'-'*3}  {'-'*12}   {'-'*12}   {'-'*8}")

    for label, sub in [("LOG only (RDG silent)", log_only),
                       ("LOG + RDG agreement",   log_plus_rdg)]:
        if len(sub) == 0:
            print(f"  {label:<22}  {0:>3}   (no rows)")
            continue
        ratio_med = sub['ratio'].median()
        ev_med = sub['ev_at_closing'].median()
        pct_neg = (sub['ev_at_closing'] <= 0).mean()
        print(f"  {label:<22}  {len(sub):>3}   "
              f"{ratio_med:>12.2f}   "
              f"{ev_med:>+12.2f}   "
              f"{pct_neg*100:>7.0f}%")

    print()
    print("  Reading:")
    print("    If LOG+RDG row has ratio closer to 1.0 and less-negative")
    print("    ev@close than LOG-only, RDG agreement filters out the")
    print("    crowd-favorite trap → 'require RDG on longshots' is good.")
    print()
    print("    If both rows look equally bad, RDG doesn't help filter")
    print("    longshots → just disable LOG-only longshot triggers.")
    print()
    print("    Sample sizes here are small. Treat as directional.")


def check_4_count_by_trigger_in_longshot(df):
    """
    Bonus context: how many bets actually fall in each (bucket, trigger)
    cell? Useful because the ROI table in backtest_kfold aggregates across
    trigger combinations.
    """
    print()
    print("="*78)
    print("CHECK 4 — Bet count by (bucket, trigger) for all predict_v2 bets")
    print("="*78)
    print("  Context: how many bets live in each trigger/bucket combination?")
    print()

    bets = df[df.get('any_bet', False) == True].copy()
    if len(bets) == 0:
        print("  No bets marked. Skipping.")
        return

    bets['bucket'] = bets['ml_decimal'].apply(bucket_ml)

    triggers = []
    for _, row in bets.iterrows():
        flags = []
        if row.get('log_bet', False):
            flags.append('LOG')
        if row.get('rdg_bet', False):
            flags.append('RDG')
        if row.get('raw_bet', False):
            flags.append('RAW')
        triggers.append('+'.join(flags) if flags else '(none)')
    bets['triggered_by'] = triggers

    tab = pd.crosstab(bets['triggered_by'], bets['bucket'])
    print(tab.to_string())


def main():
    print("Loading predictions (same pipeline as backtest_kfold)...")
    print()
    df = load_cv_predictions(source=None)
    df = compute_ev_per_2(df, takeout=WIN_TAKEOUT)

    if 'ridge_value' in df.columns:
        df = apply_ridge_race_norm(
            df, 'ridge_value', RIDGE_RACE_TOTAL, RIDGE_FLOOR
        )

    raw_mean, raw_std = compute_raw_calibration(df, 'raw_ridge_value')
    df = compute_bet_flags(df, raw_mean, raw_std)

    df = derive_closing_decimal(df)

    n_winners_with_closing = (
        (df.get('won', 0) == 1) & df['closing_decimal'].notna()
    ).sum()
    print()
    print(f"Recovered closing odds for {n_winners_with_closing} winners "
          f"(via win_paid_num / 2)")
    print()

    print()
    print("##############################################################################")
    print("# LOG SECTION — does ML-sag corrupt LOG's edge calculation?")
    print("##############################################################################")
    check_1_ml_vs_closing(df)
    check_2_log_longshot_closing(df)
    check_3_recompute_edge_at_closing(df, takeout=WIN_TAKEOUT)

    print()
    print("##############################################################################")
    print("# RDG SECTION — does the crowd independently arrive at RDG's picks?")
    print("# (Different mechanism. Don't compare numbers across LOG/RDG sections.)")
    print("##############################################################################")
    check_rdg_1_closing_vs_ml(df)
    check_rdg_2_predicted_vs_realized_value(df)
    check_rdg_3_log_only_vs_log_plus_rdg_longshots(df)

    print()
    print("##############################################################################")
    print("# CONTEXT")
    print("##############################################################################")
    check_4_count_by_trigger_in_longshot(df)

    print()
    print("="*78)
    print("INTERPRETATION GUIDE")
    print("="*78)
    print("""
  LOG section reads:
    Check 1 longshot ratio << 1.0 + Check 2 LOG-longshot ratio even
    smaller + Check 3 ev@close collapses → ML-sag corrupts LOG's edge
    on longshots. Already confirmed in your last run.

  RDG section reads:
    RDG Check 1 ratios near 1.0 → RDG picks are NOT public consensus,
      so RDG's edge is genuine and survives to closing. Strong RDG
      signal, worth co-trigger gating LOG with.
    RDG Check 1 ratios as low as LOG's → RDG is just slow-crowd, the
      'doesn't use ML' argument is structural but not predictive. In
      that case RDG is also overbet at close.

    RDG Check 2 ratios in 0.7-1.0 range → RDG well-calibrated against
      closing market (winners-bias accounts for the gap from 1.0).
    RDG Check 2 ratios <0.5 → RDG predicting much more value than
      closing market delivers. Predicted value is fictional.

    RDG Check 3 LOG+RDG row much better than LOG-only row → 'require
      RDG co-fire on longshots' is the right fix. Filters the bad
      LOG picks.
    RDG Check 3 both rows equally bad → RDG agreement doesn't help
      longshots. Just disable LOG-only longshot triggers entirely.

  Action paths regardless of RDG outcome:
    The LOG-only longshot finding is already actionable. You can
    disable LOG-only triggers above some ML threshold today, no
    closing-line scraping required. RDG section just tells you
    whether 'LOG + RDG co-fire' is a less-aggressive alternative.
""")


if __name__ == '__main__':
    main()
