"""
pick3_play_skip.py

Pick 3 play/skip analysis. Builds on pick3_greedy_backtest findings:
- greedy_value at min_ridge=1.33 was the best-performing greedy bin
- greedy_value_strict at min_ridge=2.5 was the strict-EV alternative
- Both are still negative-ROI when forced to bet every sequence

This script asks: of 135 sequences, which subset should you ACTUALLY
play? It computes several pre-race indicators per sequence, bins them
into quintiles, and reports realized ROI per bin. If any indicator has
a top bin with ROI better than -22% (random Pick 3 takeout baseline)
or ideally positive, that indicator is a useful play/skip signal.

Also surfaces:
- Card-level concentration: is the strategy's profit dominated by a
  few lucky cards? Important because consecutive Pick 3 sequences
  share races (R1-R3 and R2-R4 share two legs), inflating apparent
  sample size.
- Per-quintile sample-size and hit-count warnings.

Strategies tested (both at once, side-by-side):
  greedy_value @ min_ridge=1.33  (loose threshold, modest tickets)
  greedy_value_strict @ min_ridge=2.5  (strict, only EV-positive horses)

Pre-race indicators tested:
  ticket_cost           — lower = more concentrated opinion
  hit_prob              — model's predicted cash rate
  seed_ridge_geomean    — geometric mean of seed horses' ridge_value
                          (higher = stronger anchor opinion overall)
  seed_ridge_min        — weakest seed leg (the "limiting reagent")
  top2_ridge_gap_geomean — geomean of (top ridge - 2nd ridge) per leg
                          (high = clear-favorite races, narrow tickets)
  n_trigger_legs        — count of legs where ANY predict_v2 trigger fired
                          (LOG / RDG / RAW bet flag)

Caveat: 135 sequences / 5 quintiles = 27 per bin. With Pick 3 hit rates
of 20-40% in this strategy range, that's ~5-12 hits per bin. A single
~$200 payout swings a bin's ROI by ~5-7 points. Treat directional
findings as suggestive; only consistent monotonic patterns are reliable.

Today-mode usage:
    python pick3_play_skip.py --card 2026-04-25 --track churchill-downs --races 4-6

This computes today's indicators and reports which historical quintile
each falls in. Doesn't auto-decide; just supports the decision.
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

# Match predict_v2's trigger logic for n_trigger_legs computation.
# These are the constants from predict_v2.py at the time of writing.
LOGISTIC_EDGE_FLOOR = 0.065  # MIN_EDGE — LOG bet trigger on ev_per_2
RIDGE_EDGE_MULTIPLIER = 1.05
RIDGE_RACE_TOTAL = 18.3
TAKEOUT_GROSS = 1.0  # ridge_break_even = ridge_raw_fair * takeout_gross
RAW_EDGE_K = 1.0


# =============================================================================
# 1. LOADING (shared with pick3_backtest)
# =============================================================================

def discover_cards():
    pred_root = DATA_PATH / 'predictions'
    if not pred_root.exists():
        return []
    cards = []
    for date_dir in pred_root.iterdir():
        if not date_dir.is_dir() or not _looks_like_date(date_dir.name):
            continue
        date_str = date_dir.name
        for track_dir in date_dir.iterdir():
            if not track_dir.is_dir():
                continue
            pred = track_dir / 'predictions.csv'
            entries = DATA_PATH / 'entries' / f'{track_dir.name}_{date_str}.csv'
            meta = DATA_PATH / 'meta' / f'{track_dir.name}_{date_str}.csv'
            if pred.exists() and entries.exists() and meta.exists():
                cards.append((date_str, track_dir.name))
    return sorted(cards)


def _looks_like_date(s):
    try:
        pd.to_datetime(s, format='%Y-%m-%d')
        return True
    except (ValueError, TypeError):
        return False


def load_card(date_str, track):
    pred = pd.read_csv(DATA_PATH / 'predictions' / date_str / track / 'predictions.csv')
    entries = pd.read_csv(DATA_PATH / 'entries' / f'{track}_{date_str}.csv')
    meta = pd.read_csv(DATA_PATH / 'meta' / f'{track}_{date_str}.csv')

    for df in (pred, entries):
        df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
        if 'horse_number' in df.columns:
            df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').astype('Int64')
        if 'horse' in df.columns:
            df['horse'] = pd.to_numeric(df['horse'], errors='coerce').astype('Int64')

    if 'race' in meta.columns:
        meta['race'] = pd.to_numeric(meta['race'], errors='coerce').astype('Int64')

    return {'predictions': pred, 'entries': entries, 'meta': meta}


def get_winner(entries_df, race_num):
    race = entries_df[entries_df['race'] == race_num]
    if race.empty:
        return None
    winners = race[race['finish'] == 1]
    if winners.empty:
        return None
    return int(winners.iloc[0]['horse'])


def get_pick3_payout(meta_df, end_race):
    row = meta_df[meta_df['race'] == end_race]
    if row.empty:
        return None
    if 'pick_3_$2_payout' not in row.columns:
        return None
    val = row.iloc[0]['pick_3_$2_payout']
    if pd.isna(val):
        return None
    try:
        return float(str(val).replace('$', '').replace(',', ''))
    except (ValueError, TypeError):
        return None


# =============================================================================
# 2. GREEDY ALGORITHM (lifted from pick3_greedy_backtest, parameterized)
# =============================================================================

def build_greedy_value_ticket(pred_df, race_numbers, min_ridge, strict=False,
                               max_cost=10000):
    """
    Build a greedy_value or greedy_value_strict ticket.

    Args:
        pred_df: predictions DataFrame
        race_numbers: list of consecutive race numbers
        min_ridge: threshold below which horses can't be ADDED (seeds bypass)
        strict: if True, also stop adding when no candidate has ridge >= 2
                (positive single-race EV gate)

    Returns ticket dict or None.
    """
    df_pick = pred_df[pred_df['race'].isin(race_numbers)].copy()
    if df_pick.empty:
        return None

    # Sort each race by ridge_value desc
    sorted_horses = {}
    for race_num in sorted(race_numbers):
        race_data = df_pick[df_pick['race'] == race_num].dropna(
            subset=['ridge_value', 'ensemble_prob']
        )
        if race_data.empty:
            return None
        race_data = race_data.sort_values('ridge_value', ascending=False)
        rows = []
        for _, r in race_data.iterrows():
            rows.append({
                'horse_number': int(r['horse_number']),
                'ridge_value': float(r['ridge_value']),
                'ensemble_prob': float(r['ensemble_prob']),
            })
        if not rows:
            return None
        sorted_horses[race_num] = rows

    active_races = sorted(race_numbers)

    # Seed: top horse per race, regardless of threshold
    allocation = [1] * len(active_races)
    selections = {r: [sorted_horses[r][0]['horse_number']] for r in active_races}

    while True:
        best_idx = None
        best_mv = -np.inf

        for i, race_num in enumerate(active_races):
            next_pos = allocation[i]
            if next_pos >= len(sorted_horses[race_num]):
                continue
            candidate = sorted_horses[race_num][next_pos]
            if candidate['ridge_value'] < min_ridge:
                continue
            other_product = 1
            for j in range(len(active_races)):
                if j != i:
                    other_product *= allocation[j]
            mv = (candidate['ridge_value'] - 2) / other_product
            if mv > best_mv:
                best_mv = mv
                best_idx = i

        if best_idx is None:
            break
        if strict and best_mv <= 0:
            break

        new_alloc = allocation.copy()
        new_alloc[best_idx] += 1
        new_cost = int(np.prod(new_alloc))
        if new_cost > max_cost:
            break

        race_num = active_races[best_idx]
        next_pos = allocation[best_idx]
        added = sorted_horses[race_num][next_pos]
        selections[race_num].append(added['horse_number'])
        allocation[best_idx] += 1

    cost = int(np.prod(allocation))
    hit_prob = 1.0
    for i, race_num in enumerate(active_races):
        race_hit = sum(
            sorted_horses[race_num][j]['ensemble_prob']
            for j in range(allocation[i])
        )
        hit_prob *= max(race_hit, 1e-9)

    # Track the seed horses too — needed for indicator computation
    seeds = {r: sorted_horses[r][0] for r in active_races}
    seconds = {
        r: (sorted_horses[r][1] if len(sorted_horses[r]) > 1 else None)
        for r in active_races
    }

    return {
        'cost': cost,
        'hit_prob': hit_prob,
        'allocation': allocation,
        'selections': selections,
        'seeds': seeds,
        'seconds': seconds,
        'sorted_horses': sorted_horses,
    }


# =============================================================================
# 3. PRE-RACE INDICATORS
# =============================================================================

def compute_indicators(ticket, pred_df, race_numbers):
    """
    Compute all candidate pre-race indicators for a sequence's recommended
    ticket. These are quantities the bettor could observe BEFORE the race
    runs and use to decide whether to play.
    """
    seeds = ticket['seeds']
    seconds = ticket['seconds']
    allocation = ticket['allocation']

    seed_ridges = [seeds[r]['ridge_value'] for r in race_numbers]
    seed_probs = [seeds[r]['ensemble_prob'] for r in race_numbers]

    # Geomean is more robust than arithmetic mean to one-leg outliers.
    # (e.g., one weak leg shouldn't be hidden by two strong legs)
    def geomean(xs):
        xs = [max(x, 1e-9) for x in xs]
        return np.exp(np.mean(np.log(xs)))

    # Top vs second-ridge gap per leg. If second is None (small field),
    # use 0 as a placeholder — those races don't contribute to gap signal.
    gaps = []
    for r in race_numbers:
        if seconds[r] is None:
            gaps.append(0)
        else:
            gaps.append(max(seeds[r]['ridge_value'] - seconds[r]['ridge_value'], 0))

    # Trigger leg count: per leg, was any predict_v2 trigger active?
    # We can't easily reconstruct ridge_break_even per race here without
    # field size, so use a simpler proxy: ANY horse in the race has
    # ev_per_2 >= LOGISTIC_EDGE_FLOOR (LOG bet) OR ridge_value >= 2.0
    # (rough RDG bet proxy — actual is field-dependent but 2.0 is close
    # to the average break-even). Approximate but consistent across cards.
    n_trigger_legs = 0
    for r in race_numbers:
        race_horses = pred_df[pred_df['race'] == r]
        log_trigger = (
            'ev_per_2' in race_horses.columns
            and (race_horses['ev_per_2'] >= LOGISTIC_EDGE_FLOOR).any()
        )
        rdg_trigger = (race_horses['ridge_value'] >= 2.0).any()
        if log_trigger or rdg_trigger:
            n_trigger_legs += 1

    return {
        'ticket_cost': ticket['cost'],
        'hit_prob': ticket['hit_prob'],
        'seed_ridge_geomean': geomean(seed_ridges),
        'seed_ridge_min': min(seed_ridges),
        'top2_ridge_gap_geomean': geomean([g + 0.01 for g in gaps]),  # +eps to avoid log(0)
        'n_trigger_legs': n_trigger_legs,
    }


# =============================================================================
# 4. EVALUATE TICKET
# =============================================================================

def evaluate_ticket(ticket, winners, races, pick3_payout):
    if ticket is None:
        return (None, None, None)
    hit = all(winners[r] in ticket['selections'][r] for r in races)
    ticket_return = (pick3_payout / 2.0) if hit else 0.0
    profit = ticket_return - ticket['cost']
    return (hit, ticket_return, profit)


# =============================================================================
# 5. RUN BOTH STRATEGIES OVER ALL SEQUENCES
# =============================================================================

STRATEGIES = [
    {'name': 'greedy_value@1.33',        'min_ridge': 1.33, 'strict': False},
    {'name': 'greedy_value_strict@2.5',  'min_ridge': 2.5,  'strict': True},
]


def run_backtest(cards):
    rows = []

    for date_str, track in cards:
        try:
            card = load_card(date_str, track)
        except Exception as e:
            print(f"  WARN: load failed {date_str}/{track}: {e}")
            continue

        races_in_card = sorted(card['predictions']['race'].dropna().unique())
        for i in range(len(races_in_card) - 2):
            r1, r2, r3 = races_in_card[i], races_in_card[i+1], races_in_card[i+2]
            if r2 != r1 + 1 or r3 != r2 + 1:
                continue
            payout = get_pick3_payout(card['meta'], r3)
            if payout is None:
                continue
            winners = {r: get_winner(card['entries'], r) for r in (r1, r2, r3)}
            if any(w is None for w in winners.values()):
                continue

            for strat in STRATEGIES:
                ticket = build_greedy_value_ticket(
                    card['predictions'], [r1, r2, r3],
                    min_ridge=strat['min_ridge'],
                    strict=strat['strict'],
                )
                if ticket is None:
                    continue
                indicators = compute_indicators(
                    ticket, card['predictions'], [r1, r2, r3]
                )
                won, payback, profit = evaluate_ticket(
                    ticket, winners, [r1, r2, r3], payout
                )
                rows.append({
                    'date': date_str, 'track': track,
                    'races': f'{r1}-{r3}',
                    'strategy': strat['name'],
                    'cost': ticket['cost'],
                    'allocation': str(ticket['allocation']),
                    'won': won,
                    'payout': payback,
                    'profit': profit,
                    **indicators,
                })

    return pd.DataFrame(rows)


# =============================================================================
# 6. INDICATOR QUINTILE ANALYSIS
# =============================================================================

INDICATORS = [
    'ticket_cost', 'hit_prob', 'seed_ridge_geomean', 'seed_ridge_min',
    'top2_ridge_gap_geomean', 'n_trigger_legs',
]


def print_indicator_quintiles(df, strategy):
    """For each indicator, sort sequences into 5 quintiles and report
    realized ROI per quintile. Looking for monotonic trends."""
    sub = df[df['strategy'] == strategy].copy()
    if sub.empty:
        return

    print()
    print("=" * 78)
    print(f"INDICATOR QUINTILE ANALYSIS — {strategy}")
    print("=" * 78)
    print(f"  Sequences: {len(sub)}")
    print(f"  Aggregate ROI:  "
          f"{(sub['payout'].sum() - sub['cost'].sum()) / sub['cost'].sum():+.1%}")
    print(f"  Hit rate:       {sub['won'].mean():.1%}")
    print(f"  Mean cost:      ${sub['cost'].mean():.2f}")
    print()
    print("  For each indicator below: split sequences into quintiles by")
    print("  indicator value (Q1=lowest, Q5=highest), report realized ROI")
    print("  and key counts. Look for monotonic patterns across quintiles.")
    print("  ⚠ flags bins where 1-2 hits dominate the ROI estimate.")

    for indicator in INDICATORS:
        if indicator not in sub.columns or sub[indicator].nunique() < 2:
            continue

        # n_trigger_legs is integer-valued (0/1/2/3), use exact bins
        if indicator == 'n_trigger_legs':
            sub['_q'] = sub[indicator]
            bin_label_fn = lambda b: f"= {int(b)}"
        else:
            try:
                sub['_q'] = pd.qcut(sub[indicator], q=5, labels=False, duplicates='drop')
                bin_label_fn = lambda b: f"Q{int(b)+1}"
            except ValueError:
                continue

        n_bins = sub['_q'].nunique()
        if n_bins < 2:
            continue

        print(f"\n  {indicator}")
        print(f"    {'bin':<5} {'n':>4} {'mean_ind':>10} {'mean_cost':>10} "
              f"{'hit%':>6} {'hits':>5} {'ROI':>8}  flag")
        print(f"    {'-'*5} {'-'*4} {'-'*10} {'-'*10} {'-'*6} {'-'*5} {'-'*8}  ----")

        for b in sorted(sub['_q'].dropna().unique()):
            cell = sub[sub['_q'] == b]
            n = len(cell)
            mean_ind = cell[indicator].mean()
            mean_cost = cell['cost'].mean()
            hit_rate = cell['won'].mean()
            n_hits = cell['won'].sum()
            total_cost = cell['cost'].sum()
            total_return = cell['payout'].sum()
            roi = (total_return - total_cost) / total_cost if total_cost > 0 else 0

            # ⚠ if 1-2 hits dominate the bin's profit
            warn = ''
            if n_hits >= 1 and n_hits <= 2 and roi > -0.10:
                warn = '⚠ small-n hit'

            label = bin_label_fn(b)
            print(f"    {label:<5} {n:>4} {mean_ind:>10.3f} "
                  f"${mean_cost:>8.2f} {hit_rate:>5.1%} "
                  f"{int(n_hits):>5} {roi:>+7.1%}  {warn}")


def print_percentile_cuts(df, strategy):
    """
    For each indicator, apply percentile-based cuts and report what the
    rule would have produced. Lets us check robustness of the quintile
    pattern: does top 25% vs top 40% give similar ROI? Does the bottom
    cut maintain its negative ROI?

    A real rule should be (a) monotonic-ish across cuts (top 25% better
    than top 40% better than top 50%), and (b) symmetric on the skip
    side (bottom 40% as bad as bottom 25% gives confidence both halves
    are real, not just cherry-picking the extreme).

    The cuts we test for each indicator:
      bottom 20% (skip), bottom 40% (skip), top 40% (play), top 25% (play)

    For indicators where LOWER = better (seed_ridge_min, seed_ridge_geomean,
    top2_ridge_gap_geomean), we flip the labels so "play" always means
    "play the better-performing tail."
    """
    sub = df[df['strategy'] == strategy].copy()
    if sub.empty:
        return

    print()
    print("=" * 78)
    print(f"PERCENTILE CUT ROBUSTNESS — {strategy}")
    print("=" * 78)
    print(f"  Sequences: {len(sub)}, baseline ROI (play all): "
          f"{(sub['payout'].sum() - sub['cost'].sum()) / sub['cost'].sum():+.1%}")
    print()
    print("  For each indicator, apply different play/skip thresholds and")
    print("  report aggregate ROI. A real signal should show:")
    print("    - Top 25% better than Top 40% better than baseline")
    print("    - Bottom 25% worse than Bottom 40% worse than baseline")
    print("  If 25%/40% cuts diverge wildly, the signal is a few outliers.")
    print("  Note: 'better tail' is auto-detected per indicator from quintile ROI.")

    for indicator in INDICATORS:
        if indicator not in sub.columns or sub[indicator].nunique() < 2:
            continue

        # Skip integer-valued indicators where percentile cuts don't make sense
        if indicator == 'n_trigger_legs':
            print(f"\n  {indicator}: skipping (integer-valued, see quintile table above)")
            continue

        # Determine which tail is "better" by comparing top vs bottom 20%
        ind_sorted = sub.sort_values(indicator)
        n = len(sub)
        n_20 = max(1, int(0.20 * n))
        bot_20 = ind_sorted.head(n_20)
        top_20 = ind_sorted.tail(n_20)

        bot_roi = _roi(bot_20)
        top_roi = _roi(top_20)

        if bot_roi > top_roi:
            better_tail = 'lower'
        else:
            better_tail = 'higher'

        print(f"\n  {indicator}  (better tail: {better_tail})")
        print(f"    {'cut':<24} {'n':>4} {'mean_cost':>10} {'hit%':>6} "
              f"{'hits':>5} {'ROI':>8}")
        print(f"    {'-'*24} {'-'*4} {'-'*10} {'-'*6} {'-'*5} {'-'*8}")

        cuts = [
            ('PLAY top 25%', 0.25, 'play'),
            ('PLAY top 40%', 0.40, 'play'),
            ('SKIP bottom 25%', 0.25, 'skip-result'),
            ('SKIP bottom 40%', 0.40, 'skip-result'),
        ]

        for label, frac, mode in cuts:
            n_cut = max(1, int(frac * n))
            if better_tail == 'lower':
                # "play" subset = sequences with LOW indicator values
                if mode == 'play':
                    subset = ind_sorted.head(n_cut)
                else:  # skip-result: report what's LEFT after dropping bottom (= worst) tail
                    subset = ind_sorted.tail(n - n_cut)
            else:
                # "play" subset = sequences with HIGH indicator values
                if mode == 'play':
                    subset = ind_sorted.tail(n_cut)
                else:
                    subset = ind_sorted.head(n - n_cut)

            mean_cost = subset['cost'].mean()
            hit_rate = subset['won'].mean()
            n_hits = int(subset['won'].sum())
            roi = _roi(subset)

            print(f"    {label:<24} {len(subset):>4} ${mean_cost:>8.2f} "
                  f"{hit_rate:>5.1%} {n_hits:>5} {roi:>+7.1%}")

    print()
    print("  How to read:")
    print("    'PLAY top 25%' = filter to top 25% of sequences by this indicator's")
    print("                     better tail, report ROI on that subset")
    print("    'SKIP bottom 25%' = drop the worst 25%, report ROI on the remaining 75%")
    print("    Similar ROI between top 25% and top 40% = pattern is robust")
    print("    Top 25% much better than top 40% = +EV is a thin slice that may not replicate")


def _roi(subset):
    """ROI helper: cost-weighted aggregate."""
    if subset.empty:
        return 0.0
    total_cost = subset['cost'].sum()
    total_return = subset['payout'].sum()
    if total_cost <= 0:
        return 0.0
    return (total_return - total_cost) / total_cost


# =============================================================================
# 7. CARD CONCENTRATION DIAGNOSTIC
# =============================================================================

def print_card_concentration(df, strategy):
    """Show per-card profit contribution and per-card ROI. If most of
    the strategy's profit comes from 1-2 cards, the apparent edge is
    fragile. Also surfaces consecutive-sequence overlap effect within cards."""
    sub = df[df['strategy'] == strategy].copy()
    if sub.empty:
        return

    print()
    print("=" * 78)
    print(f"CARD CONCENTRATION — {strategy}")
    print("=" * 78)
    print("  Sequences sharing races (R1-R3 and R2-R4 share two legs) inflate")
    print("  apparent sample size. If a single card's bomb leg drives multiple")
    print("  sequence wins, we're double-counting that signal.")
    print()
    print("  Per-card breakdown (sorted by total profit contribution):")

    by_card = sub.groupby(['date', 'track']).agg(
        n_seqs=('races', 'count'),
        n_hits=('won', 'sum'),
        total_cost=('cost', 'sum'),
        total_payout=('payout', 'sum'),
    ).reset_index()
    by_card['profit'] = by_card['total_payout'] - by_card['total_cost']
    by_card['roi'] = by_card['profit'] / by_card['total_cost']
    by_card = by_card.sort_values('profit', ascending=False)

    print(f"    {'date':<11} {'track':<18} {'seqs':>4} {'hits':>4} "
          f"{'cost':>7} {'payout':>8} {'profit':>8} {'ROI':>7}")
    print(f"    {'-'*11} {'-'*18} {'-'*4} {'-'*4} {'-'*7} {'-'*8} "
          f"{'-'*8} {'-'*7}")
    for _, row in by_card.iterrows():
        print(f"    {row['date']:<11} {row['track']:<18} "
              f"{int(row['n_seqs']):>4} {int(row['n_hits']):>4} "
              f"${row['total_cost']:>5.0f} ${row['total_payout']:>6.0f} "
              f"${row['profit']:>+6.0f} {row['roi']:>+6.1%}")

    # Concentration metric: top-N cards as fraction of total profit
    total_profit = by_card['profit'].sum()
    if total_profit > 0:
        sorted_profits = by_card['profit'].sort_values(ascending=False).values
        top_3_share = sorted_profits[:3].sum() / total_profit
        top_5_share = sorted_profits[:5].sum() / total_profit
        print()
        print(f"  Top 3 cards contribute {top_3_share:>+.0%} of total profit")
        print(f"  Top 5 cards contribute {top_5_share:>+.0%} of total profit")
        if top_3_share > 0.7:
            print(f"  ⚠ HIGH CONCENTRATION: removing top 3 cards likely flips strategy negative.")
    elif total_profit < 0:
        # All-negative: report the worst losers instead
        sorted_profits = by_card['profit'].sort_values(ascending=True).values
        bottom_3_share = sorted_profits[:3].sum() / total_profit
        print()
        print(f"  Bottom 3 cards contribute {bottom_3_share:>+.0%} of total losses")


# =============================================================================
# 8. TODAY MODE — apply historical quintiles to a current sequence
# =============================================================================

def today_mode(date_str, track, race_numbers, hist_df, strategy):
    """Compute today's indicators and report which historical quintile
    each falls in. Doesn't auto-decide; just shows the comparison."""
    try:
        card = load_card(date_str, track)
    except Exception as e:
        print(f"Couldn't load {date_str}/{track}: {e}")
        return

    strat_def = next((s for s in STRATEGIES if s['name'] == strategy), None)
    if strat_def is None:
        print(f"Unknown strategy: {strategy}")
        return

    ticket = build_greedy_value_ticket(
        card['predictions'], race_numbers,
        min_ridge=strat_def['min_ridge'], strict=strat_def['strict'],
    )
    if ticket is None:
        print("Couldn't build ticket — missing predictions for these races?")
        return

    indicators = compute_indicators(
        ticket, card['predictions'], race_numbers
    )

    print()
    print("=" * 78)
    print(f"TODAY: {date_str} / {track} / Pick 3 over races {race_numbers}")
    print(f"Strategy: {strategy}")
    print("=" * 78)
    print(f"  Recommended ticket: ${ticket['cost']}  alloc={ticket['allocation']}")
    print(f"  Predicted hit prob: {ticket['hit_prob']:.1%}")
    for r in sorted(ticket['selections']):
        print(f"    R{r}: {ticket['selections'][r]}")

    print()
    print("  Today's indicators vs historical distribution (this strategy only):")
    hist = hist_df[hist_df['strategy'] == strategy]
    print(f"    {'indicator':<24} {'today':>10} {'pctile':>8} "
          f"{'hist Q5 ROI':>12}  guidance")
    print(f"    {'-'*24} {'-'*10} {'-'*8} {'-'*12}  --------")

    for ind in INDICATORS:
        today_val = indicators[ind]
        hist_vals = hist[ind].dropna()
        if len(hist_vals) == 0:
            continue
        pctile = (hist_vals < today_val).mean() * 100

        # Compute Q5 ROI for this indicator
        if ind == 'n_trigger_legs':
            top_bin = hist[hist[ind] == hist[ind].max()]
        else:
            try:
                hist['_q_temp'] = pd.qcut(
                    hist[ind], q=5, labels=False, duplicates='drop'
                )
                top_bin = hist[hist['_q_temp'] == hist['_q_temp'].max()]
            except ValueError:
                top_bin = pd.DataFrame()

        if len(top_bin) > 0 and top_bin['cost'].sum() > 0:
            top_roi = (top_bin['payout'].sum() - top_bin['cost'].sum()) / top_bin['cost'].sum()
            top_roi_str = f"{top_roi:+.1%}"
        else:
            top_roi_str = "—"

        # Guidance: where does today's value land?
        if pctile >= 80:
            guide = "TOP — best history bucket"
        elif pctile >= 60:
            guide = "above median"
        elif pctile >= 40:
            guide = "near median"
        elif pctile >= 20:
            guide = "below median"
        else:
            guide = "BOTTOM — worst history bucket"

        print(f"    {ind:<24} {today_val:>10.3f} {pctile:>6.0f}% "
              f"{top_roi_str:>12}  {guide}")

    print()
    print("  How to decide: if multiple indicators land in their TOP bucket")
    print("  AND those buckets show meaningfully better historical ROI,")
    print("  this is a play. If they're mixed or land in BOTTOM buckets,")
    print("  consider skipping this sequence. None of this is determinative")
    print("  — historical patterns on 135 sequences are noisy.")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pick 3 play/skip analysis')
    parser.add_argument('--card', type=str, default=None,
                        help='Today mode: date YYYY-MM-DD')
    parser.add_argument('--track', type=str, default=None,
                        help='Today mode: track key (required with --card)')
    parser.add_argument('--races', type=str, default=None,
                        help='Today mode: e.g. "4-6"')
    parser.add_argument('--strategy', type=str, default='greedy_value@1.33',
                        choices=[s['name'] for s in STRATEGIES],
                        help='Today mode: which strategy to run')
    parser.add_argument('--save', type=str, default=None,
                        help='Save full per-ticket results to CSV')
    args = parser.parse_args()

    cards = discover_cards()
    if not cards:
        print("No cards. Run backfill_results.py first.")
        return

    print(f"Pick 3 play/skip analysis")
    print(f"  Cards: {len(cards)}")

    df = run_backtest(cards)
    if df.empty:
        print("  No evaluable sequences found.")
        return

    # Per-strategy tables: indicator quintiles, percentile cuts, then card concentration
    for strat in STRATEGIES:
        print_indicator_quintiles(df, strat['name'])
        print_percentile_cuts(df, strat['name'])
        print_card_concentration(df, strat['name'])

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved per-ticket data to {out}")

    # Today mode
    if args.card and args.track and args.races:
        if '-' in args.races:
            lo, hi = args.races.split('-')
            race_numbers = list(range(int(lo), int(hi) + 1))
        else:
            race_numbers = [int(r) for r in args.races.split(',')]
        if len(race_numbers) != 3:
            print(f"\nToday mode needs exactly 3 races, got {race_numbers}")
            return
        today_mode(args.card, args.track, race_numbers, df, args.strategy)


if __name__ == '__main__':
    main()