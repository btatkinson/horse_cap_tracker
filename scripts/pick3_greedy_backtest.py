"""
pick3_greedy_backtest.py

Pick 3 backtest comparing FIVE ticket-construction strategies, each
with its own cost lever swept across 9 bins. Same 135 sequences as
pick3_backtest.py, same payout math.

Strategies tested:

    v4_baseline (reference)
        v4's existing top-K-per-race construction. Single tier per
        sequence at each cost cap. Included for comparison.

    greedy_prob
        Greedy build: start with top-ensemble_prob horse per race,
        repeatedly add the candidate with highest marginal hit-prob
        per dollar. Cost lever: min_prob threshold (only consider
        horses with ensemble_prob >= threshold).

    greedy_joint
        Same algorithm but marginal value = (p × ridge) / marginal_cost.
        This is v4 scoring divided by the cost-asymmetry term, so it's
        "v4 done greedy." Cost lever: min_joint threshold on p × ridge.

    greedy_value
        Marginal value = (ridge - 2) / marginal_cost. Tests the
        hypothesis that ridge implicitly contains probability and
        shouldn't be multiplied by p again. Cost lever: min_ridge.

    greedy_value_strict
        Same as greedy_value but stops adding when no candidate has
        ridge >= 2 (only EV-positive horses considered). Tests
        whether the "include negative-EV horses for coverage"
        question matters in practice.

Why no shared threshold across variants:
    Each variant's score has different units (probability vs $-EV vs
    $-EV minus stake) so a shared threshold would mean different
    things in each. Per-variant thresholds isolate "scoring rule"
    from "what's in the candidate pool."

Output:
    Per strategy, a sweep table: 9 thresholds × {n_sequences,
    mean_cost, hit%, ROI, mean ev_per_dollar}. Reader looks for
    monotonic trends rather than the best single cell.

Honest caveats:
    - 135 sequences is small. Standard error on ROI per cell is
      large (a single $500 payout can swing a cell by 20+ pts).
      Look for trends across thresholds, not single-cell winners.
    - All variants use real Pick 3 payouts from horseracingnation
      meta CSVs. Hit-rate logic and payout math is tested in
      pick3_backtest.py.
    - $1-base-bet assumption (Keeneland/Churchill default). Tracks
      with $0.50 minimums would have actual ROI roughly 2x better
      than reported here.
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

# Threshold sweeps for each greedy variant. Ranges chosen to span
# "include almost everyone" → "only the strongest picks." Adjust if
# trends are flat at one end.
SWEEP_PROB    = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25]
SWEEP_JOINT   = [0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.30, 1.70]
SWEEP_RIDGE   = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

# v4 baseline tier sweep — for parity, use 9 cost caps too
SWEEP_COST_CAP = [3, 6, 12, 20, 40, 75, 150, 300, 600]


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
# 2. STRATEGY: v4 baseline (top-K per race, max-cost cap)
# =============================================================================

def build_v4_ticket(pred_df, race_numbers, max_cost):
    """
    Recreate v4's top-K-per-race optimizer with a hard cost cap. Returns
    the highest-ev_proxy ticket within the cap. None if no eligible.

    This is a simplified rerun of horizontals_v4.optimize_pick_tickets.
    """
    df_pick = pred_df[pred_df['race'].isin(race_numbers)].copy()
    if 'ridge_value' not in df_pick.columns or 'ensemble_prob' not in df_pick.columns:
        return None

    ridge_card_mean = df_pick['ridge_value'].mean()
    if not (ridge_card_mean > 0):
        ridge_card_mean = 1.0

    sorted_horses = {}
    for race_num in sorted(race_numbers):
        race_data = df_pick[df_pick['race'] == race_num]
        race_data = race_data.dropna(subset=['ridge_value', 'ensemble_prob'])
        if race_data.empty:
            return None
        score = race_data['ridge_value'] * race_data['ensemble_prob']
        rows = list(zip(
            race_data['horse_number'],
            race_data['ridge_value'],
            race_data['ensemble_prob'],
            score,
        ))
        rows.sort(key=lambda r: r[3], reverse=True)
        sorted_horses[race_num] = rows

    active_races = sorted(race_numbers)
    max_horses = [len(sorted_horses[r]) for r in active_races]

    best = None
    # Enumerate allocations up to max_cost
    def recurse(idx, current_alloc, current_cost):
        nonlocal best
        if idx == len(active_races):
            ticket = _eval_allocation(sorted_horses, active_races, current_alloc,
                                       ridge_card_mean)
            if ticket is None:
                return
            if best is None or ticket['ev_proxy'] > best['ev_proxy']:
                best = ticket
            return
        for k in range(1, max_horses[idx] + 1):
            new_cost = current_cost * k
            if new_cost > max_cost:
                break
            recurse(idx + 1, current_alloc + [k], new_cost)

    recurse(0, [], 1)
    return best


def _eval_allocation(sorted_horses, active_races, allocation, ridge_card_mean):
    cost = int(np.prod(allocation))
    hit_prob = 1.0
    ev_factor_product = 1.0
    selections = {}
    for i, race_num in enumerate(active_races):
        top = sorted_horses[race_num][:allocation[i]]
        race_hit = sum(p for _, _, p, _ in top if not np.isnan(p))
        race_pw_ridge = 0.0
        race_pw_total = 0.0
        for _, rv, p, _ in top:
            if np.isnan(p):
                continue
            race_pw_ridge += p * rv
            race_pw_total += p
        race_ev_factor = (race_pw_ridge / race_pw_total) if race_pw_total > 0 else 1.0
        race_ev_factor /= ridge_card_mean
        hit_prob *= max(race_hit, 1e-9)
        ev_factor_product *= max(race_ev_factor, 1e-9)
        selections[race_num] = [int(h) for h, _, _, _ in top]
    ev_proxy = hit_prob * ev_factor_product
    return {
        'cost': cost, 'hit_prob': hit_prob, 'ev_proxy': ev_proxy,
        'ev_per_dollar': ev_proxy / cost if cost > 0 else 0,
        'allocation': allocation, 'selections': selections,
    }


# =============================================================================
# 3. STRATEGY: greedy variants
# =============================================================================

def build_greedy_ticket(pred_df, race_numbers, *,
                         score_fn, threshold, marginal_score_fn,
                         max_cost=10000, require_positive_marginal=False):
    """
    Greedy ticket construction.

    Args:
        score_fn: callable(row) → float. Used to (a) pick the seed horse
                  in each race and (b) sort horses within race for ordering.
        threshold: float. Only horses with score_fn(horse) >= threshold are
                   eligible to be ADDED beyond the seed.
        marginal_score_fn: callable(horse_score, allocation, race_idx) → float.
                           The marginal-value criterion used to choose which
                           horse to add next.
        require_positive_marginal: if True, stops when no candidate has
                                    positive marginal value.

    Returns ticket dict or None if can't build (missing data).

    The seed horse is ALWAYS the top horse in each race by score_fn,
    regardless of threshold. The threshold only gates additional adds.
    Without a seed in every leg, the ticket can't exist.
    """
    df_pick = pred_df[pred_df['race'].isin(race_numbers)].copy()
    if df_pick.empty:
        return None

    # Pre-sort each race by score_fn descending. Each row of sorted_horses[race]
    # is a dict with horse_number, score, ensemble_prob, ridge_value.
    sorted_horses = {}
    for race_num in sorted(race_numbers):
        race_data = df_pick[df_pick['race'] == race_num].copy()
        if race_data.empty:
            return None
        race_data['_score'] = race_data.apply(score_fn, axis=1)
        race_data = race_data.dropna(subset=['_score'])
        if race_data.empty:
            return None
        race_data = race_data.sort_values('_score', ascending=False)
        rows = []
        for _, r in race_data.iterrows():
            if pd.isna(r.get('ensemble_prob')) or pd.isna(r.get('ridge_value')):
                continue
            rows.append({
                'horse_number': int(r['horse_number']),
                'score': float(r['_score']),
                'ensemble_prob': float(r['ensemble_prob']),
                'ridge_value': float(r['ridge_value']),
            })
        if not rows:
            return None
        sorted_horses[race_num] = rows

    active_races = sorted(race_numbers)

    # Seed: top horse per race (always, regardless of threshold)
    allocation = [1] * len(active_races)
    selections = {r: [sorted_horses[r][0]['horse_number']] for r in active_races}

    # Greedy loop: add the next horse with highest marginal value, gated
    # by threshold. The "next horse" in race i is index allocation[i].
    while True:
        best_idx = None
        best_mv = -np.inf

        for i, race_num in enumerate(active_races):
            next_pos = allocation[i]
            if next_pos >= len(sorted_horses[race_num]):
                continue  # no more horses in this race
            candidate = sorted_horses[race_num][next_pos]
            if candidate['score'] < threshold:
                continue  # below threshold, skip
            # Marginal cost of adding to leg i: product of OTHER legs' counts
            other_product = 1
            for j, _ in enumerate(active_races):
                if j == i:
                    continue
                other_product *= allocation[j]
            mv = marginal_score_fn(candidate, other_product)
            if mv > best_mv:
                best_mv = mv
                best_idx = i

        if best_idx is None:
            break  # no eligible candidates
        if require_positive_marginal and best_mv <= 0:
            break  # no positive-EV candidates left

        # Check cost cap: would adding this exceed the cap?
        new_allocation = allocation.copy()
        new_allocation[best_idx] += 1
        new_cost = int(np.prod(new_allocation))
        if new_cost > max_cost:
            break

        # Add the horse
        race_num = active_races[best_idx]
        next_pos = allocation[best_idx]
        added = sorted_horses[race_num][next_pos]
        selections[race_num].append(added['horse_number'])
        allocation[best_idx] += 1

    # Compute ticket-level metrics for output consistency
    cost = int(np.prod(allocation))
    hit_prob = 1.0
    for i, race_num in enumerate(active_races):
        race_hit = sum(
            sorted_horses[race_num][j]['ensemble_prob']
            for j in range(allocation[i])
        )
        hit_prob *= max(race_hit, 1e-9)

    return {
        'cost': cost,
        'hit_prob': hit_prob,
        'allocation': allocation,
        'selections': selections,
    }


# Score function and marginal value function PER STRATEGY
# Each pair defines variant 1, 2, 3, 4

def _score_prob(row):
    return row.get('ensemble_prob', 0)

def _score_joint(row):
    p = row.get('ensemble_prob', 0)
    rv = row.get('ridge_value', 0)
    return p * rv

def _score_ridge(row):
    return row.get('ridge_value', 0)


def _marginal_prob(candidate, other_product):
    return candidate['ensemble_prob'] / other_product

def _marginal_joint(candidate, other_product):
    return (candidate['ensemble_prob'] * candidate['ridge_value']) / other_product

def _marginal_value(candidate, other_product):
    # ridge - 2 = expected profit on $2 single-race win bet
    return (candidate['ridge_value'] - 2) / other_product


# =============================================================================
# 4. EVALUATE A TICKET
# =============================================================================

def evaluate_ticket(ticket, winners, races, pick3_payout):
    """
    Given a ticket and the actual winners per race, compute realized
    profit. Returns (won, payout, profit).

    Pick 3 payout math: pick_3_$2_payout / 2 if hit (since v4/greedy
    treat 1 combo = $1, and one specific winning combo is the only one
    that pays). Other combos in the ticket lose $1 each.
    """
    if ticket is None:
        return (None, None, None)
    hit = all(
        winners[r] in ticket['selections'][r]
        for r in races
    )
    ticket_return = (pick3_payout / 2.0) if hit else 0.0
    profit = ticket_return - ticket['cost']
    return (hit, ticket_return, profit)


# =============================================================================
# 5. RUN ALL STRATEGIES OVER ALL SEQUENCES
# =============================================================================

def run_backtest(cards, race_window=3):
    """
    For every (date, track), find all consecutive 3-race windows with
    a Pick 3 pool, evaluate every strategy at every threshold.

    Returns DataFrame with columns:
        date, track, races, strategy, threshold, cost, hit_prob,
        won, payout, profit
    """
    rows = []

    for date_str, track in cards:
        try:
            card = load_card(date_str, track)
        except Exception as e:
            print(f"  WARN: load failed {date_str}/{track}: {e}")
            continue

        races_in_card = sorted(card['predictions']['race'].dropna().unique())
        for i in range(len(races_in_card) - race_window + 1):
            race_seq = list(races_in_card[i:i + race_window])
            # Only consecutive
            consecutive = all(race_seq[j+1] == race_seq[j] + 1
                              for j in range(len(race_seq) - 1))
            if not consecutive:
                continue
            payout = get_pick3_payout(card['meta'], race_seq[-1])
            if payout is None:
                continue
            winners = {r: get_winner(card['entries'], r) for r in race_seq}
            if any(w is None for w in winners.values()):
                continue

            # ── v4 baseline at each cost cap ────────────────────────
            for cap in SWEEP_COST_CAP:
                t = build_v4_ticket(card['predictions'], race_seq, max_cost=cap)
                won, ret, profit = evaluate_ticket(t, winners, race_seq, payout)
                rows.append({
                    'date': date_str, 'track': track,
                    'races': '-'.join(str(r) for r in race_seq),
                    'strategy': 'v4_baseline', 'threshold': cap,
                    'cost': t['cost'] if t else None,
                    'hit_prob': t['hit_prob'] if t else None,
                    'won': won, 'payout': ret, 'profit': profit,
                })

            # ── Greedy variants at each threshold ───────────────────
            for thresh in SWEEP_PROB:
                t = build_greedy_ticket(
                    card['predictions'], race_seq,
                    score_fn=_score_prob, threshold=thresh,
                    marginal_score_fn=_marginal_prob,
                )
                won, ret, profit = evaluate_ticket(t, winners, race_seq, payout)
                rows.append({
                    'date': date_str, 'track': track,
                    'races': '-'.join(str(r) for r in race_seq),
                    'strategy': 'greedy_prob', 'threshold': thresh,
                    'cost': t['cost'] if t else None,
                    'hit_prob': t['hit_prob'] if t else None,
                    'won': won, 'payout': ret, 'profit': profit,
                })

            for thresh in SWEEP_JOINT:
                t = build_greedy_ticket(
                    card['predictions'], race_seq,
                    score_fn=_score_joint, threshold=thresh,
                    marginal_score_fn=_marginal_joint,
                )
                won, ret, profit = evaluate_ticket(t, winners, race_seq, payout)
                rows.append({
                    'date': date_str, 'track': track,
                    'races': '-'.join(str(r) for r in race_seq),
                    'strategy': 'greedy_joint', 'threshold': thresh,
                    'cost': t['cost'] if t else None,
                    'hit_prob': t['hit_prob'] if t else None,
                    'won': won, 'payout': ret, 'profit': profit,
                })

            for thresh in SWEEP_RIDGE:
                # greedy_value: no positivity stop
                t = build_greedy_ticket(
                    card['predictions'], race_seq,
                    score_fn=_score_ridge, threshold=thresh,
                    marginal_score_fn=_marginal_value,
                )
                won, ret, profit = evaluate_ticket(t, winners, race_seq, payout)
                rows.append({
                    'date': date_str, 'track': track,
                    'races': '-'.join(str(r) for r in race_seq),
                    'strategy': 'greedy_value', 'threshold': thresh,
                    'cost': t['cost'] if t else None,
                    'hit_prob': t['hit_prob'] if t else None,
                    'won': won, 'payout': ret, 'profit': profit,
                })

                # greedy_value_strict: positivity stop
                t = build_greedy_ticket(
                    card['predictions'], race_seq,
                    score_fn=_score_ridge, threshold=thresh,
                    marginal_score_fn=_marginal_value,
                    require_positive_marginal=True,
                )
                won, ret, profit = evaluate_ticket(t, winners, race_seq, payout)
                rows.append({
                    'date': date_str, 'track': track,
                    'races': '-'.join(str(r) for r in race_seq),
                    'strategy': 'greedy_value_strict', 'threshold': thresh,
                    'cost': t['cost'] if t else None,
                    'hit_prob': t['hit_prob'] if t else None,
                    'won': won, 'payout': ret, 'profit': profit,
                })

    return pd.DataFrame(rows)


# =============================================================================
# 6. REPORT
# =============================================================================

def print_strategy_table(df, strategy):
    """Per-strategy table: threshold sweep on rows."""
    sub = df[df['strategy'] == strategy].dropna(subset=['cost'])
    if sub.empty:
        print(f"\n  {strategy}: no successful tickets.")
        return

    print(f"\n  {strategy}")
    print(f"  {'thresh':>7} {'n':>4} {'mean_cost':>10} {'hit%':>6} "
          f"{'mean_payout':>12} {'ROI':>8}")
    print(f"  {'-'*7} {'-'*4} {'-'*10} {'-'*6} {'-'*12} {'-'*8}")

    for thresh in sorted(sub['threshold'].unique()):
        cell = sub[sub['threshold'] == thresh]
        n = len(cell)
        mean_cost = cell['cost'].mean()
        hit_rate = cell['won'].mean()
        mean_payout = cell['payout'].mean()
        total_cost = cell['cost'].sum()
        total_return = cell['payout'].sum()
        roi = (total_return - total_cost) / total_cost if total_cost > 0 else 0

        # Format threshold based on its size — small for prob, larger for cost
        if thresh >= 1:
            thresh_str = f"{thresh:>6.2f}"
        else:
            thresh_str = f"{thresh:>6.3f}"

        print(f"  {thresh_str} {n:>4} ${mean_cost:>8.2f} "
              f"{hit_rate:>5.1%} ${mean_payout:>10.2f} "
              f"{roi:>+7.1%}")


def print_summary(df):
    print()
    print("=" * 78)
    print(f"PICK 3 BACKTEST — STRATEGY × THRESHOLD COMPARISON")
    print("=" * 78)

    n_seqs = df[['date', 'track', 'races']].drop_duplicates().shape[0]
    print(f"  Evaluated on {n_seqs} Pick 3 sequences with actual payouts.")
    print()
    print("  How to read each table:")
    print("    threshold = the cost lever (lower → wider/more expensive tickets)")
    print("    n         = sequences where strategy emitted an evaluable ticket")
    print("    mean_cost = average ticket cost across those sequences")
    print("    hit%      = fraction of sequences where the ticket cashed")
    print("    ROI       = (sum of payouts - sum of costs) / sum of costs")
    print()
    print("  Look for monotonic trends across thresholds, NOT single-cell winners.")
    print("  A single $500 hit can swing one cell by 20+ pts of ROI.")

    for strategy in ['v4_baseline', 'greedy_prob', 'greedy_joint',
                     'greedy_value', 'greedy_value_strict']:
        print_strategy_table(df, strategy)


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pick 3 strategy comparison')
    parser.add_argument('--tracks', type=str, default=None)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--save', type=str, default=None,
                        help='Save full per-ticket results to CSV')
    args = parser.parse_args()

    cards = discover_cards()
    if args.tracks:
        wanted = set(t.strip() for t in args.tracks.split(','))
        cards = [(d, t) for d, t in cards if t in wanted]
    if args.start_date:
        cards = [(d, t) for d, t in cards if d >= args.start_date]
    if args.end_date:
        cards = [(d, t) for d, t in cards if d <= args.end_date]

    print(f"Pick 3 strategy backtest")
    print(f"  Cards: {len(cards)}")
    if not cards:
        print("  No cards. Run backfill_results.py first.")
        return

    df = run_backtest(cards)
    print_summary(df)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n  Saved per-ticket results to {out_path}")


if __name__ == '__main__':
    main()
