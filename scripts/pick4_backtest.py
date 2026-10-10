"""
pick4_backtest.py

Pick 4 historical backtest using v5's strategy and play/skip rule:
    - Strategy: greedy_value_strict at min_ridge=2.375 (leg-aware
      relaxation from the Pick 3 baseline of 2.5)
    - Play/skip indicator: seed_ridge_geomean in bottom 33%

Tests whether the rule we tuned on Pick 3 data actually transfers to
Pick 4. Two things being validated simultaneously:
    1. Does min_ridge=2.375 produce reasonable Pick 4 tickets?
    2. Does the play/skip rule still discriminate good vs bad Pick 4 sequences?

Reports cutoff thresholds two ways:
    (a) Computed from the Pick 4 distribution itself (its own data)
    (b) Transferred from Pick 3 (what v5 actually does when Pick 4 sample
        is below MIN_HIST_FOR_BETTYPE_CUTOFF=50)

If (a) and (b) give similar cutoffs and similar play/skip ROIs, the
transfer is justified. If they differ a lot, the geomean's scale-invariance
claim is weaker than we thought.

Caveats:
    - Smaller sample than Pick 3 (~80-100 sequences expected). ROI
      confidence intervals are wider.
    - $1-base assumption (matches Keeneland/Churchill defaults). Tracks
      with $0.50 minimums would have ~2x better realized ROI.
    - 19 cards is still 19 cards. Card concentration matters.

Usage:
    python pick4_backtest.py
    python pick4_backtest.py --save data/pick4_results.csv
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

# Match v5 constants
MIN_RIDGE_PICK3 = 2.5
MIN_RIDGE_RELAXATION_PER_LEG = 0.125
PLAY_FRACTION = 0.33

PICK4_LEGS = 4
MIN_RIDGE_PICK4 = max(2.0, MIN_RIDGE_PICK3 - (PICK4_LEGS - 3) * MIN_RIDGE_RELAXATION_PER_LEG)


# =============================================================================
# 1. LOADING
# =============================================================================

def discover_cards():
    pred_root = DATA_PATH / 'predictions'
    if not pred_root.exists():
        return []
    cards = []
    for date_dir in pred_root.iterdir():
        if not date_dir.is_dir() or not _looks_like_date(date_dir.name):
            continue
        for track_dir in date_dir.iterdir():
            if not track_dir.is_dir():
                continue
            pred = track_dir / 'predictions.csv'
            entries = DATA_PATH / 'entries' / f'{track_dir.name}_{date_dir.name}.csv'
            meta = DATA_PATH / 'meta' / f'{track_dir.name}_{date_dir.name}.csv'
            if pred.exists() and entries.exists() and meta.exists():
                cards.append((date_dir.name, track_dir.name))
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


def get_pick4_payout(meta_df, end_race):
    """Pick 4 pool payout for the sequence ending at end_race."""
    row = meta_df[meta_df['race'] == end_race]
    if row.empty:
        return None
    if 'pick_4_$2_payout' not in row.columns:
        return None
    val = row.iloc[0]['pick_4_$2_payout']
    if pd.isna(val):
        return None
    try:
        return float(str(val).replace('$', '').replace(',', ''))
    except (ValueError, TypeError):
        return None


def get_pick3_payout(meta_df, end_race):
    """Used for the Pick 3 cutoff comparison only."""
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
# 2. STRATEGY (lifted from horizontals_v5)
# =============================================================================

def build_ticket(pred_df, race_numbers, min_ridge, max_cost=10000):
    """greedy_value_strict ticket build. Same as v5."""
    df_pick = pred_df[pred_df['race'].isin(race_numbers)].copy()
    if df_pick.empty:
        return None
    if 'ridge_value' not in df_pick.columns or 'ensemble_prob' not in df_pick.columns:
        return None

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
    allocation = [1] * len(active_races)
    selections = {r: [sorted_horses[r][0]['horse_number']] for r in active_races}

    while True:
        best_idx, best_mv = None, -np.inf
        for i, race_num in enumerate(active_races):
            next_pos = allocation[i]
            if next_pos >= len(sorted_horses[race_num]):
                continue
            cand = sorted_horses[race_num][next_pos]
            if cand['ridge_value'] < min_ridge:
                continue
            other_product = 1
            for j in range(len(active_races)):
                if j != i:
                    other_product *= allocation[j]
            mv = (cand['ridge_value'] - 2) / other_product
            if mv > best_mv:
                best_mv = mv
                best_idx = i

        if best_idx is None:
            break
        if best_mv <= 0:  # strict EV gate
            break

        new_alloc = allocation.copy()
        new_alloc[best_idx] += 1
        if int(np.prod(new_alloc)) > max_cost:
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

    seeds = {r: sorted_horses[r][0] for r in active_races}
    return {
        'cost': cost,
        'hit_prob': hit_prob,
        'allocation': allocation,
        'selections': selections,
        'seeds': seeds,
    }


def compute_seed_ridge_geomean(ticket):
    seeds = ticket['seeds']
    seed_ridges = [seeds[r]['ridge_value'] for r in seeds]
    seed_ridges = [max(rv, 1e-9) for rv in seed_ridges]
    return float(np.exp(np.mean(np.log(seed_ridges))))


# =============================================================================
# 3. EVALUATE
# =============================================================================

def evaluate_ticket(ticket, winners, races, payout):
    if ticket is None:
        return (None, None, None)
    hit = all(winners[r] in ticket['selections'][r] for r in races)
    ticket_return = (payout / 2.0) if hit else 0.0
    profit = ticket_return - ticket['cost']
    return (hit, ticket_return, profit)


# =============================================================================
# 4. RUN PICK 4 BACKTEST + PICK 3 CUTOFF FOR COMPARISON
# =============================================================================

def run_pick4_backtest(cards):
    """Build the Pick 4 ticket per sequence, evaluate, return DataFrame."""
    rows = []
    for date_str, track in cards:
        try:
            card = load_card(date_str, track)
        except Exception as e:
            print(f"  WARN: load failed {date_str}/{track}: {e}")
            continue

        races_in_card = sorted(card['predictions']['race'].dropna().unique())
        for i in range(len(races_in_card) - 3):
            seq = list(races_in_card[i:i+4])
            consecutive = all(seq[j+1] == seq[j]+1 for j in range(3))
            if not consecutive:
                continue
            payout = get_pick4_payout(card['meta'], seq[-1])
            if payout is None:
                continue
            winners = {r: get_winner(card['entries'], r) for r in seq}
            if any(w is None for w in winners.values()):
                continue

            ticket = build_ticket(card['predictions'], seq, min_ridge=MIN_RIDGE_PICK4)
            if ticket is None:
                continue
            geomean = compute_seed_ridge_geomean(ticket)
            won, ret, profit = evaluate_ticket(ticket, winners, seq, payout)

            rows.append({
                'date': date_str, 'track': track,
                'races': '-'.join(str(r) for r in seq),
                'cost': ticket['cost'],
                'allocation': str(ticket['allocation']),
                'hit_prob': ticket['hit_prob'],
                'seed_ridge_geomean': geomean,
                'won': won,
                'payout': ret,
                'profit': profit,
            })

    return pd.DataFrame(rows)


def collect_pick3_geomeans(cards):
    """Compute the Pick 3 seed_ridge_geomean distribution at min_ridge=2.5
    (the v5 Pick 3 threshold). Used only for the cutoff transfer comparison."""
    geomeans = []
    for date_str, track in cards:
        try:
            card = load_card(date_str, track)
        except Exception:
            continue
        races_in_card = sorted(card['predictions']['race'].dropna().unique())
        for i in range(len(races_in_card) - 2):
            seq = list(races_in_card[i:i+3])
            consecutive = all(seq[j+1] == seq[j]+1 for j in range(2))
            if not consecutive:
                continue
            payout = get_pick3_payout(card['meta'], seq[-1])
            if payout is None:
                continue
            ticket = build_ticket(card['predictions'], seq, min_ridge=MIN_RIDGE_PICK3)
            if ticket is None:
                continue
            geomeans.append(compute_seed_ridge_geomean(ticket))
    return geomeans


# =============================================================================
# 5. REPORT
# =============================================================================

def print_summary(df, pick3_geomeans):
    n_seqs = len(df)
    if n_seqs == 0:
        print("\n  No evaluable Pick 4 sequences.")
        return

    total_cost = df['cost'].sum()
    total_payout = df['payout'].sum()
    total_profit = total_payout - total_cost
    baseline_roi = total_profit / total_cost if total_cost > 0 else 0
    hit_rate = df['won'].mean()

    print()
    print("=" * 78)
    print(f"PICK 4 BACKTEST — greedy_value_strict @ min_ridge={MIN_RIDGE_PICK4}")
    print("=" * 78)
    print(f"  Sequences evaluated: {n_seqs}")
    print(f"  Hit rate:            {hit_rate:.1%} ({int(df['won'].sum())} hits)")
    print(f"  Mean ticket cost:    ${df['cost'].mean():.2f}")
    print(f"  Mean payout (winners): "
          f"${df[df['won'] == True]['payout'].mean():.2f}" if df['won'].any() else "  (no hits)")
    print(f"  Aggregate ROI (play all): {baseline_roi:+.1%}")
    print(f"  ↑ baseline; -22% = random-bettor takeout reference")

    # Two cutoff sources
    print()
    print("  Cutoff sources for play/skip rule:")
    own_q = float(np.quantile(df['seed_ridge_geomean'], PLAY_FRACTION))
    print(f"    (a) Pick 4 own distribution ({n_seqs} seqs): "
          f"{PLAY_FRACTION*100:.0f}-pct = {own_q:.3f}")
    if pick3_geomeans:
        p3_q = float(np.quantile(pick3_geomeans, PLAY_FRACTION))
        print(f"    (b) Pick 3 transfer ({len(pick3_geomeans)} seqs): "
              f"{PLAY_FRACTION*100:.0f}-pct = {p3_q:.3f}")
        print(f"    Difference: {own_q - p3_q:+.3f}  "
              f"({'consistent' if abs(own_q - p3_q) < 0.5 else 'meaningful divergence'})")
    else:
        p3_q = None

    # Apply each cutoff and report ROI
    print()
    print("  Apply cutoff: PLAY when seed_ridge_geomean <= cutoff")
    print(f"    {'cutoff source':<28} {'cutoff':>7} {'n':>4} {'hit%':>6} "
          f"{'mean_cost':>10} {'ROI':>8}")
    print(f"    {'-'*28} {'-'*7} {'-'*4} {'-'*6} {'-'*10} {'-'*8}")

    for label, cutoff in [('Pick 4 own', own_q),
                          ('Pick 3 transfer', p3_q)]:
        if cutoff is None:
            continue
        played = df[df['seed_ridge_geomean'] <= cutoff]
        n = len(played)
        if n == 0:
            print(f"    {label:<28} {cutoff:>7.3f} {n:>4} {'—':>6} {'—':>10} {'—':>8}")
            continue
        hit = played['won'].mean()
        mean_cost = played['cost'].mean()
        c = played['cost'].sum()
        p = played['payout'].sum()
        roi = (p - c) / c if c > 0 else 0
        print(f"    {label:<28} {cutoff:>7.3f} {n:>4} {hit:>5.1%} "
              f"${mean_cost:>8.2f} {roi:>+7.1%}")


def print_quintile_analysis(df):
    """Same kind of analysis as pick3_play_skip — does seed_ridge_geomean
    discriminate within Pick 4 too?"""
    if len(df) < 10:
        print("\n  Skipping quintile analysis (too few sequences)")
        return

    print()
    print("=" * 78)
    print("SEED_RIDGE_GEOMEAN QUINTILE ANALYSIS — PICK 4")
    print("=" * 78)
    print("  Same diagnostic as Pick 3: split sequences by indicator value,")
    print("  report per-quintile ROI. If Q1 (low) has positive ROI and Q5")
    print("  (high) negative, the rule transfers to Pick 4.")

    df = df.copy()
    try:
        df['_q'] = pd.qcut(df['seed_ridge_geomean'], q=5, labels=False, duplicates='drop')
    except ValueError:
        print("  Couldn't form 5 distinct bins — sample too small or too clustered.")
        return

    print(f"\n    {'q':>3} {'n':>4} {'mean_geomean':>12} {'mean_cost':>10} "
          f"{'hit%':>6} {'hits':>5} {'ROI':>8}  flag")
    print(f"    {'-'*3} {'-'*4} {'-'*12} {'-'*10} {'-'*6} {'-'*5} {'-'*8}  ----")

    for q in sorted(df['_q'].dropna().unique()):
        cell = df[df['_q'] == q]
        n = len(cell)
        n_hits = int(cell['won'].sum())
        mean_g = cell['seed_ridge_geomean'].mean()
        mean_c = cell['cost'].mean()
        hit = cell['won'].mean()
        c = cell['cost'].sum()
        p = cell['payout'].sum()
        roi = (p - c) / c if c > 0 else 0

        warn = '⚠ small-n hit' if 1 <= n_hits <= 2 and roi > -0.10 else ''
        print(f"    Q{int(q)+1:>1}  {n:>4} {mean_g:>12.3f} ${mean_c:>8.2f} "
              f"{hit:>5.1%} {n_hits:>5} {roi:>+7.1%}  {warn}")


def print_card_concentration(df):
    """Same card-level diagnostic as Pick 3 version."""
    if len(df) == 0:
        return

    print()
    print("=" * 78)
    print("CARD CONCENTRATION — PICK 4")
    print("=" * 78)
    print("  Sequences sharing races inflate effective sample. A card with")
    print("  one bomb-leg winner contributes to multiple sequences if that")
    print("  leg is shared. If top cards dominate profit, the apparent edge")
    print("  is more fragile than n_seqs suggests.")

    by_card = df.groupby(['date', 'track']).agg(
        n_seqs=('races', 'count'),
        n_hits=('won', 'sum'),
        cost=('cost', 'sum'),
        payout=('payout', 'sum'),
    ).reset_index()
    by_card['profit'] = by_card['payout'] - by_card['cost']
    by_card['roi'] = by_card['profit'] / by_card['cost']
    by_card = by_card.sort_values('profit', ascending=False)

    print()
    print(f"    {'date':<11} {'track':<18} {'seqs':>4} {'hits':>4} "
          f"{'cost':>7} {'payout':>8} {'profit':>8} {'ROI':>7}")
    print(f"    {'-'*11} {'-'*18} {'-'*4} {'-'*4} {'-'*7} {'-'*8} "
          f"{'-'*8} {'-'*7}")
    for _, r in by_card.iterrows():
        print(f"    {r['date']:<11} {r['track']:<18} "
              f"{int(r['n_seqs']):>4} {int(r['n_hits']):>4} "
              f"${r['cost']:>5.0f} ${r['payout']:>6.0f} "
              f"${r['profit']:>+6.0f} {r['roi']:>+6.1%}")

    total_profit = by_card['profit'].sum()
    if total_profit > 0:
        sorted_p = by_card['profit'].sort_values(ascending=False).values
        print()
        print(f"  Top 3 cards contribute {sorted_p[:3].sum() / total_profit:>+.0%} of total profit")
        if sorted_p[:3].sum() / total_profit > 0.7:
            print(f"  ⚠ HIGH CONCENTRATION: removing top 3 cards likely flips strategy negative")
    elif total_profit < 0:
        sorted_p = by_card['profit'].sort_values(ascending=True).values
        print()
        print(f"  Bottom 3 cards contribute {sorted_p[:3].sum() / total_profit:>+.0%} of total losses")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pick 4 backtest with v5 strategy')
    parser.add_argument('--save', type=str, default=None,
                        help='Save per-ticket data to CSV')
    args = parser.parse_args()

    cards = discover_cards()
    if not cards:
        print("No cards available. Run backfill_results.py.")
        return

    print(f"Pick 4 backtest")
    print(f"  Cards: {len(cards)}")
    print(f"  Strategy: greedy_value_strict at min_ridge={MIN_RIDGE_PICK4}")
    print(f"  Play/skip: seed_ridge_geomean bottom {PLAY_FRACTION*100:.0f}%")

    df = run_pick4_backtest(cards)
    pick3_gm = collect_pick3_geomeans(cards)

    print_summary(df, pick3_gm)
    print_quintile_analysis(df)
    print_card_concentration(df)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved per-ticket data to {out}")


if __name__ == '__main__':
    main()