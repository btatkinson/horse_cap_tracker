"""
pick_optimizer.py

Optimize Pick 3/4/5/6 tickets using Pareto frontier over model probabilities.

Reads predictions.csv output from predict_races.py — avoids re-running
the probability engine.

Usage:
    python pick_optimizer.py --date 2026-04-08 --track keeneland --races 1-5
    python pick_optimizer.py --date 2026-04-08 --track keeneland --races 7-9 --pool 50000
    python pick_optimizer.py --date 2026-04-08 --track keeneland --races 3-8 --max-cost 500

Reads from:
    {DATA_PATH}/predictions/{date}/{track}/predictions.csv

Arguments:
    --races     Race range, e.g. "1-5" or "3,4,5,6,7"
    --pool      Estimated pool size ($)
    --takeout   Track takeout rate (default 0.22)
    --max-cost  Max ticket cost in combos ($1 base)
    --blend     Model vs ML blend (1.0 = all model, 0.5 = 50/50)
    --top-n     Number of frontier tickets to show
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from operator import itemgetter

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])


# =============================================================================
# 1. LOADING
# =============================================================================

def load_predictions(date_str, track):
    """Load predictions.csv from predict_races output."""
    pred_path = DATA_PATH / 'predictions' / date_str / track / 'predictions.csv'
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No predictions found at {pred_path}\n"
            f"Run: python predict_races.py --date {date_str} --track {track}"
        )
    df = pd.read_csv(pred_path)
    df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').astype('Int64')
    return df


def parse_race_range(race_str):
    """Parse '1-5' or '3,4,5,6,7' into sorted list of ints."""
    if '-' in race_str and ',' not in race_str:
        start, end = race_str.split('-')
        return list(range(int(start), int(end) + 1))
    return sorted(int(r.strip()) for r in race_str.split(','))


# =============================================================================
# 2. TICKET OPTIMIZER (Pareto frontier)
# =============================================================================

def optimize_pick_tickets(df, race_numbers, blend_factor=0.5,
                          pool_size=100000, takeout_rate=0.22,
                          max_cost=None, min_prob=None, min_total_prob=None,
                          budget_steps=20,
                          prob_col='ensemble_prob', ml_prob_col='ml_win_prob_norm'):
    """
    Optimize Pick N tickets using model probabilities blended with ML.

    Uses Pareto frontier: for each cost level, find the selection with
    highest EV. EV = (our_prob / market_prob) * net_pool - cost.

    Parameters
    ----------
    df : DataFrame with race data (from predictions.csv)
    race_numbers : list of race numbers (e.g. [1,2,3,4,5])
    blend_factor : weight on model (1.0 = all model, 0.0 = all ML)
    pool_size : estimated pool size in dollars
    takeout_rate : track takeout (default 0.22)
    max_cost : max ticket cost in combinations
    min_prob : min blended coverage per race (e.g. 0.7 = 70%)
    min_total_prob : min total hit probability
    budget_steps : number of frontier points to return

    Returns
    -------
    list of (cost, ev, probability, selections_dict)
    """
    bet_type = len(race_numbers)
    df_pick = df[df['race'].isin(race_numbers)].copy()

    # Check for ML probs
    has_ml = (ml_prob_col in df_pick.columns and df_pick[ml_prob_col].notna().any())

    if not has_ml:
        blend_factor = 1.0
        ml_prob_col_used = prob_col
        print("No ML odds available. Using model probs only (no overlay detection).")
    else:
        ml_prob_col_used = ml_prob_col

    # Blended probabilities
    df_pick['blended_prob'] = (blend_factor * df_pick[prob_col] +
                               (1 - blend_factor) * df_pick[ml_prob_col_used])

    # Normalize within each race
    for race_num in race_numbers:
        mask = df_pick['race'] == race_num
        total = df_pick.loc[mask, 'blended_prob'].sum()
        if total > 0:
            df_pick.loc[mask, 'blended_prob'] /= total

    # Build sorted horse lists per race
    races = sorted(race_numbers)
    blended_probs = {}
    market_probs = {}
    sorted_horses = {}

    for race_num in races:
        rkey = f"Race {race_num}"
        race_data = df_pick[df_pick['race'] == race_num]

        blended_probs[rkey] = dict(zip(race_data['horse_number'], race_data['blended_prob']))
        market_probs[rkey] = dict(zip(race_data['horse_number'], race_data[ml_prob_col_used]))
        sorted_horses[rkey] = sorted(blended_probs[rkey].items(),
                                      key=itemgetter(1), reverse=True)

    # Min selections per race
    min_selection = [1] * bet_type
    if min_prob is not None:
        for i, race in enumerate(sorted(blended_probs.keys())):
            cum = 0
            count = 0
            for _, prob in sorted_horses[race]:
                cum += prob
                count += 1
                if cum >= min_prob:
                    break
            min_selection[i] = max(1, count)

    max_horses = [len(sorted_horses[r]) for r in sorted(blended_probs.keys())]

    # Generate all allocations
    all_allocations = []
    for total in range(sum(min_selection), sum(max_horses) + 1):
        base = min_selection.copy()
        remaining = total - sum(base)

        def distribute(current, rem, idx=0):
            if idx == len(current) - 1:
                if current[idx] + rem <= max_horses[idx]:
                    alloc = current.copy()
                    alloc[idx] += rem
                    all_allocations.append(alloc)
                return
            for i in range(rem + 1):
                if current[idx] + i <= max_horses[idx]:
                    c = current.copy()
                    c[idx] += i
                    distribute(c, rem - i, idx + 1)

        distribute(base, remaining)

    # Evaluate each allocation
    results = []
    race_list = sorted(blended_probs.keys())

    for allocation in all_allocations:
        cost = int(np.prod(allocation))

        if max_cost is not None and cost > max_cost:
            continue

        total_prob = 1.0
        # Product of per-race "ratio sums": sum(our_i / mkt_i) per race
        # This is the correct factored form of sum-over-all-combos(our/mkt)
        ratio_product = 1.0
        selections = {}

        for i, race in enumerate(race_list):
            top = sorted_horses[race][:allocation[i]]
            race_prob = sum(p for _, p in top)

            # Sum of our_prob/market_prob ratios for selected horses
            race_ratio_sum = 0
            for h, our_p in top:
                mkt_p = market_probs[race].get(h, 0)
                if mkt_p > 0:
                    race_ratio_sum += our_p / mkt_p
                else:
                    race_ratio_sum += our_p / 1e-6  # avoid div zero

            total_prob *= race_prob
            ratio_product *= race_ratio_sum
            selections[race] = [h for h, _ in top]

        if min_total_prob is not None and total_prob < min_total_prob:
            continue

        # Correct pari-mutuel EV:
        # EV = (1 - takeout) * product(sum(our_i/mkt_i) per race) - cost
        # Pool size cancels: payout = net_pool / (pool * mkt_combo_prob)
        # and net_pool = pool * (1-takeout), so payout = (1-takeout) / mkt_combo_prob
        ev = (1 - takeout_rate) * ratio_product - cost

        results.append((cost, ev, total_prob, selections))

    # Sort by EV descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Efficient frontier: best EV per cost level
    frontier = []
    best_ev = {}
    for r in results:
        cost = r[0]
        if cost not in best_ev or r[1] > best_ev[cost]:
            best_ev[cost] = r[1]
            frontier.append(r)

    frontier.sort(key=lambda x: x[0])

    # Sample frontier points
    if frontier and budget_steps > 0 and budget_steps < len(frontier):
        indices = np.linspace(0, len(frontier) - 1, budget_steps, dtype=int)
        return [frontier[i] for i in indices]

    return frontier


# =============================================================================
# 3. REPORTING
# =============================================================================

def print_race_probs(df, race_numbers, prob_col='ensemble_prob', ml_col='ml_win_prob_norm'):
    """Print probability table for the pick ticket races."""
    has_ml = ml_col in df.columns and df[ml_col].notna().any()

    for race_num in race_numbers:
        race = df[df['race'] == race_num].copy()
        race = race.sort_values(prob_col, ascending=False)

        print(f"\n--- Race {race_num} ({len(race)} horses) ---")

        if has_ml:
            print(f"{'#':>3} {'Model':>7} {'ML':>7} {'Edge':>7}")
            for _, row in race.iterrows():
                edge = row.get('edge', np.nan)
                edge_str = f"{edge:>+6.1%}" if pd.notna(edge) else f"{'—':>6}"
                ml_str = f"{row[ml_col]:>6.1%}" if pd.notna(row.get(ml_col)) else f"{'—':>6}"
                print(f"{int(row['horse_number']):>3} "
                      f"{row[prob_col]:>6.1%} "
                      f"{ml_str} "
                      f"{edge_str}")
        else:
            print(f"{'#':>3} {'Model':>7}")
            for _, row in race.iterrows():
                print(f"{int(row['horse_number']):>3} {row[prob_col]:>6.1%}")


def print_tickets(frontier, bet_type, top_n=10):
    """Print top ticket options from the Pareto frontier."""
    sorted_f = sorted(frontier, key=lambda x: x[1], reverse=True)

    print(f"\n{'='*65}")
    print(f"TOP PICK {bet_type} TICKETS BY EXPECTED VALUE")
    print(f"{'='*65}")

    for i, (cost, ev, prob, selections) in enumerate(sorted_f[:top_n]):
        roi = ev / cost * 100 if cost > 0 else 0
        # Expected payout if the ticket hits
        exp_payout = (ev + cost) / prob if prob > 0 else 0

        print(f"\nOption {i+1}: ${cost} ticket | EV ${ev:+,.2f} ({roi:+.1f}% ROI) | "
              f"Hit {prob:.2%} | Avg payout ${exp_payout:,.0f}")
        print("-" * 55)

        for race_key in sorted(selections.keys()):
            race_num = int(race_key.split()[1])
            horses = selections[race_key]
            print(f"  R{race_num}: {', '.join(str(int(h)) for h in horses)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pick N ticket optimizer (Pareto frontier)")
    parser.add_argument('--date', type=str, required=True, help='Date (e.g. 2026-04-08)')
    parser.add_argument('--track', type=str, required=True, help='Track (e.g. keeneland)')
    parser.add_argument('--races', type=str, required=True,
                        help='Race range: "1-5" or "3,4,5,6,7"')
    parser.add_argument('--pool', type=int, default=100000, help='Estimated pool size ($)')
    parser.add_argument('--takeout', type=float, default=0.22, help='Track takeout rate')
    parser.add_argument('--max-cost', type=int, default=None, help='Max ticket cost ($)')
    parser.add_argument('--blend', type=float, default=0.5,
                        help='Model weight (1.0=all model, 0.0=all ML)')
    parser.add_argument('--min-prob', type=float, default=None,
                        help='Min coverage per race (e.g. 0.7)')
    parser.add_argument('--min-total', type=float, default=None,
                        help='Min total hit probability')
    parser.add_argument('--top-n', type=int, default=10, help='Number of tickets to show')
    args = parser.parse_args()

    race_numbers = parse_race_range(args.races)
    bet_type = len(race_numbers)

    print(f"Pick {bet_type} Optimizer: {args.date} / {args.track}")
    print(f"  Races: {race_numbers}")
    print(f"  Pool: ${args.pool:,}  |  Takeout: {args.takeout:.0%}  |  Blend: {args.blend}")
    if args.max_cost:
        print(f"  Max cost: ${args.max_cost}")

    # Load predictions
    df = load_predictions(args.date, args.track)
    available_races = sorted(df['race'].unique())
    missing = [r for r in race_numbers if r not in available_races]
    if missing:
        print(f"\n  WARNING: Races {missing} not in predictions (available: {available_races})")
        race_numbers = [r for r in race_numbers if r in available_races]
        if len(race_numbers) < 2:
            print("  Need at least 2 races. Exiting.")
            return

    # Print race probabilities
    print_race_probs(df, race_numbers)

    # Run optimizer
    print(f"\nOptimizing Pick {len(race_numbers)} tickets...")
    frontier = optimize_pick_tickets(
        df, race_numbers,
        blend_factor=args.blend,
        pool_size=args.pool,
        takeout_rate=args.takeout,
        max_cost=args.max_cost,
        min_prob=args.min_prob,
        min_total_prob=args.min_total,
        budget_steps=500,
    )

    if not frontier:
        print("No valid tickets found with given constraints.")
        return

    print(f"\n  Evaluated → {len(frontier)} frontier points")
    print_tickets(frontier, bet_type=len(race_numbers), top_n=args.top_n)


if __name__ == '__main__':
    main()