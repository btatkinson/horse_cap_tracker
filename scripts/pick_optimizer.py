"""
pick_optimizer.py

Optimize Pick 3/4/5/6 tickets using Pareto frontier over a composite
logistic + ridge heuristic.

Reads predictions.csv output from predict_races.py — avoids re-running
the probability engine.

Heuristic per horse (within each race):
    logistic_share = ensemble_prob / sum(ensemble_prob)
    ridge_share    = max($0.50, ridge_value) / sum(max($0.50, ridge_value))
    score_i        = alpha * logistic_share + (1-alpha) * ridge_share

Both shares sum to 1 within a race, so alpha is an honest 50/50 knob
and the composite stays on [0, 1]. Ridge values are floored at $0.50
to keep the signal positive without fully masking negative predictions.

Ticket score = sum of selected horse scores across all races.
Ticket cost  = product of selection counts per race.

Frontier maximizes ticket score subject to cost. Tickets that increase
cost without increasing composite score are pruned. Hit probability is
still reported (from ensemble_prob) as a sanity-check metric, not as
the optimization target.

Usage:
    python pick_optimizer.py --date 2026-04-08 --track keeneland --races 1-5
    python pick_optimizer.py --date 2026-04-08 --track keeneland --races 7-9 \
        --alpha 0.3 --max-cost 500

Reads from:
    {DATA_PATH}/predictions/{date}/{track}/predictions.csv
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
# 2. COMPOSITE SCORE
# =============================================================================

def build_composite_score(df, alpha=0.5, prob_col='ensemble_prob',
                          ridge_col='ridge_value'):
    """
    Build composite score per horse, normalized within each race.

    Both signals are converted to within-race shares (sum to 1 per race),
    then blended:

        logistic share = ensemble_prob         (already sums to 1 per race)
        ridge share    = max($0.50, ridge_value) / sum(max($0.50, ridge_value))
        score          = alpha * logistic_share + (1-alpha) * ridge_share

    Flooring ridge at $0.50 keeps the signal positive without masking
    negatives entirely — a horse the model slightly hates contributes
    little, but not nothing. The floor also prevents div-by-zero in
    degenerate races. Alpha is an honest mixing weight (alpha=0.5 =
    equal footing).

    If a race has no ridge signal at all, falls back to pure logistic
    share for that race.
    """
    df = df.copy()

    has_ridge = ridge_col in df.columns and df[ridge_col].notna().any()

    df['logistic_share'] = np.nan
    df['ridge_share'] = np.nan
    df['composite_score'] = np.nan

    for race_num, race_df in df.groupby('race'):
        idx = race_df.index

        # Logistic share — renormalize per race in case ensemble_prob drifted
        probs = race_df[prob_col].values
        valid = ~np.isnan(probs)
        if valid.any() and np.nansum(probs) > 0:
            df.loc[idx, 'logistic_share'] = np.where(
                valid, probs / np.nansum(probs), np.nan
            )

        # Ridge share — apply a small positive floor (to avoid zeros /
        # div-by-zero in degenerate races), then normalize. No clipping of
        # negatives: in practice ridge values are almost always positive
        # because the intercept is positive and every horse has some
        # positive pull through its win-prob features. On the rare
        # occasion ridge predicts slightly negative, the floor treats it
        # as near-zero value without masking the signal entirely.
        RIDGE_FLOOR = 0.5
        race_has_ridge = False
        if has_ridge:
            rv = race_df[ridge_col].values.astype(float)
            floored = np.where(np.isnan(rv), RIDGE_FLOOR,
                               np.maximum(rv, RIDGE_FLOOR))
            total = floored.sum()
            if total > 0:
                df.loc[idx, 'ridge_share'] = floored / total
                race_has_ridge = True

        # Composite: blend if ridge present for this race, else logistic only
        ls = df.loc[idx, 'logistic_share'].fillna(0).values
        if race_has_ridge:
            rs = df.loc[idx, 'ridge_share'].fillna(0).values
            composite = alpha * ls + (1 - alpha) * rs
        else:
            composite = ls
        df.loc[idx, 'composite_score'] = composite

    return df


# =============================================================================
# 3. TICKET OPTIMIZER (Pareto frontier on composite score)
# =============================================================================

def optimize_pick_tickets(df, race_numbers, alpha=0.5,
                          max_cost=None, min_score=None,
                          budget_steps=250):
    """
    Optimize Pick N tickets by maximizing composite heuristic at each cost.

    Ticket score = sum of composite_score across all selected horses.
    Ticket cost  = product of selection counts per race.

    For each allocation (n_1, n_2, ..., n_K):
      - Take the top n_i composite-scoring horses in race i
      - Sum their composite scores
      - Multiply the n_i's to get cost

    Pareto frontier: for each cost level, keep the highest-scoring ticket,
    then prune tickets that raise cost without raising score.

    Reports ensemble_prob-based hit rate as a secondary metric (not the
    optimization target).
    """
    df_pick = df[df['race'].isin(race_numbers)].copy()

    if 'composite_score' not in df_pick.columns:
        raise RuntimeError("build_composite_score() must be called first")

    # Sort horses within each race by composite score (descending)
    races = sorted(race_numbers)
    sorted_horses = {}       # race_num -> [(horse_number, score, ensemble_prob), ...]
    for race_num in races:
        race_data = df_pick[df_pick['race'] == race_num]
        race_data = race_data.dropna(subset=['composite_score'])
        if race_data.empty:
            continue
        triples = list(zip(
            race_data['horse_number'],
            race_data['composite_score'],
            race_data.get('ensemble_prob', pd.Series([np.nan] * len(race_data))),
        ))
        triples.sort(key=itemgetter(1), reverse=True)
        sorted_horses[race_num] = triples

    active_races = [r for r in races if r in sorted_horses]
    if len(active_races) < 2:
        return []

    max_horses = [len(sorted_horses[r]) for r in active_races]
    min_selection = [1] * len(active_races)

    # Enumerate all allocations
    all_allocations = []

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

    for total in range(sum(min_selection), sum(max_horses) + 1):
        base = min_selection.copy()
        remaining = total - sum(base)
        distribute(base, remaining)

    # Evaluate each allocation
    results = []
    for allocation in all_allocations:
        cost = int(np.prod(allocation))
        if max_cost is not None and cost > max_cost:
            continue

        ticket_score = 0.0
        hit_prob = 1.0
        selections = {}

        for i, race_num in enumerate(active_races):
            top = sorted_horses[race_num][:allocation[i]]
            race_score = sum(s for _, s, _ in top)
            race_hit = sum(p for _, _, p in top if not np.isnan(p))

            ticket_score += race_score
            hit_prob *= race_hit if race_hit > 0 else 1e-6
            selections[race_num] = [int(h) for h, _, _ in top]

        if min_score is not None and ticket_score < min_score:
            continue

        results.append({
            'cost': cost,
            'score': ticket_score,
            'hit_prob': hit_prob,
            'selections': selections,
            'allocation': allocation,
        })

    # Pareto frontier on (cost, score): for each cost, keep best score
    best_at_cost = {}
    for r in results:
        if r['cost'] not in best_at_cost or r['score'] > best_at_cost[r['cost']]['score']:
            best_at_cost[r['cost']] = r

    # Sort ascending by cost, prune dominated points
    sorted_by_cost = sorted(best_at_cost.values(), key=lambda x: x['cost'])
    frontier = []
    max_score_seen = -np.inf
    for r in sorted_by_cost:
        if r['score'] > max_score_seen:
            frontier.append(r)
            max_score_seen = r['score']

    # Sample frontier evenly if large
    if frontier and budget_steps > 0 and budget_steps < len(frontier):
        indices = np.linspace(0, len(frontier) - 1, budget_steps, dtype=int)
        return [frontier[i] for i in indices]

    return frontier


# =============================================================================
# 4. REPORTING
# =============================================================================

def print_race_probs(df, race_numbers, prob_col='ensemble_prob',
                     ml_col='ml_win_prob_norm', ridge_col='ridge_value',
                     score_col='composite_score'):
    """Print per-race table with model prob, ML, ridge value, composite score."""
    has_ml = ml_col in df.columns and df[ml_col].notna().any()
    has_ridge = ridge_col in df.columns and df[ridge_col].notna().any()

    for race_num in race_numbers:
        race = df[df['race'] == race_num].copy()
        if race.empty:
            continue
        # Sort by composite score (what the optimizer actually uses)
        race = race.sort_values(score_col, ascending=False)

        header = f"{'#':>3}  {'Model':>7}"
        if has_ml:
            header += f"  {'ML':>6}  {'Edge':>7}"
        if has_ridge:
            header += f"  {'Ridge$':>7}"
        header += f"  {'Score':>7}"

        print(f"\n--- Race {race_num} ({len(race)} horses) ---")
        print(header)

        for _, row in race.iterrows():
            parts = [f"{int(row['horse_number']):>3}",
                     f"{row[prob_col]:>6.1%}"]
            if has_ml:
                ml_val = row.get(ml_col, np.nan)
                edge = row.get('edge', np.nan)
                parts.append(f"{ml_val:>5.1%}" if pd.notna(ml_val) else f"{'—':>6}")
                parts.append(f"{edge:>+6.1%}" if pd.notna(edge) else f"{'—':>7}")
            if has_ridge:
                rv = row.get(ridge_col, np.nan)
                parts.append(f"{rv:>6.2f}" if pd.notna(rv) else f"{'—':>7}")
            sc = row.get(score_col, np.nan)
            parts.append(f"{sc:>6.1%}" if pd.notna(sc) else f"{'—':>6}")
            print("  ".join(parts))


def print_tickets(frontier, bet_type, top_n=10):
    """
    Print the top-N tickets from the Pareto frontier, sorted by composite
    score descending (best ticket first, regardless of cost).
    """
    if not frontier:
        print("  No tickets on frontier.")
        return

    # Sort descending by score — best ticket first
    sorted_f = sorted(frontier, key=lambda r: r['score'], reverse=True)
    selected = sorted_f[:top_n]

    print(f"\n{'='*70}")
    print(f"PICK {bet_type} — TOP {len(selected)} TICKETS BY COMPOSITE SCORE")
    print(f"{'='*70}")

    for i, r in enumerate(selected):
        header_line = (f"Option {i+1}: ${r['cost']} ticket | "
                       f"Score {r['score']:.3f} | "
                       f"Hit {r['hit_prob']:.2%}")
        print(f"\n{header_line}")
        print("-" * 55)
        for race_num in sorted(r['selections'].keys()):
            horses = r['selections'][race_num]
            print(f"  R{race_num}: {', '.join(str(h) for h in horses)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pick N ticket optimizer (composite logistic + ridge heuristic)")
    parser.add_argument('--date', type=str, required=True, help='Date (e.g. 2026-04-08)')
    parser.add_argument('--track', type=str, required=True, help='Track (e.g. keeneland)')
    parser.add_argument('--races', type=str, required=True,
                        help='Race range: "1-5" or "3,4,5,6,7"')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Composite blend: 1.0=all logistic share, '
                             '0.0=all ridge share (default 0.5)')
    parser.add_argument('--max-cost', type=int, default=None, help='Max ticket cost ($)')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Min composite score for a ticket')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of tickets to show (default 10)')
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        print(f"  WARNING: alpha={args.alpha} outside [0,1]; clamping")
        args.alpha = max(0.0, min(1.0, args.alpha))

    race_numbers = parse_race_range(args.races)
    bet_type = len(race_numbers)

    print(f"Pick {bet_type} Optimizer: {args.date} / {args.track}")
    print(f"  Races: {race_numbers}")
    print(f"  Alpha: {args.alpha}  "
          f"({args.alpha:.0%} logistic / {1-args.alpha:.0%} ridge value)")
    if args.max_cost:
        print(f"  Max cost: ${args.max_cost}")

    # Load predictions
    df = load_predictions(args.date, args.track)
    available_races = sorted(df['race'].dropna().unique())
    missing = [r for r in race_numbers if r not in available_races]
    if missing:
        print(f"\n  WARNING: Races {missing} not in predictions "
              f"(available: {available_races})")
        race_numbers = [r for r in race_numbers if r in available_races]
        if len(race_numbers) < 2:
            print("  Need at least 2 races. Exiting.")
            return

    # Build composite score
    if 'ridge_value' not in df.columns or not df['ridge_value'].notna().any():
        print(f"  No ridge_value column present — composite will be pure logistic")
    df = build_composite_score(df, alpha=args.alpha)

    # Print race tables (sorted by composite score)
    print_race_probs(df, race_numbers)

    # Run optimizer
    print(f"\nOptimizing Pick {len(race_numbers)} tickets...")
    frontier = optimize_pick_tickets(
        df, race_numbers,
        alpha=args.alpha,
        max_cost=args.max_cost,
        min_score=args.min_score,
        budget_steps=250,
    )

    if not frontier:
        print("No valid tickets found with given constraints.")
        return

    print(f"\n  Evaluated → {len(frontier)} frontier points")
    print_tickets(frontier, bet_type=len(race_numbers), top_n=args.top_n)


if __name__ == '__main__':
    main()