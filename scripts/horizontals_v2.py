"""
pick_optimizer.py

Optimize Pick 3/4/5/6 tickets using a 2D Pareto frontier over
(hit_probability, expected_value_per_dollar).

Reads predictions.csv output from predict_races.py — avoids re-running
the probability engine.

Three signals per horse (all normalized to per-race shares on [0,1]):
    - logistic_share: ensemble_prob (already sums to 1 per race)
    - ridge_share:    log-ridge value, floored and normalized per race
    - raw_share:      raw-ridge value (calibrated μ+σ from training),
                      shifted so all values are positive, then normalized

Composite score (3-way blend):
    score = w_log * logistic + w_rdg * ridge + w_raw * raw

Ticket metrics (the 2D frontier axes):
    - hit_prob:       product of per-race Σ(selected ensemble_prob)
    - ev_per_dollar:  expected $ return per $1 of ticket cost, using
                      raw-ridge predictions as per-horse payout proxy

The optimizer enumerates all allocations (how many horses per race),
computes (cost, hit_prob, ev_per_dollar) for each, and reports the
Pareto frontier over the two objectives. You can also constrain hit_prob
to a target range (e.g. 5–20% for Pick 5s) as a hard filter before the
frontier is computed — this is the recommended workflow when you want
tickets you'll actually hit occasionally.

Usage:
    python pick_optimizer.py --date 2026-04-08 --track keeneland --races 1-5
    python pick_optimizer.py --date 2026-04-08 --track keeneland --races 7-9 \
        --max-cost 500 --hit-min 0.05 --hit-max 0.20 \
        --w-log 0.4 --w-rdg 0.3 --w-raw 0.3

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

# Ridge break-even (matches predict_races.py: 4*2 + 2*2 + 1*2 = 14)
RIDGE_BREAKEVEN = 14.0


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

def build_composite_score(df, w_log=0.4, w_rdg=0.3, w_raw=0.3,
                          prob_col='ensemble_prob',
                          ridge_col='ridge_value',
                          raw_col='raw_value'):
    """
    Build 3-way composite score per horse, normalized within each race.

    Three signals, each converted to within-race shares that sum to 1:
        logistic share = ensemble_prob (already per-race normalized)
        ridge share    = max($0.50, ridge_value) / Σ
        raw share      = max(raw - (μ_card - k_floor·σ_card), 0.01) / Σ

    The raw-ridge share is the trickiest. Raw predictions from the
    calibrated model cluster tightly around μ (heavy regularization),
    and some horses are legitimately below μ. Treating those as "no
    signal" by flooring at 0 would throw away meaningful negative
    information. Instead we shift the race by (μ - k·σ) and floor at
    a small positive ε — this preserves the rank ordering and the
    σ-distance information while keeping everything positive for
    normalization. Horses more than k·σ below the mean contribute
    near-ε (minimal weight); horses above the mean contribute
    proportionally to their σ-distance.

    Composite: score = w_log * log + w_rdg * ridge + w_raw * raw.
    Weights are renormalized to sum to 1 over whatever signals are
    actually present in the data. So `--w-raw 0.3` on a card without
    raw-ridge becomes an effective "put zero on the missing signal,
    rebalance the remaining two."

    Returns df with logistic_share, ridge_share, raw_share, composite_score.
    """
    df = df.copy()

    has_logistic = prob_col in df.columns and df[prob_col].notna().any()
    has_ridge = ridge_col in df.columns and df[ridge_col].notna().any()
    has_raw = raw_col in df.columns and df[raw_col].notna().any()

    # Rebalance weights to sum to 1 over active signals
    w_active = {
        'log': w_log if has_logistic else 0,
        'rdg': w_rdg if has_ridge else 0,
        'raw': w_raw if has_raw else 0,
    }
    w_total = sum(w_active.values())
    if w_total == 0:
        raise RuntimeError("No active signals (logistic/ridge/raw) in predictions.csv")
    w_active = {k: v / w_total for k, v in w_active.items()}

    df['logistic_share'] = np.nan
    df['ridge_share'] = np.nan
    df['raw_share'] = np.nan
    df['composite_score'] = np.nan

    RIDGE_FLOOR = 0.5
    # How many σ below the card mean before a horse gets near-zero raw share
    RAW_SIGMA_FLOOR = 1.5

    # Card-wide raw mean/std (same calibration scope predict_v2 uses)
    raw_card_mean, raw_card_std = (np.nan, np.nan)
    if has_raw:
        raw_card_mean = df[raw_col].mean()
        raw_card_std = df[raw_col].std()
        # If the card is unusually flat (std≈0), force a tiny positive std
        # so the shift math doesn't blow up
        if not np.isfinite(raw_card_std) or raw_card_std < 1e-6:
            raw_card_std = 1e-6

    for race_num, race_df in df.groupby('race'):
        idx = race_df.index

        # Logistic share — renormalize per race in case ensemble_prob drifted
        probs = race_df[prob_col].values
        valid = ~np.isnan(probs)
        if valid.any() and np.nansum(probs) > 0:
            df.loc[idx, 'logistic_share'] = np.where(
                valid, probs / np.nansum(probs), np.nan
            )

        # Ridge share — small positive floor, then normalize
        race_has_ridge = False
        if has_ridge:
            rv = race_df[ridge_col].values.astype(float)
            floored = np.where(np.isnan(rv), RIDGE_FLOOR,
                               np.maximum(rv, RIDGE_FLOOR))
            total = floored.sum()
            if total > 0:
                df.loc[idx, 'ridge_share'] = floored / total
                race_has_ridge = True

        # Raw share — shift so all values are positive, normalize.
        # The shift is (μ_card - k·σ_card); horses that deep-negative get
        # near-ε weight. Using card-wide μ,σ rather than per-race because
        # the raw-ridge model is calibrated card-wide in predict_v2, so
        # "deviation from card mean" is the right scale.
        race_has_raw = False
        if has_raw:
            rw = race_df[raw_col].values.astype(float)
            shift = raw_card_mean - RAW_SIGMA_FLOOR * raw_card_std
            shifted = np.where(np.isnan(rw), 0.01, np.maximum(rw - shift, 0.01))
            total = shifted.sum()
            if total > 0:
                df.loc[idx, 'raw_share'] = shifted / total
                race_has_raw = True

        # Composite: blend active signals. Rebalance in case this specific
        # race is missing one (rare, but defensive).
        ls = df.loc[idx, 'logistic_share'].fillna(0).values
        rs = df.loc[idx, 'ridge_share'].fillna(0).values if race_has_ridge else np.zeros_like(ls)
        ws = df.loc[idx, 'raw_share'].fillna(0).values if race_has_raw else np.zeros_like(ls)

        w = {'log': w_active['log'] if has_logistic else 0,
             'rdg': w_active['rdg'] if race_has_ridge else 0,
             'raw': w_active['raw'] if race_has_raw else 0}
        wt = sum(w.values())
        if wt > 0:
            w = {k: v / wt for k, v in w.items()}
            composite = w['log'] * ls + w['rdg'] * rs + w['raw'] * ws
        else:
            composite = ls  # pathological, fall back to logistic
        df.loc[idx, 'composite_score'] = composite

    return df


# =============================================================================
# 3. TICKET OPTIMIZER (Pareto frontier on composite score)
# =============================================================================

def optimize_pick_tickets(df, race_numbers, alpha=None,
                          max_cost=None, min_score=None,
                          hit_min=None, hit_max=None,
                          budget_steps=250,
                          raw_col='raw_value', prob_col='ensemble_prob'):
    """
    Optimize Pick N tickets and return a 2D Pareto frontier over
    (hit_probability, ev_per_dollar).

    Cost is a constraint (--max-cost), not an objective. Hit probability
    and EV are the two axes of the frontier — a ticket is on the frontier
    if no other feasible ticket dominates it on both.

    Per-allocation computation:
      - cost       = Π n_i                        (product of selections)
      - hit_prob_i = Σ ensemble_prob of picked    (per-race hit chance)
      - hit_prob   = Π hit_prob_i                 (joint)
      - ev_factor_i = Σ(p·raw) / Σp of picks      (prob-weighted raw-$,
                     conditioned on one of your picks winning this race)
      - ev_proxy  = hit_prob × Π ev_factor_i      (expected $ payout proxy;
                    raw-ridge is WPS-$, not Pick-N-$, but correlates —
                    ticket-level EV in proportional units)
      - ev_per_$  = ev_proxy / cost

    Filters applied BEFORE building the frontier:
      - max_cost: drops tickets above budget
      - hit_min/hit_max: your requested 5–20% hit-probability window
                        — this is what you actually want for Pick-5 sanity

    Returns list of dicts, each with cost, hit_prob, ev_per_dollar,
    ev_proxy, score, selections, allocation.

    The `alpha` arg is deprecated; kept for back-compat so existing
    callers don't break. Use build_composite_score's 3-way weights instead.
    """
    df_pick = df[df['race'].isin(race_numbers)].copy()

    if 'composite_score' not in df_pick.columns:
        raise RuntimeError("build_composite_score() must be called first")

    has_raw = raw_col in df_pick.columns and df_pick[raw_col].notna().any()

    # Card-wide raw mean, used to normalize the EV factor.
    # Without this, ev_per_dollar scales with whatever the absolute raw-value
    # units are: $0.30-$0.90 (old normalized predict_v2) vs $4.00-$4.50
    # (new calibrated predict_v2) would give 5 orders of magnitude difference
    # in ev/$ for the same overlay quality. Dividing by card mean makes the
    # ev_factor a dimensionless "overlay multiplier" — 1.0 is average,
    # >1.0 means your picks lean toward overlay horses.
    raw_card_mean = df_pick[raw_col].mean() if has_raw else 1.0
    if not has_raw or not (raw_card_mean > 0):
        raw_card_mean = 1.0  # degrades ev_factor to weighted composite share

    # Sort horses within each race by composite score. Each entry is a
    # 4-tuple: (horse_number, composite_score, ensemble_prob, raw_value)
    races = sorted(race_numbers)
    sorted_horses = {}
    for race_num in races:
        race_data = df_pick[df_pick['race'] == race_num]
        race_data = race_data.dropna(subset=['composite_score'])
        if race_data.empty:
            continue
        rows = list(zip(
            race_data['horse_number'],
            race_data['composite_score'],
            race_data.get(prob_col, pd.Series([np.nan] * len(race_data))),
            race_data.get(raw_col, pd.Series([np.nan] * len(race_data))) if has_raw
                else pd.Series([np.nan] * len(race_data)),
        ))
        rows.sort(key=itemgetter(1), reverse=True)
        sorted_horses[race_num] = rows

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

        ticket_score = 0.0     # sum of composite scores (for score-based ranking)
        hit_prob = 1.0         # product of per-race hit chances
        ev_factor_product = 1.0  # product of per-race prob-weighted raw-$
        selections = {}

        for i, race_num in enumerate(active_races):
            top = sorted_horses[race_num][:allocation[i]]
            # score: sum of composite scores in the selection
            race_score = sum(s for _, s, _, _ in top)
            # hit: sum of ensemble_prob in selection
            race_hit = sum(p for _, _, p, _ in top if not np.isnan(p))
            # ev factor: Σ(p·raw) / Σp for this race's picks.
            # If raw-ridge missing for this race, use the prob-weighted
            # "score share" as a proxy (equivalent to saying "each horse
            # contributes its composite mass") — keeps the ranking honest.
            race_pw_raw = 0.0
            race_pw_total = 0.0
            for _, _, p, r in top:
                if np.isnan(p):
                    continue
                if has_raw and not np.isnan(r):
                    race_pw_raw += p * r
                else:
                    race_pw_raw += p * 1.0  # fallback: treat raw as $1
                race_pw_total += p
            race_ev_factor = (race_pw_raw / race_pw_total) if race_pw_total > 0 else 1.0
            # Normalize by card mean so ev_factor is a scale-invariant
            # "overlay multiplier" — 1.0 = average-value horse, >1 means
            # picks lean toward overlays, <1 toward underlays.
            race_ev_factor /= raw_card_mean

            ticket_score += race_score
            hit_prob *= max(race_hit, 1e-9)
            ev_factor_product *= max(race_ev_factor, 1e-9)
            selections[race_num] = [int(h) for h, _, _, _ in top]

        if min_score is not None and ticket_score < min_score:
            continue
        if hit_min is not None and hit_prob < hit_min:
            continue
        if hit_max is not None and hit_prob > hit_max:
            continue

        ev_proxy = hit_prob * ev_factor_product
        ev_per_dollar = ev_proxy / cost

        results.append({
            'cost': cost,
            'score': ticket_score,
            'hit_prob': hit_prob,
            'ev_proxy': ev_proxy,
            'ev_per_dollar': ev_per_dollar,
            'ev_factor_product': ev_factor_product,
            'selections': selections,
            'allocation': allocation,
        })

    if not results:
        return []

    # 2D Pareto frontier over (hit_prob, ev_per_dollar).
    # A ticket is dominated if another ticket has
    #   hit_prob >= self.hit_prob AND ev_per_dollar >= self.ev_per_dollar
    #   (with at least one strict inequality).
    # Efficient O(n log n) method: sort by hit_prob desc (primary) and
    # ev_per_dollar desc (tiebreak), then sweep. A ticket survives if
    # its ev_per_dollar is strictly greater than any previously-seen
    # ticket with equal-or-higher hit_prob.
    sorted_results = sorted(results,
                           key=lambda r: (-r['hit_prob'], -r['ev_per_dollar']))
    frontier = []
    max_ev_seen = -np.inf
    for r in sorted_results:
        if r['ev_per_dollar'] > max_ev_seen + 1e-12:
            frontier.append(r)
            max_ev_seen = r['ev_per_dollar']

    # Sample frontier evenly if very large (preserving endpoints)
    if frontier and budget_steps > 0 and budget_steps < len(frontier):
        indices = np.linspace(0, len(frontier) - 1, budget_steps, dtype=int)
        frontier = [frontier[i] for i in indices]

    return frontier


# =============================================================================
# 4. REPORTING
# =============================================================================

def print_race_probs(df, race_numbers, prob_col='ensemble_prob',
                     ml_col='ml_win_prob_norm', ridge_col='ridge_value',
                     raw_col='raw_value', score_col='composite_score'):
    """Print per-race table with model prob, ML, ridge value, raw value, composite score."""
    has_ml = ml_col in df.columns and df[ml_col].notna().any()
    has_ridge = ridge_col in df.columns and df[ridge_col].notna().any()
    has_raw = raw_col in df.columns and df[raw_col].notna().any()

    # For raw σ-distance annotation, compute card-wide mean/std once
    raw_mean = df[raw_col].mean() if has_raw else np.nan
    raw_std = df[raw_col].std() if has_raw else np.nan

    for race_num in race_numbers:
        race = df[df['race'] == race_num].copy()
        if race.empty:
            continue
        race = race.sort_values(score_col, ascending=False)

        header = f"{'#':>3}  {'Model':>7}"
        if has_ml:
            header += f"  {'ML':>6}  {'Edge':>7}"
        if has_ridge:
            header += f"  {'Ridge$':>7}"
        if has_raw:
            header += f"  {'Raw$':>6}  {'σ':>5}"
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
            if has_raw:
                rw = row.get(raw_col, np.nan)
                parts.append(f"{rw:>5.2f}" if pd.notna(rw) else f"{'—':>6}")
                if pd.notna(rw) and pd.notna(raw_std) and raw_std > 0:
                    sig = (rw - raw_mean) / raw_std
                    parts.append(f"{sig:>+4.1f}σ")
                else:
                    parts.append(f"{'—':>5}")
            sc = row.get(score_col, np.nan)
            parts.append(f"{sc:>6.1%}" if pd.notna(sc) else f"{'—':>6}")
            print("  ".join(parts))


def print_tickets(frontier, bet_type, top_n=10, sort_by='ev_per_dollar'):
    """
    Print tickets from the 2D Pareto frontier.

    Each ticket has:
      - hit_prob:      chance you cash
      - ev_per_dollar: (hit × overlay_multiplier) / cost, a RELATIVE
                       value score. Not in dollars-per-dollar — the
                       raw-ridge proxy isn't calibrated to Pick-N
                       payouts. Use for ranking, not for P&L estimates.
      - overlay_mult:  product of per-race prob-weighted raw-ridge
                       values (normalized by card mean). 1.0 = your
                       picks are on average-value horses. 1.5 = your
                       picks average 50% above card-mean raw-$ (strong
                       overlay lean). Product across races.

    Tickets are sorted by `sort_by`. The frontier is already pruned,
    so every shown ticket is non-dominated — moving down the list
    trades one axis for the other.
    """
    if not frontier:
        print("  No tickets on frontier.")
        return

    if sort_by == 'ev_per_dollar':
        sorted_f = sorted(frontier, key=lambda r: r['ev_per_dollar'], reverse=True)
    elif sort_by == 'hit_prob':
        sorted_f = sorted(frontier, key=lambda r: r['hit_prob'], reverse=True)
    else:
        sorted_f = sorted(frontier, key=lambda r: r['score'], reverse=True)

    selected = sorted_f[:top_n]

    print(f"\n{'='*82}")
    print(f"PICK {bet_type} — TOP {len(selected)} TICKETS BY {sort_by.upper()}")
    print(f"  (frontier over hit_prob × ev_per_dollar; none dominates another)")
    print(f"{'='*82}")

    for i, r in enumerate(selected):
        # Scale ev/$ to a readable integer range. The raw number is
        # typically 1e-4 to 1e-3; ×1000 gives 0.1-10, which is legible.
        ev_score = r['ev_per_dollar'] * 1000
        header_line = (
            f"Option {i+1}: ${r['cost']} ticket  |  "
            f"Hit {r['hit_prob']:.1%}  |  "
            f"EV-score {ev_score:.2f}  |  "
            f"Overlay×{r['ev_factor_product']:.2f}"
        )
        print(f"\n{header_line}")
        print("-" * len(header_line))
        for race_num in sorted(r['selections'].keys()):
            horses = r['selections'][race_num]
            print(f"  R{race_num}: {', '.join(str(h) for h in horses)} ({len(horses)})")


def print_frontier_summary(frontier):
    """
    Print a compact summary of the full frontier so you can see the
    shape of the trade-off curve without printing every point.
    """
    if not frontier:
        return

    hit_vals = [r['hit_prob'] for r in frontier]
    ev_vals = [r['ev_per_dollar'] * 1000 for r in frontier]  # same scaling
    cost_vals = [r['cost'] for r in frontier]

    print(f"\n{'─'*82}")
    print(f"FRONTIER SUMMARY ({len(frontier)} non-dominated tickets)")
    print(f"{'─'*82}")
    print(f"  hit_prob range:  {min(hit_vals):.2%}  →  {max(hit_vals):.2%}")
    print(f"  EV-score range:  {min(ev_vals):.2f}  →  {max(ev_vals):.2f}")
    print(f"  cost range:      ${min(cost_vals)}  →  ${max(cost_vals)}")

    # Sample 5 points by hit_prob (ascending) so you can see the curve
    sorted_by_hit = sorted(frontier, key=lambda r: r['hit_prob'])
    n = len(sorted_by_hit)
    idxs = [int(i * (n - 1) / 4) for i in range(5)] if n >= 5 else list(range(n))

    print(f"\n  Sampled frontier (ascending by hit_prob):")
    print(f"  {'hit':>7}  {'EV-score':>8}  {'overlay':>8}  {'cost':>6}  {'alloc':>20}")
    for i in idxs:
        r = sorted_by_hit[i]
        alloc_str = 'x'.join(str(n) for n in r['allocation'])
        print(f"  {r['hit_prob']:>6.1%}  {r['ev_per_dollar']*1000:>8.2f}  "
              f"{r['ev_factor_product']:>7.2f}×  "
              f"${r['cost']:>5}  {alloc_str:>20}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pick N ticket optimizer (3-way blend + 2D Pareto frontier)")
    parser.add_argument('--date', type=str, required=True, help='Date (e.g. 2026-04-08)')
    parser.add_argument('--track', type=str, required=True, help='Track (e.g. keeneland)')
    parser.add_argument('--races', type=str, required=True,
                        help='Race range: "1-5" or "3,4,5,6,7"')
    # Three-way blend — weights are auto-normalized over active signals
    parser.add_argument('--w-log', type=float, default=0.4, dest='w_log',
                        help='Weight on logistic share (default 0.4)')
    parser.add_argument('--w-rdg', type=float, default=0.3, dest='w_rdg',
                        help='Weight on log-ridge share (default 0.3)')
    parser.add_argument('--w-raw', type=float, default=0.3, dest='w_raw',
                        help='Weight on raw-ridge share (default 0.3)')
    # Back-compat: --alpha maps to logistic vs log-ridge, zeroes raw
    parser.add_argument('--alpha', type=float, default=None,
                        help='Back-compat: if set, maps to --w-log=alpha '
                             '--w-rdg=(1-alpha) --w-raw=0')
    # Constraints
    parser.add_argument('--max-cost', type=int, default=None, help='Max ticket cost ($)')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Min composite score for a ticket')
    parser.add_argument('--hit-min', type=float, default=None,
                        help='Hard filter: minimum ticket hit probability '
                             '(e.g. 0.05 = 5%% — gets rid of longshot-heavy tickets)')
    parser.add_argument('--hit-max', type=float, default=None,
                        help='Hard filter: maximum ticket hit probability '
                             '(e.g. 0.20 = 20%% — gets rid of "bet every favorite" tickets)')
    # Output
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of tickets to show (default 10)')
    parser.add_argument('--sort', type=str, default='ev_per_dollar',
                        choices=['ev_per_dollar', 'hit_prob', 'score'],
                        help='Sort order for displayed tickets (default ev_per_dollar)')
    parser.add_argument('--show-frontier', action='store_true',
                        help='Print frontier summary (hit × ev range + samples)')
    args = parser.parse_args()

    # Back-compat: --alpha overrides --w-log/--w-rdg/--w-raw
    if args.alpha is not None:
        if not 0.0 <= args.alpha <= 1.0:
            print(f"  WARNING: alpha={args.alpha} outside [0,1]; clamping")
            args.alpha = max(0.0, min(1.0, args.alpha))
        args.w_log = args.alpha
        args.w_rdg = 1.0 - args.alpha
        args.w_raw = 0.0
        print(f"  (--alpha set: w_log={args.w_log}, w_rdg={args.w_rdg}, w_raw=0)")

    race_numbers = parse_race_range(args.races)
    bet_type = len(race_numbers)

    print(f"Pick {bet_type} Optimizer: {args.date} / {args.track}")
    print(f"  Races: {race_numbers}")
    print(f"  Weights (pre-normalize): log={args.w_log} rdg={args.w_rdg} raw={args.w_raw}")
    if args.max_cost:
        print(f"  Max cost: ${args.max_cost}")
    if args.hit_min or args.hit_max:
        lo = f"{args.hit_min:.1%}" if args.hit_min else "—"
        hi = f"{args.hit_max:.1%}" if args.hit_max else "—"
        print(f"  Hit-prob window: {lo} to {hi}")

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

    # Diagnostic: what signals are available?
    has_rdg = 'ridge_value' in df.columns and df['ridge_value'].notna().any()
    has_raw = 'raw_value' in df.columns and df['raw_value'].notna().any()
    if not has_rdg:
        print(f"  (no ridge_value — log-ridge weight will be redistributed)")
    if not has_raw:
        print(f"  (no raw_value — raw-ridge weight will be redistributed; "
              f"re-run predict_v2.py after updating raw_value_model.json)")

    # Build composite score
    df = build_composite_score(df, w_log=args.w_log, w_rdg=args.w_rdg, w_raw=args.w_raw)

    # Print race tables
    print_race_probs(df, race_numbers)

    # Run optimizer — 2D Pareto over (hit_prob, ev_per_dollar)
    print(f"\nOptimizing Pick {len(race_numbers)} tickets...")
    frontier = optimize_pick_tickets(
        df, race_numbers,
        max_cost=args.max_cost,
        min_score=args.min_score,
        hit_min=args.hit_min,
        hit_max=args.hit_max,
        budget_steps=250,
    )

    if not frontier:
        print("No valid tickets found with given constraints.")
        print("  (try relaxing --hit-min/--hit-max or --max-cost)")
        return

    print(f"\n  → {len(frontier)} non-dominated tickets on (hit_prob, EV/$) frontier")

    if args.show_frontier:
        print_frontier_summary(frontier)

    print_tickets(frontier, bet_type=len(race_numbers),
                  top_n=args.top_n, sort_by=args.sort)


if __name__ == '__main__':
    main()