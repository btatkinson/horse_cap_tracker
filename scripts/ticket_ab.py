"""
ticket_ab.py

Build Pick-N tickets as A/B part-wheels from win projections.

Structure: a main ticket using each leg's A horses, plus one extra ticket
per leg that swaps in that leg's B horses (A everywhere else). Compared to
one rectangular ticket, this stops paying for combos where two or more
"B-grade" horses win, which buys more hit probability per dollar.
(On 2026-10-08's early P5: 9.0% vs 7.9% hit for the same $150.)

A = top a horses by projected win prob, B = the next b. Every (a, b) per leg
is enumerated; structures within the budget are scored on:
    hit   = P(ticket set cashes) under the projections
    EV    = expected return / cost, pricing each combo at
            base * (1 - takeout) / prod(ML-implied prob)  (ML as pool proxy)

Pair mode (--pair): two tickets chosen together, splitting --budget. Ticket 1
is the best hit rate in budget; ticket 2 is the structure adding the most new
coverage on top of it. Each ticket is capped at --max-lines lines (main + one
backup line per leg with B horses) to keep entry quick. On 10/09 the cap of 3
cost nothing on the late card.

Usage:
    python ticket_ab.py --date 2026-10-08 --track keeneland --races 6-10 --budget 150
    python ticket_ab.py ... --proj projections_openrouter.csv,projections_openrouter_blind.csv  # averaged
    python ticket_ab.py --date 2026-10-10 --track keeneland --races 1-5 --budget 160 --pair

Reads: {DATA_PATH}/predictions/{date}/{track}/{proj files}
"""

import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])


def parse_races(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_legs(date, track, races, proj_files):
    """Per leg: DataFrame sorted by p (averaged over projection files)."""
    d = DATA_PATH / 'predictions' / date / track
    frames = [pd.read_csv(d / f) for f in proj_files]
    base = frames[0][['race', 'horse_number', 'horse_name', 'ml_prob', 'ml_odds']].copy()
    base['p'] = np.mean([f.set_index(['race', 'horse_number']).loc[
        list(zip(base.race, base.horse_number)), 'proj_prob'].values for f in frames], axis=0)
    legs = []
    for r in races:
        g = base[base.race == r].copy()
        g['p'] = g['p'] / g['p'].sum()
        g['ml_prob'] = g['ml_prob'] / g['ml_prob'].sum()
        g['v'] = g['p'] / g['ml_prob']
        legs.append(g.sort_values('p', ascending=False).reset_index(drop=True))
    return legs


def leg_options(g, a_max, b_max):
    cp = np.concatenate([[0], np.cumsum(g['p'].values)])
    cv = np.concatenate([[0], np.cumsum(g['p'].values / g['ml_prob'].values)])
    n = len(g)
    opts = [(a, b, cp[a], cp[a + b] - cp[a], cv[a], cv[a + b] - cv[a])
            for a in range(1, min(a_max, n) + 1) for b in range(0, b_max + 1) if a + b <= n]
    return np.array(opts, dtype=float)  # columns: nA, nB, pA, pB, sA, sB


def search(legs, base, takeout, lo, hi, a_max=6, b_max=5):
    opts = [leg_options(g, a_max, b_max) for g in legs]
    k = len(legs)
    rest = np.meshgrid(*[np.arange(len(o)) for o in opts[1:]], indexing='ij')
    rest = [x.ravel() for x in rest]
    keep = []
    for i0 in range(len(opts[0])):
        idx = [np.full(rest[0].shape, i0)] + rest
        cols = [np.stack([opts[j][idx[j], c] for j in range(k)]) for c in range(6)]
        nA, nB, pA, pB, sA, sB = cols
        combos = np.prod(nA, 0); hit = np.prod(pA, 0); ret = np.prod(sA, 0)
        for j in range(k):
            m = (np.arange(k) == j)[:, None]
            combos = combos + np.prod(np.where(m, nB, nA), 0)
            hit = hit + np.prod(np.where(m, pB, pA), 0)
            ret = ret + np.prod(np.where(m, sB, sA), 0)
        cost = base * combos
        ok = (cost >= lo) & (cost <= hi)
        for t in np.flatnonzero(ok):
            keep.append((hit[t], (1 - takeout) * ret[t] / combos[t], cost[t],
                         tuple(int(opts[j][idx[j][t], 0]) for j in range(k)),
                         tuple(int(opts[j][idx[j][t], 1]) for j in range(k))))
    return pd.DataFrame(keep, columns=['hit', 'ev', 'cost', 'a', 'b'])


def describe(legs, races, a, b, base):
    A = [list(g['horse_number'].iloc[:a[i]]) for i, g in enumerate(legs)]
    B = [list(g['horse_number'].iloc[a[i]:a[i] + b[i]]) for i, g in enumerate(legs)]
    lines = []
    main = int(np.prod([len(x) for x in A]))
    lines.append(f"    Main: " + " / ".join(",".join(map(str, x)) for x in A) + f"   (${base*main:.2f})")
    for i in range(len(legs)):
        if not B[i]:
            continue
        legs_txt = [",".join(map(str, B[i] if j == i else A[j])) for j in range(len(legs))]
        n = len(B[i]) * int(np.prod([len(A[j]) for j in range(len(legs)) if j != i]))
        lines.append(f"    B in R{races[i]}: " + " / ".join(legs_txt) + f"   (${base*n:.2f})")
    return "\n".join(lines)


def coverage(legs, a, b):
    """Boolean array over every winner combination (one axis per leg): True
    where the A/B ticket set (main + one backup line per leg) cashes."""
    k = len(legs)
    idx = [np.arange(len(g)) for g in legs]
    A = [idx[i] < a[i] for i in range(k)]
    B = [(idx[i] >= a[i]) & (idx[i] < a[i] + b[i]) for i in range(k)]

    def outer(vs):
        m = vs[0]
        for v in vs[1:]:
            m = np.logical_and.outer(m, v)
        return m

    M = outer(A)
    for i in range(k):
        if b[i]:
            M |= outer([B[j] if j == i else A[j] for j in range(k)])
    return M


def best_pair(legs, T, scan=4000):
    """Ticket 1 = top hit rate; ticket 2 = the most new probability on top of it."""
    P = legs[0]['p'].values
    for g in legs[1:]:
        P = np.multiply.outer(P, g['p'].values)
    t1 = T.iloc[0]
    M1 = coverage(legs, t1['a'], t1['b'])
    gains = [P[coverage(legs, r['a'], r['b']) & ~M1].sum() for _, r in T.head(scan).iterrows()]
    t2 = T.iloc[int(np.argmax(gains))]
    return t1, t2, P[M1 | coverage(legs, t2['a'], t2['b'])].sum()


def slip(legs, races, t, base):
    """Ticket lines with horse numbers sorted, as entered on a betting app."""
    k = len(legs)
    A = [sorted(legs[i]['horse_number'].iloc[:t['a'][i]].astype(int).tolist()) for i in range(k)]
    B = [sorted(legs[i]['horse_number'].iloc[t['a'][i]:t['a'][i] + t['b'][i]].astype(int).tolist())
         for i in range(k)]
    rows = [('Main', A)] + [(f'R{races[i]} backups', [B[i] if j == i else A[j] for j in range(k)])
                            for i in range(k) if B[i]]
    out = []
    for name, sets in rows:
        n = int(np.prod([len(s) for s in sets]))
        out.append(f"    {name:<12} " + " / ".join(",".join(map(str, s)) for s in sets) + f"   ${base * n:.2f}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description='A/B part-wheel Pick-N ticket builder')
    parser.add_argument('--date', required=True)
    parser.add_argument('--track', required=True)
    parser.add_argument('--races', required=True, help='e.g. 6-10')
    parser.add_argument('--budget', type=float, default=150.0)
    parser.add_argument('--base', type=float, default=0.50)
    parser.add_argument('--takeout', type=float, default=0.15)
    parser.add_argument('--proj', default='projections_openrouter.csv',
                        help='Projection file(s) to average, comma-separated')
    parser.add_argument('--pair', action='store_true',
                        help='Two complementary tickets splitting --budget (see module docstring)')
    parser.add_argument('--max-lines', type=int, default=3,
                        help='Pair mode: max lines per ticket (main + backup lines)')
    args = parser.parse_args()

    races = parse_races(args.races)
    legs = load_legs(args.date, args.track, races, [p.strip() for p in args.proj.split(',')])
    for r, g in zip(races, legs):
        print(f"R{r}: " + "  ".join(f"#{h} {n[:14]} {p:.0%} (ML {ml}, v{v:.2f})"
                                    for h, n, p, ml, v in zip(g.horse_number, g.horse_name, g.p, g.ml_odds, g.v)
                                    if p >= 0.04))

    if args.pair:
        half = args.budget / 2
        T = search(legs, args.base, args.takeout, 0.87 * half, 1.03 * half, a_max=6, b_max=6)
        T['lines'] = T['b'].map(lambda b: 1 + sum(x > 0 for x in b))
        T = T[T['lines'] <= args.max_lines].sort_values('hit', ascending=False).reset_index(drop=True)
        if T.empty:
            raise SystemExit("No structures in budget")
        t1, t2, both = best_pair(legs, T)
        print(f"\nTwo tickets, {args.max_lines} lines max each: {both:.1%} hit together, "
              f"${t1['cost'] + t2['cost']:.2f} total")
        for name, t in (('Ticket 1', t1), ('Ticket 2', t2)):
            print(f"\n  {name}: ${t['cost']:.2f}, {t['hit']:.1%} alone, est. EV {t['ev']:.2f}x")
            print(slip(legs, races, t, args.base))
        return

    T = search(legs, args.base, args.takeout, 0.87 * args.budget, 1.03 * args.budget)
    if T.empty:
        raise SystemExit("No structures in budget")
    T = T.sort_values('hit', ascending=False).reset_index(drop=True)
    rect = T[T['b'].map(lambda b: sum(b) == 0)]
    picks = {
        'Max hit rate': T.iloc[0],
        'Balanced (max hit x EV)': T.loc[(T['hit'] * T['ev']).idxmax()],
    }
    if not rect.empty:
        picks['Best single ticket'] = rect.iloc[0]
    print(f"\n{len(T):,} A/B structures between ${0.87*args.budget:.0f} and ${1.03*args.budget:.0f}")
    for name, row in picks.items():
        print(f"\n{name}: hit {row['hit']:.2%}, est. EV {row['ev']:.2f}x, total ${row['cost']:.2f}")
        print(describe(legs, races, row['a'], row['b'], args.base))


if __name__ == '__main__':
    main()
