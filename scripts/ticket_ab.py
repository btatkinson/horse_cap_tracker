"""
ticket_ab.py

Build Pick-N tickets as A/B part-wheels from win projections, on the Pareto
frontier of hit rate and value.

Structure: a main ticket using each leg's A horses, plus one extra line per
leg that swaps in that leg's B horses (A everywhere else). Compared to one
rectangular ticket, this stops paying for combos where two or more "B-grade"
horses win, which buys more hit probability per dollar.
(On 2026-10-08's early P5: 9.0% vs 7.9% hit for the same $150.)

Two objectives, both under the model's win probabilities:
    hit    P(the tickets cash)
    value  expected return per $1: sum over the combos bought of
           P(combo) x estimated payout(combo), divided by the combos bought
The payout model is pari-mutuel by construction. The public's share of the
pool on each horse is modeled as q^beta * p^gamma, normalized within the race
(q = ML chance normalized over the field, p = held-out model probability). A
combo pays (1 - takeout) / product of the leg shares. The shares over all combos
sum to 1, so any ticket returns exactly (1 - takeout) under the public's
probabilities. beta and gamma are fit to the winners' log payouts in every
earlier Pick 4/5 in data/meta.

Fit on 113 sequences (2026-10-12): beta 0.37, gamma 0.71, i.e. the public's
money tracks the model more than the ML. check_payout_model.py compares it
with a free log-linear fit on winners. The free fit implied a 52% takeout for
Pick 5s and underpredicted all-combos tickets: 0.37 per $1 against 0.70
realized, versus 0.77 for this model. Top-2-by-ML tickets realized below both
models, but on only 1-3 hits, so that is not evidence either way.

The trade-off theta runs from 0 to 1. Each leg's horses are ranked by
p x price^theta, and structures maximize (1 - theta) log(hit) + theta log(value).
theta = 0 is pure hit rate (the old behaviour); theta = 1 is pure expected
return. The frontier keeps the results over a theta grid that no other result
beats on both hit and value. A longshot the model likes enters smoothly as
theta rises; there is no cutoff.

Pair mode (--pair): two tickets chosen together, splitting --budget. At each
theta, ticket 1 is the best structure in half the budget; ticket 2 is the one
that most improves the pair's objective given ticket 1 (a combo on both
tickets is held twice). Each ticket is capped at --max-lines lines.

Usage:
    python ticket_ab.py --date 2026-10-10 --track keeneland --races 6-10 --budget 160 --pair
    python ticket_ab.py ... --pair --frontier           # whole frontier (default: theta 0.85 only)
    python ticket_ab.py ... --proj projections_openrouter.csv,projections_capper_late.csv  # averaged

Reads: {DATA_PATH}/predictions/{date}/{track}/{proj files}
       {DATA_PATH}/meta/*.csv + entries/*.csv (payout model, cards before --date)
"""

import os, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
THETAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def parse_races(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def ml_to_prob(s):
    m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*[/-]\s*(\d+(?:\.\d+)?)\s*$', str(s))
    return np.nan if not m else 1.0 / (float(m.group(1)) / float(m.group(2)) + 1.0)


TAKEOUT = {5: 0.15}     # Pick 5 at Keeneland/Gulfstream; Pick 3/4/6 ~0.19-0.22
DEFAULT_TAKEOUT = 0.20


def public_share(q, p, beta, gamma):
    w = q ** beta * p ** gamma
    return w / w.sum()


def fit_payout_model(before_date):
    """Fit the public-share shape (beta, gamma) to the leg winners' payouts in
    every Pick 4/5 on cards before `before_date`, with each combo paying
    (1 - takeout) / product of shares. p is the blend refit without that card."""
    import fit_openrouter_blend as F
    from diagnose_segments import loco_probs, FEATS
    from scipy.optimize import minimize
    H = F.load_history(['geminiflash_blind'], marker='geminiflash_blind.csv')
    H = H[H['date'] < before_date]
    H = H.groupby('race_id').filter(lambda g: g[FEATS].notna().all().all()).reset_index(drop=True)
    H['p'] = loco_probs(H, FEATS)
    seqs = []
    for (date, track), g in H.groupby(['date', 'track']):
        meta = pd.read_csv(DATA_PATH / 'meta' / f'{track}_{date}.csv')
        for k in (4, 5):
            col = f'pick_{k}_$2_payout'
            if col not in meta:
                continue
            for _, r in meta[meta[col].notna()].iterrows():
                legs = [g[g['race'] == x] for x in range(int(r['race']) - k + 1, int(r['race']) + 1)]
                if any(len(L) == 0 or L['win'].sum() != 1 for L in legs):
                    continue
                pay = float(str(r[col]).replace('$', '').replace(',', '')) / 2
                seqs.append((k, pay, legs))
    if len(seqs) < 20:
        raise SystemExit(f"Only {len(seqs)} earlier Pick 4/5 payouts; not enough to fit the payout model")
    idx = [[int(np.flatnonzero(L['win'].values)[0]) for L in legs] for _, _, legs in seqs]
    y = np.log([pay for _, pay, _ in seqs])
    keep = np.log([1 - TAKEOUT.get(k, DEFAULT_TAKEOUT) for k, _, _ in seqs])

    def resid(th):
        return keep - np.array([sum(np.log(public_share(L['ml'].values, L['p'].values, *th)[i])
                                    for L, i in zip(legs, ix)) for (_, _, legs), ix in zip(seqs, idx)]) - y

    th = minimize(lambda t: (resid(t) ** 2).sum(), x0=[0.5, 0.5], method='Nelder-Mead').x
    r = resid(th)
    return {'beta': float(th[0]), 'gamma': float(th[1]), 'n': len(seqs),
            'winner_ratio': float(np.exp(-r.mean())), 'sd': float(r.std(ddof=2))}


def load_legs(date, track, races, proj_files, beta, gamma):
    """Per leg: DataFrame with p (averaged over projection files), q (ML chance
    normalized over the field) and price = 1 / the public's modeled share."""
    d = DATA_PATH / 'predictions' / date / track
    frames = [pd.read_csv(d / f) for f in proj_files]
    base = frames[0][['race', 'horse_number', 'horse_name', 'ml_prob', 'ml_odds']].copy()
    base['p'] = np.mean([f.set_index(['race', 'horse_number']).loc[
        list(zip(base.race, base.horse_number)), 'proj_prob'].values for f in frames], axis=0)
    legs = []
    for r in races:
        g = base[base.race == r].copy()
        g['p'] = g['p'] / g['p'].sum()
        g['q'] = g['ml_prob'] / g['ml_prob'].sum()
        g['price'] = 1 / public_share(g['q'].values, g['p'].values, beta, gamma)
        legs.append(g.reset_index(drop=True))
    return legs


def order_legs(legs, theta):
    """Rank each leg's horses by p x price^theta (theta = 0: by win probability)."""
    return [g.assign(key=g['p'] * g['price'] ** theta)
             .sort_values(['key', 'p'], ascending=False).reset_index(drop=True) for g in legs]


def leg_options(g, a_max, b_max):
    cp = np.concatenate([[0], np.cumsum(g['p'].values)])
    cv = np.concatenate([[0], np.cumsum(g['p'].values * g['price'].values)])
    n = len(g)
    opts = [(a, b, cp[a], cp[a + b] - cp[a], cv[a], cv[a + b] - cv[a])
            for a in range(1, min(a_max, n) + 1) for b in range(0, b_max + 1) if a + b <= n]
    return np.array(opts, dtype=float)  # columns: nA, nB, pA, pB, sA, sB


def search(legs, base, lo, hi, scale, a_max=6, b_max=5):
    """Every A/B structure with cost in [lo, hi]: hit, value (est. return per $1), cost."""
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
            keep.append((hit[t], scale * ret[t] / combos[t], cost[t],
                         tuple(int(opts[j][idx[j][t], 0]) for j in range(k)),
                         tuple(int(opts[j][idx[j][t], 1]) for j in range(k))))
    T = pd.DataFrame(keep, columns=['hit', 'value', 'cost', 'a', 'b'])
    T['lines'] = T['b'].map(lambda b: 1 + sum(x > 0 for x in b))
    return T


def objective(hit, value, theta):
    return (1 - theta) * np.log(hit) + theta * np.log(value)


def outer(vs, op):
    m = vs[0]
    for v in vs[1:]:
        m = op.outer(m, v)
    return m


def coverage(legs, a, b):
    """Boolean array over every winner combination (one axis per leg): True
    where the A/B ticket (main + one backup line per leg) cashes."""
    k = len(legs)
    idx = [np.arange(len(g)) for g in legs]
    A = [idx[i] < a[i] for i in range(k)]
    B = [(idx[i] >= a[i]) & (idx[i] < a[i] + b[i]) for i in range(k)]
    M = outer(A, np.logical_and)
    for i in range(k):
        if b[i]:
            M |= outer([B[j] if j == i else A[j] for j in range(k)], np.logical_and)
    return M


def best_pair(legs, T, theta, scale, scan=1500):
    """Ticket 1 = best single structure at theta; ticket 2 = best addition to it."""
    P = outer([g['p'].values for g in legs], np.multiply)
    PAY = scale * outer([g['price'].values for g in legs], np.multiply)
    T = T.assign(J=objective(T['hit'], T['value'], theta)).sort_values('J', ascending=False)
    t1 = T.iloc[0]
    M1 = coverage(legs, t1['a'], t1['b'])
    best = None
    for _, r in T.head(scan).iterrows():
        if r['a'] == t1['a'] and r['b'] == t1['b']:
            continue   # the same ticket twice adds no coverage
        M2 = coverage(legs, r['a'], r['b'])
        hit = P[M1 | M2].sum()
        value = (P * PAY * (M1.astype(int) + M2)).sum() / (M1.sum() + M2.sum())
        J = objective(hit, value, theta)
        if best is None or J > best[0]:
            best = (J, r, hit, value)
    return t1, best[1], best[2], best[3]


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


def used_horses(legs, tickets):
    """(leg index, row) for every horse on any line of the given tickets."""
    seen = set()
    for t in tickets:
        for i, g in enumerate(legs):
            seen |= {(i, j) for j in range(t['a'][i] + t['b'][i])}
    return seen


def main():
    parser = argparse.ArgumentParser(description='A/B part-wheel Pick-N tickets on the hit/value frontier')
    parser.add_argument('--date', required=True)
    parser.add_argument('--track', required=True)
    parser.add_argument('--races', required=True, help='e.g. 6-10')
    parser.add_argument('--budget', type=float, default=150.0)
    parser.add_argument('--base', type=float, default=0.50)
    parser.add_argument('--takeout', type=float, default=None,
                        help='Pool takeout (default 0.15 for a Pick 5, 0.20 otherwise)')
    parser.add_argument('--proj', default='projections_openrouter.csv',
                        help='Projection file(s) to average, comma-separated')
    parser.add_argument('--pair', action='store_true',
                        help='Two complementary tickets splitting --budget (see module docstring)')
    parser.add_argument('--max-lines', type=int, default=3,
                        help='Max lines per ticket (main + backup lines)')
    parser.add_argument('--theta', type=float, default=0.85,
                        help='Trade-off to build tickets at (0 = hit rate, 1 = value). Default 0.85, '
                             'chosen 2026-10-12 from the frontier; pending the backtest')
    parser.add_argument('--frontier', action='store_true',
                        help='Show the whole frontier (theta grid) instead of one theta')
    args = parser.parse_args()

    races = parse_races(args.races)
    k = len(races)
    pm = fit_payout_model(args.date)
    takeout = args.takeout if args.takeout is not None else TAKEOUT.get(k, DEFAULT_TAKEOUT)
    scale = 1 - takeout       # payout per $1 = (1 - takeout) x product of leg prices
    print(f"Payout model ({pm['n']} earlier Pick 4/5s): public share ~ ML^{pm['beta']:.2f} x model^{pm['gamma']:.2f} "
          f"per race; a combo pays (1 - {takeout:.0%}) / product of shares, so a ticket returns "
          f"{scale:.2f} under the public's own odds")
    legs = load_legs(args.date, args.track, races, [p.strip() for p in args.proj.split(',')],
                     pm['beta'], pm['gamma'])
    # Sanity check: every combo once. Under the public's own odds (shares) this
    # returns exactly 1 - takeout; under the model it can differ.
    n_all = np.prod([len(g) for g in legs])
    pub = scale * np.prod([(1 / g['price']).mul(g['price']).sum() for g in legs]) / n_all
    mod = scale * np.prod([(g['p'] * g['price']).sum() for g in legs]) / n_all
    print(f"All-combos ticket ({n_all:,} combos): est. return {pub:.2f} per $1 under the public's odds, "
          f"{mod:.2f} under the model")
    for r, g in zip(races, legs):
        g = g.sort_values('p', ascending=False)
        print(f"R{r}: " + "  ".join(f"#{h} {n[:14]} {p:.0%} (ML {ml}, x{p / q:.2f})"
                                    for h, n, p, ml, q in zip(g.horse_number, g.horse_name, g.p, g.ml_odds, g.q)
                                    if p >= 0.04))

    thetas = THETAS if args.frontier else [args.theta]
    budget = args.budget / 2 if args.pair else args.budget
    results = []
    for th in thetas:
        L = order_legs(legs, th)
        T = search(L, args.base, 0.87 * budget, 1.03 * budget, scale)
        T = T[T['lines'] <= args.max_lines]
        if T.empty:
            continue
        if args.pair:
            t1, t2, hit, value = best_pair(L, T, th, scale)
            tickets = (t1, t2)
        else:
            t1 = T.loc[objective(T['hit'], T['value'], th).idxmax()]
            tickets, hit, value = (t1,), t1['hit'], t1['value']
        used = used_horses(L, tickets)
        longest = max(1 / L[i].loc[j, 'q'] - 1 for i, j in used)
        results.append(dict(theta=th, hit=hit, value=value, cost=sum(t['cost'] for t in tickets),
                            horses=len(used), longest=longest, tickets=tickets, legs=L))
    if not results:
        raise SystemExit("No structures in budget")
    for r in results:
        r['frontier'] = not any(o['hit'] >= r['hit'] and o['value'] >= r['value'] and
                                (o['hit'] > r['hit'] or o['value'] > r['value']) for o in results)

    print(f"\n{'theta':>6}{'hit':>8}{'est. return/$1':>16}{'cost':>9}{'horses':>8}{'longest (field-adj ML)':>24}")
    for r in results:
        print(f"{r['theta']:>6.2f}{r['hit']:>8.1%}{r['value']:>16.2f}{r['cost']:>9.2f}{r['horses']:>8}"
              f"{r['longest']:>19.0f}-1{'' if r['frontier'] else '   (dominated)'}")
    print("Hit and return are under the model's probabilities; the return also depends on the payout model.")

    for r in results:
        if not r['frontier']:
            continue
        print(f"\ntheta {r['theta']:.2f}: hit {r['hit']:.1%}, est. return {r['value']:.2f} per $1, ${r['cost']:.2f}")
        for n, t in zip(('Ticket 1', 'Ticket 2'), r['tickets']):
            print(f"  {n}: ${t['cost']:.2f}, {t['hit']:.1%} alone")
            print(slip(r['legs'], races, t, args.base))


if __name__ == '__main__':
    main()
