"""
check_payout_model.py

Sanity checks for ticket_ab.py's estimated returns, on every historical
Pick 4/5 with a payout. The payout model is refit leaving the card out.

1. Pool identity. In a pari-mutuel pool, payout(c) = (1 - takeout) / share of the
   pool on c, and the shares sum to 1. So the model's prices over ALL combos
   imply a takeout: 1 - t = 1 / sum_c E[1 / payout(c)]. It should land near the
   real takeout (Pick 5 ~15%, Pick 4 ~19-22% at these tracks).
2. All-combos ticket (a uniformly random ticket has the same expected return).
   Predicted with ticket_ab's estimator vs realized = winning payout / combos.
   With an efficient pool it would return 1 - takeout; the longshot bias pulls
   it below that.
3. Simple rectangular tickets (top 2 / top 3 per leg, by model or by ML):
   predicted vs realized = 1{winner covered} x payout / combos.

Usage:
    python check_payout_model.py
"""

import numpy as np
import pandas as pd

import fit_openrouter_blend as F
from diagnose_segments import loco_probs, FEATS

TAKEOUT = {4: (0.19, 0.22), 5: (0.15, 0.15)}   # published ranges, dated sources


def sequences():
    H = F.load_history(['geminiflash_blind'], marker='geminiflash_blind.csv')
    H = H.groupby('race_id').filter(lambda g: g[FEATS].notna().all().all()).reset_index(drop=True)
    H['p'] = loco_probs(H, FEATS)
    out = []
    for (date, track), g in H.groupby(['date', 'track']):
        meta = pd.read_csv(F.DATA_PATH / 'meta' / f'{track}_{date}.csv')
        for k in (4, 5):
            col = f'pick_{k}_$2_payout'
            if col not in meta:
                continue
            for _, r in meta[meta[col].notna()].iterrows():
                legs = [g[g['race'] == x].reset_index(drop=True)
                        for x in range(int(r['race']) - k + 1, int(r['race']) + 1)]
                if any(len(L) == 0 or L['win'].sum() != 1 for L in legs):
                    continue
                pay = float(str(r[col]).replace('$', '').replace(',', '')) / 2   # per $1
                out.append({'card': f'{date}/{track}', 'k': k, 'pay': pay, 'legs': legs})
    return out


KEEP = {4: 0.80, 5: 0.85}   # 1 - takeout used by the pool-identity model


def fit(seqs):
    """Model A: free log-linear fit on the winners' payouts."""
    win = [[L[L['win'] == 1].iloc[0] for L in s['legs']] for s in seqs]
    kk = np.array([s['k'] for s in seqs])
    y = np.log([s['pay'] for s in seqs])
    X = np.column_stack([kk == 4, kk == 5,
                         [-sum(np.log(w['ml']) for w in W) for W in win],
                         [-sum(np.log(w['p']) for w in W) for W in win]]).astype(float)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {4: c[0], 5: c[1]}, c[2], c[3], float(np.std(y - X @ c, ddof=4))


def public_share(L, beta, gamma):
    w = L['ml'] ** beta * L['p'] ** gamma
    return w / w.sum()


def fit_pool(seqs):
    """Model B: the public's share on each horse is ML^beta x model^gamma,
    normalized within the race; a combo pays keep / product of shares. The
    shares over all combos sum to 1, so the pool identity holds exactly; only
    the shape (beta, gamma) is fit, to the winners' log payouts."""
    from scipy.optimize import minimize
    idx = [[int(np.flatnonzero(L['win'].values)[0]) for L in s['legs']] for s in seqs]
    y = np.log([s['pay'] for s in seqs])

    def resid(th):
        return np.array([np.log(KEEP[s['k']]) - sum(np.log(public_share(L, *th).values[i])
                                                   for L, i in zip(s['legs'], ix))
                         for s, ix in zip(seqs, idx)]) - y

    th = minimize(lambda t: (resid(t) ** 2).sum(), x0=[0.5, 0.5], method='Nelder-Mead').x
    r = resid(th)
    return th[0], th[1], float(r.mean()), float(r.std(ddof=2))


def ci(x, n=4000, seed=1):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    seqs = sequences()
    rows, fits = [], {}
    for s in seqs:
        if s['card'] not in fits:
            rest = [o for o in seqs if o['card'] != s['card']]
            fits[s['card']] = (fit(rest), fit_pool(rest))
        (a, beta, gamma, sd), (bb, gb, bias, sdb) = fits[s['card']]
        n_all = np.prod([len(L) for L in s['legs']])
        win_idx = [int(np.flatnonzero(L['win'].values)[0]) for L in s['legs']]
        # Per-horse price: A = exp-scaled q^-beta p^-gamma (mean payout); B = keep / public share
        price = {'A': [np.exp((a[s['k']] + sd ** 2 / 2) / len(s['legs'])) * L['ml'] ** -beta * L['p'] ** -gamma
                       for L in s['legs']],
                 'B': [KEEP[s['k']] ** (1 / len(s['legs'])) / public_share(L, bb, gb) for L in s['legs']]}
        row = {'k': s['k'], 'card': s['card'], 'all_real': s['pay'] / n_all}
        for m, pr in price.items():
            row[f'keep_{m}'] = 1 / np.prod([(1 / x).sum() for x in pr])
            row[f'all_{m}'] = np.prod([(L['p'] * x).sum() for L, x in zip(s['legs'], pr)]) / n_all
        for by in ('p', 'ml'):
            for top in (2, 3):
                sets = [L[by].rank(ascending=False, method='first').values <= top for L in s['legs']]
                n = np.prod([m.sum() for m in sets])
                for m, pr in price.items():
                    row[f'{by}{top}_{m}'] = np.prod([(L['p'].values[st] * x.values[st]).sum()
                                                     for L, x, st in zip(s['legs'], pr, sets)]) / n
                row[f'{by}{top}_real'] = s['pay'] / n if all(st[i] for st, i in zip(sets, win_idx)) else 0.0
        rows.append(row)
    R = pd.DataFrame(rows)
    bb, gb, bias, sdb = fit_pool(seqs)
    a, beta, gamma, sd = fit(seqs)

    print(f"{len(R)} sequences ({(R.k == 4).sum()} Pick 4, {(R.k == 5).sum()} Pick 5); models refit "
          f"without each sequence's card. Returns are per $1 wagered.")
    print(f"  A (free fit on winners):   log pay = a_k + {beta:.2f} sum log(1/ML) + {gamma:.2f} sum log(1/model)")
    print(f"  B (pool identity built in): public share ~ ML^{bb:.2f} x model^{gb:.2f}, pay = keep / share; "
          f"winners pay {np.exp(-bias):.2f}x that on average (log sd {sdb:.2f})\n")
    print("1. Pool identity: 1 - takeout implied by the prices over all combos")
    for k in (4, 5):
        S = R[R.k == k]
        lo, hi = TAKEOUT[k]
        print(f"   Pick {k}: A {S['keep_A'].mean():.2f}   B {S['keep_B'].mean():.2f}   "
              f"published 1 - takeout {1 - hi:.2f}-{1 - lo:.2f}")

    print("\n2-3. Predicted vs realized return per $1")
    print(f"   {'ticket':<24}{'k':>3}{'pred A':>8}{'pred B':>8}{'realized':>10}  {'95% CI':<14}{'hit':>5}")
    for lab, key in (('All combos (= random)', 'all'), ('Top 2 by model', 'p2'), ('Top 3 by model', 'p3'),
                     ('Top 2 by ML', 'ml2'), ('Top 3 by ML', 'ml3')):
        for k in (4, 5):
            S = R[R.k == k]
            lo, hi = ci(S[f'{key}_real'])
            print(f"   {lab:<24}{k:>3}{S[f'{key}_A'].mean():>8.2f}{S[f'{key}_B'].mean():>8.2f}"
                  f"{S[f'{key}_real'].mean():>10.2f}  [{lo:.2f}, {hi:.2f}]{(S[f'{key}_real'] > 0).mean():>6.0%}")


if __name__ == '__main__':
    main()
