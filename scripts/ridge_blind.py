"""
ridge_blind.py

Stage 1.5: one odds-blind score from every LLM signal, by ridge regression.
The ML is not used in this step. Stage 2 then adds the ML back.

Features (per horse, every blind model x prompt):
  lengths   lengths-vs-median prompt, centered within the race
  beyer     expected Beyer, centered within the race
  winprob   log of the normalized win prob, centered within the race
Each is scaled by its SD, so one alpha shrinks every feature alike. A horse a
model skipped gets that model's race min. A race a model never saw gets 0 (no
information).

Target: finish position as a normal score (Blom), i.e. the expected z-value of
that order statistic in a field of n. It is the closest thing to "lengths
gained" the order of finish supports (HRN publishes no margins).

Out of sample = leave-one-card-out. On the held-out card:
  rho     per-race Spearman of the score vs the finish order
  top PG  positions gained vs the median horse by the score's top pick
  R2      on the Blom target
Stage 2 is nested, so every training card's ridge score is itself
out-of-sample. Its form is P ∝ ML^a × exp(b × score), scored by winner
log-loss against ML alone and the saved blend's features, on the same races.

Races where first-time starters make up --max-fts or more of the field are
dropped from fitting and scoring (default 0.30).

Usage:
    python ridge_blind.py
    python ridge_blind.py --max-fts 1.01      # keep every race
    python ridge_blind.py --no-opus           # Opus has no fall picks
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import fit_openrouter_blend as F
from diagnose_segments import attach_race_info

PROMPTS = {'lengths': '_lengths', 'beyer': '_expected_beyer', 'winprob': ''}
ALPHAS = [0, 10, 100, 300, 1000, 3000, 10000, 30000, 100000]


def centered(H, s):
    """Fill skipped horses with the race min, center within the race; races with no data -> 0."""
    filled = s.fillna(s.groupby(H['race_id']).transform('min'))
    return (filled - filled.groupby(H['race_id']).transform('mean')).fillna(0.0)


def build_features(H, providers):
    cols, names = [], []
    for p in providers:
        for prompt, suf in PROMPTS.items():
            raw = H[f'{p}{suf}'] if suf else np.log(H[p])
            covered = raw.notna().groupby(H['race_id']).transform('any')
            if covered.groupby(H['race_id']).first().mean() < 0.5:
                continue
            c = centered(H, raw)
            sd = c[covered].std()
            cols.append((c / sd).values)
            names.append(f"{p.replace('_blind', '')} {prompt}")
    return np.column_stack(cols), names


def blom(H):
    n = H.groupby('race_id')['finish'].transform('size')
    return norm.ppf(((n - H['finish'] + 1 - 0.375) / (n + 0.25)).values)


def ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def race_metrics(H, score):
    """Per race: Spearman vs finish, positions gained by the top-scored horse."""
    rows = []
    for rid, g in H.assign(s=score).groupby('race_id', sort=False):
        fin = g['finish'].values
        rho = spearmanr(g['s'].values, -fin).correlation if np.ptp(g['s'].values) > 0 else 0.0
        rows.append((rid, rho, (len(g) + 1) / 2 - fin[np.argmax(g['s'].values)]))
    return pd.DataFrame(rows, columns=['race_id', 'rho', 'top_pg']).set_index('race_id')


def loco_ridge(H, X, y, alpha):
    s = np.zeros(len(H))
    for c in H['card'].unique():
        te = (H['card'] == c).values
        s[te] = X[te] @ ridge(X[~te], y[~te], alpha)
    return s


def conditional_logit_arrays(H, mask, feats):
    g = H[mask]
    rid = pd.factorize(g['race_id'])[0]
    return feats[mask], rid, g['win'].values, rid.max() + 1


def stage2(H, X, y, alpha):
    """Nested leave-one-card-out: P ∝ ML^a × exp(b × ridge score). Per-race winner loss."""
    logml = np.log(H['ml'].values)
    loss = pd.Series(np.nan, index=H['race_id'].unique())
    cards = H['card'].unique()
    for c in cards:
        te = (H['card'] == c).values
        s_in = np.zeros(len(H))
        for c2 in cards:
            if c2 == c:
                continue
            inner = (H['card'] == c2).values
            fit = ~te & ~inner
            s_in[inner] = X[inner] @ ridge(X[fit], y[fit], alpha)
        s_in[te] = X[te] @ ridge(X[~te], y[~te], alpha)
        Z = np.column_stack([logml, s_in])
        b = F.fit_nonneg(*conditional_logit_arrays(H, ~te, Z))
        Xo, ro, wo, no = conditional_logit_arrays(H, te, Z)
        loss.loc[pd.unique(H.loc[te, 'race_id'])] = F.race_losses(b, Xo, ro, wo, no)
    return loss


def stage2_fixed(H, feats):
    """Leave-one-card-out conditional logit on fixed log-prob features."""
    R = F.Races(H, feats)
    loss = {}
    for c in R.cards:
        Xi, Ri, Wi, ni = R.stack([x for x in R.cards if x != c], list(range(len(feats))))
        b = F.fit_nonneg(Xi, Ri, Wi, ni)
        Xo, Ro, Wo, no = R.stack([c], list(range(len(feats))))
        rids = pd.unique(H.loc[H['card'] == c, 'race_id'])
        loss.update(dict(zip(rids, F.race_losses(b, Xo, Ro, Wo, no))))
    return pd.Series(loss)


def ci(x, n=4000, seed=1):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description='Ridge over odds-blind LLM signals, out of sample')
    parser.add_argument('--max-fts', type=float, default=0.30,
                        help='Drop races where first-time starters are at least this share of the field')
    parser.add_argument('--no-opus', action='store_true')
    args = parser.parse_args()

    providers = [p for p in F.discover_providers() if not (args.no_opus and p.startswith('opus'))]
    H = F.load_history(providers, marker='geminiflash_blind.csv')
    H = attach_race_info(H)
    share = H.groupby('race_id')['fts'].transform('mean')
    n_all = H['race_id'].nunique()
    H = H[share < args.max_fts].reset_index(drop=True)
    X, names = build_features(H, providers)
    y = blom(H)
    print(f"{H['race_id'].nunique()} of {n_all} races / {H['card'].nunique()} cards kept "
          f"(dropped races with first-time starters >= {args.max_fts:.0%} of the field). "
          f"{len(names)} features: {', '.join(names)}")

    # Reference signals on the same races (no fitting)
    print("\nOut of sample, leave-one-card-out (rho = per-race Spearman vs finish; "
          "PG = top pick's positions gained vs the median horse):")
    print(f"  {'score':<34}{'rho':>7}{'top PG':>8}{'R2':>7}")
    ref = {'ML (reference, not in the ridge)': np.log(H['ml'].values),
           'Gemini Beyer alone': X[:, names.index('geminiflash beyer')],
           'Gemini lengths alone': X[:, names.index('geminiflash lengths')],
           'Gemini 3 prompts, equal weight': X[:, [i for i, n in enumerate(names) if 'geminiflash' in n]].mean(1),
           'All features, equal weight': X.mean(1)}
    for lab, s in ref.items():
        m = race_metrics(H, s)
        print(f"  {lab:<34}{m['rho'].mean():>7.3f}{m['top_pg'].mean():>+8.2f}{'':>7}")
    base = race_metrics(H, ref['Gemini lengths alone'])['rho']
    oos = {}
    for a in ALPHAS:
        s = loco_ridge(H, X, y, a)
        m = race_metrics(H, s)
        r2 = 1 - ((y - s) ** 2).sum() / (y ** 2).sum()
        oos[a] = m
        print(f"  {'ridge alpha=' + format(a, 'g'):<34}{m['rho'].mean():>7.3f}{m['top_pg'].mean():>+8.2f}{r2:>7.3f}")
    best = max(oos, key=lambda a: oos[a]['rho'].mean())
    d = oos[best]['rho'] - base
    lo, hi = ci(d)
    print(f"  Best alpha {best:g}: rho vs Gemini lengths alone {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")

    w = ridge(X, y, best)
    print(f"\nRidge weights at alpha={best:g}, fit on all kept cards (per 1 SD of each feature):")
    for n, v in sorted(zip(names, w), key=lambda t: -abs(t[1])):
        print(f"  {n:<24}{v:>+8.3f}")

    print("\nStage 2, winner log-loss, leave-one-card-out on the same races (lower is better):")
    ml_only = stage2_fixed(H, ['ml'])
    saved = stage2_fixed(H, ['ml', 'geminiflash_blind_bs', 'geminiflash_blind_len'])
    print(f"  {'ML only':<34}{ml_only.mean():>7.4f}")
    dd = saved - ml_only
    lo, hi = ci(dd.values)
    print(f"  {'ML + Gemini Beyer + lengths (saved)':<34}{saved.mean():>7.4f}  vs ML {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
    for a in sorted({best, 100, 1000, 10000}):
        l2 = stage2(H, X, y, a)
        dd = l2 - ml_only.reindex(l2.index)
        lo, hi = ci(dd.values)
        print(f"  {'ML + ridge (alpha=' + format(a, 'g') + ')':<34}{l2.mean():>7.4f}  vs ML {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")


if __name__ == '__main__':
    main()
