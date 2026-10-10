"""
rank_metrics.py

Score every signal on the whole order of finish, not just the winner. This is
like "strokes gained", counted in positions: HRN has no beaten margins (only
the order of finish), so this is positions gained vs the median horse.

  rho        per-race Spearman correlation between the signal and the actual
             finish order (1 = called the order exactly, 0 = no better than random)
  top PG     positions gained by the signal's top-rated horse:
             (field + 1) / 2 - finish, so a winner in a 9-horse field is +4
  top-3 PG   the same, averaged over the signal's top 3

Signals: each blind model's lengths-vs-median, Beyer and win-prob prompts; the
ML; and the saved blend (weights fit in-sample on winners only). Comparisons
use the races every listed signal covers. "vs ML" is a paired difference with a
95% bootstrap CI over races.

Usage:
    python rank_metrics.py
    python rank_metrics.py --spring-only
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import fit_openrouter_blend as F

PROMPTS = {'lengths': '_lengths', 'beyer': '_expected_beyer', 'winprob': '_win_prob'}


def per_race(H, signals):
    """One row per race x signal: rho, top-pick and top-3 positions gained."""
    rows = []
    for rid, g in H.groupby('race_id', sort=False):
        fin = g['finish'].values
        mid = (len(g) + 1) / 2
        for name, col in signals.items():
            x = g[col].values.astype(float)
            if np.isnan(x).any() or np.ptp(x) == 0:
                continue
            rho = spearmanr(x, -fin).correlation
            order = np.argsort(-x, kind='stable')
            rows.append({'race_id': rid, 'card': g['card'].iat[0], 'signal': name, 'rho': rho,
                         'top_pg': mid - fin[order[0]], 'top3_pg': mid - fin[order[:3]].mean(),
                         'field': len(g)})
    return pd.DataFrame(rows)


def boot_ci(x, n=4000, seed=1):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def report(R, title):
    common = R.groupby('race_id')['signal'].nunique()
    races = common[common == R['signal'].nunique()].index
    C = R[R['race_id'].isin(races)]
    ml = C[C['signal'] == 'ML'].set_index('race_id')['rho']
    print(f"\n{title}: {len(races)} races every signal covers")
    print(f"  {'signal':<22}{'rho':>7}{'vs ML':>8}  {'95% CI':<17}{'top PG':>7}{'top-3 PG':>9}")
    stats = []
    for s, g in C.groupby('signal'):
        d = g.set_index('race_id')['rho'] - ml
        stats.append((g['rho'].mean(), s, d, g))
    for rho, s, d, g in sorted(stats, key=lambda t: -t[0]):
        lo, hi = boot_ci(d.values) if s != 'ML' else (np.nan, np.nan)
        ci = f"[{lo:+.3f},{hi:+.3f}]" if s != 'ML' else ''
        print(f"  {s:<22}{rho:>7.3f}{d.mean():>+8.3f}  {ci:<17}{g['top_pg'].mean():>+7.2f}{g['top3_pg'].mean():>+9.2f}")


def main():
    parser = argparse.ArgumentParser(description='Order-of-finish metrics for every model signal')
    parser.add_argument('--spring-only', action='store_true')
    args = parser.parse_args()

    base = F.discover_providers()
    H = F.load_history(base, marker='ai_combined.csv' if args.spring_only else 'geminiflash_blind.csv')
    blend = json.loads((F.DATA_PATH / 'openrouter_blend.json').read_text())
    u = blend['ml_weight'] * np.log(H['ml'])
    for p, w in blend['weights'].items():
        u = u + w * np.log(H[p])
    H['blend'] = np.exp(u)   # rank within race is all that matters here

    signals = {'ML': 'ml', 'blend (in-sample)': 'blend'}
    for b in base:
        for prompt, suffix in PROMPTS.items():
            col = f'{b}{suffix}'
            cover = H.groupby('race_id')[col].apply(lambda s: s.notna().all()).mean()
            if cover >= 0.5:
                signals[f"{b.replace('_blind', '')} {prompt}"] = col
    R = per_race(H, signals)
    print(f"{H['race_id'].nunique()} races / {H['card'].nunique()} cards. rho = per-race Spearman vs finish order; "
          f"PG = positions gained vs the median horse.")
    print("Races per signal: " + ", ".join(f"{s} {n}" for s, n in R.groupby('signal').size().items()))

    lengths = [s for s in signals if s.endswith('lengths')]
    report(R[R['signal'].isin(['ML', 'blend (in-sample)'] + lengths)], 'Lengths prompts')
    report(R[R['signal'].isin(['ML'] + [s for s in signals if 'geminiflash' in s])], 'Gemini Flash, all prompts')
    report(R, 'Every signal')


if __name__ == '__main__':
    main()
