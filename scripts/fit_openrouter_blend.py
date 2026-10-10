"""
fit_openrouter_blend.py

Stage 2 of the two-stage model: learn how much each odds-blind LLM adds to
the market (morning line), and drop the ones that don't pull their weight.

Model (Benter-style conditional logit), per race over starters:
    P(i wins) = softmax_i( b_ml * log p_ml,i + sum_m b_m * log p_m,i ),  b >= 0
p = each signal's win prob normalized over starters, floored at 0.003.

Bagging + pruning (default):
    1. Candidates: every *_blind provider found in ai_picks (from
       backfill_openrouter.py / send_anthropic.py batch), plus ML (always kept).
    2. For B bootstrap resamples of cards: fit the weights in-bag, score
       log-loss on the out-of-bag cards, and refit with each model left out.
       dOOB_m = OOB loss without m - OOB loss with everyone (> 0: m helps).
    3. Backward elimination: while some model's mean dOOB <= 0 or it helps
       in fewer than 60% of bags, drop the weakest and repeat.
    4. Final weights = mean of the in-bag fits for the survivors (bagged).

ROI mode (--metric roi): judges models by money instead of log-loss, under
the constraint of no closing odds. Rule: $2 win on every horse with
P x (ML+1) - 1 >= tau, decided on the HRN morning line and settled at HRN
win payoffs. Reports leave-one-card-out ROI per model set (CI over cards),
where the bets land by ML bucket, and bagged pruning on OOB profit per race
(betting nothing = $0). On 2026-10-08 (209 races) every set lost 33-67%;
winners paid ~35-40% under their ML, so ML-priced "overlays" mostly vanish.

Usage:
    python fit_openrouter_blend.py                # bag, prune, report
    python fit_openrouter_blend.py --save         # ... and write data/openrouter_blend.json
    python fit_openrouter_blend.py --candidates deepseek_blind,glm_blind,opus_blind --bags 300
    python fit_openrouter_blend.py --metric roi --tau 0.10

Writes: {DATA_PATH}/openrouter_blend.json (with --save)
"""

import os, re, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
FLOOR = 0.003
KEEP_FRACTION = 0.60   # a survivor must help out-of-bag in at least this share of bags

# Beyer prompt -> win probabilities: each horse's figure + scale * t(df) noise,
# winner = best figure. Scale/df were fit by the spring pipeline (backtest.py).
_FP = json.loads((DATA_PATH / 'fitted_params.json').read_text())
SIM_STD, SIM_DF, SIM_N = _FP['std_dev'], _FP['degrees_of_freedom'], 20000
_SIM_RNG = np.random.default_rng(0)


def beyer_sim(beyers) -> np.ndarray:
    b = np.asarray(beyers, float)
    draws = b[None, :] + SIM_STD * _SIM_RNG.standard_t(SIM_DF, size=(SIM_N, len(b)))
    return np.bincount(draws.argmax(1), minlength=len(b)) / SIM_N


# Lengths prompt (margin vs the median horse) -> a within-race softmax. Its log
# is linear in lengths, so the stage-2 weight is "log-odds per length" times
# LEN_SCALE; the scale only keeps far-back horses off the probability floor.
LEN_SCALE = 0.2


def lengths_softmax(lengths) -> np.ndarray:
    z = LEN_SCALE * np.asarray(lengths, float)
    e = np.exp(z - z.max())
    return e / e.sum()


def ml_to_prob(s):
    m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*[/-]\s*(\d+(?:\.\d+)?)\s*$', str(s))
    return np.nan if not m else 1.0 / (float(m.group(1)) / float(m.group(2)) + 1.0)


def norm(x):
    x = np.clip(np.asarray(x, float), FLOOR, None)
    return x / x.sum()


def discover_providers():
    return sorted({f.stem for f in (DATA_PATH / 'ai_picks').glob('2026-*/*/*_blind.csv')})


def load_history(providers, marker='ai_combined.csv'):
    """marker: a card is loaded when ai_picks/{date}/{track}/{marker} exists
    (default: the spring cards that went through combine_ai_picks.py)."""
    rows = []
    for comb in sorted((DATA_PATH / 'ai_picks').glob(f'2026-*/*/{marker}')):
        date, track = comb.parent.parent.name, comb.parent.name
        ef = DATA_PATH / 'entries' / f'{track}_{date}.csv'
        if not ef.exists():
            continue
        ent = pd.read_csv(ef)
        # ML-implied chance summed over the main body (scratched included,
        # AEs excluded): the field the morning line was set for.
        ent['q_ml'] = ent['ml_odds'].map(ml_to_prob)
        ae = ent['also_eligible'] if 'also_eligible' in ent else pd.Series(0, index=ent.index)
        main_sum = ent[ae == 0].groupby('race')['q_ml'].sum()
        ent = ent[ent['running'] == 1].copy()
        ent['main_q_sum'] = ent['race'].map(main_sum)
        ent['horse_number'] = pd.to_numeric(ent['horse'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        ent['finish'] = pd.to_numeric(ent['finish'], errors='coerce')
        df = ent[['race', 'horse_number', 'ml_odds', 'finish', 'win_paid', 'q_ml', 'main_q_sum']]
        for p in providers:
            f = comb.parent / f'{p}.csv'
            cols = [f'{p}_win_prob', f'{p}_expected_beyer', f'{p}_lengths']
            if f.exists():
                d = pd.read_csv(f)
                d['horse_number'] = pd.to_numeric(d['horse_number'].astype(str).str.extract(r'(\d+)')[0],
                                                  errors='coerce')
                for c in cols:
                    if c not in d:
                        d[c] = np.nan
                d = d.drop_duplicates(['race', 'horse_number'])[['race', 'horse_number'] + cols]
                df = df.merge(d, on=['race', 'horse_number'], how='left')
            else:
                for c in cols:
                    df[c] = np.nan
        df['date'], df['track'] = date, track
        rows.append(df)
    H = pd.concat(rows, ignore_index=True)

    out = []
    for (date, track, race), g in H.groupby(['date', 'track', 'race']):
        if len(g) < 3 or (g['finish'] == 1).sum() != 1:
            continue
        g = g.copy()
        g['ml'] = norm(g['ml_odds'].map(ml_to_prob).fillna(0.02))
        for p in providers:
            col = g[f'{p}_win_prob']
            # A provider that skipped a starter gets its field-min; one with
            # no picks for the race stays NaN (race excluded for that set).
            g[p] = norm(col.fillna(col.min())) if col.notna().any() else np.nan
            # Second view from the Beyer prompt, simulated over starters only.
            by = g[f'{p}_expected_beyer']
            g[f'{p}_bs'] = norm(beyer_sim(by.fillna(by.min()).values)) if by.notna().any() else np.nan
            # Third view from the lengths-vs-median prompt.
            ln = g[f'{p}_lengths']
            g[f'{p}_len'] = norm(lengths_softmax(ln.fillna(ln.min()).values)) if ln.notna().any() else np.nan
        g['race_id'] = f'{date}/{track}/{race}'
        g['card'] = f'{date}/{track}'
        g['win'] = (g['finish'] == 1).astype(int)
        # Decision price = HRN morning line; settlement = HRN $2 win payoff.
        g['ml_dec'] = 1.0 / g['ml_odds'].map(ml_to_prob).fillna(1 / 31) - 1.0
        # Field-adjusted ML: each starter's ML chance rescaled from the field
        # the ML was set for to the field that ran (scratches shorten the rest).
        q = g['q_ml'].fillna(1 / 31)
        g['ml_dec_adj'] = 1.0 / (q * g['main_q_sum'].fillna(q.sum()) / q.sum()) - 1.0
        paid = pd.to_numeric(g['win_paid'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
        g['paid'] = np.where(g['win'] == 1, paid.fillna(0.0), 0.0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


class Races:
    """Per-race arrays for fast refits on card subsets."""

    def __init__(self, data, feats):
        self.feats = feats
        self.cards = sorted(data['card'].unique())
        self.by_card, self.money = {}, {}
        for c, g in data.groupby('card'):
            rid = pd.factorize(g['race_id'])[0]
            self.by_card[c] = (np.log(g[feats].values), rid, g['win'].values, rid.max() + 1)
            self.money[c] = (g['ml_dec'].values, g['paid'].values)

    def stack(self, cards, cols):
        X, R, W, off = [], [], [], 0
        for c in cards:
            x, rid, w, nr = self.by_card[c]
            X.append(x[:, cols]); R.append(rid + off); W.append(w); off += nr
        return np.vstack(X), np.concatenate(R), np.concatenate(W), off

    def stack_money(self, cards):
        return (np.concatenate([self.money[c][0] for c in cards]),
                np.concatenate([self.money[c][1] for c in cards]))


RULE = {'name': 'value', 'min_ml': 2.0}   # set from --rule / --min-ml


def win_bets(b, X, rid, nr, ml_dec, paid, tau):
    """$2 win bets under the active RULE:
        value    every horse with P * (ML + 1) - 1 >= tau
        toppick  the model's top pick, only when its ML is longer than min_ml
                 (default 2.0, i.e. 5/2 or longer)

    Returns (profit, staked, n_bets, n_winning_bets) at HRN payoffs.
    """
    u = X @ b
    mx = np.full(nr, -np.inf); np.maximum.at(mx, rid, u)
    e = np.exp(u - mx[rid])
    s = np.zeros(nr); np.add.at(s, rid, e)
    p = e / s[rid]
    if RULE['name'] == 'toppick':
        best = np.full(nr, -1.0); np.maximum.at(best, rid, p)
        bet = (p >= best[rid]) & (ml_dec > RULE['min_ml'])
    else:
        bet = p * (ml_dec + 1.0) - 1.0 >= tau
    n = int(bet.sum())
    return float(paid[bet].sum() - 2.0 * n), 2.0 * n, n, int((paid[bet] > 0).sum())


def race_losses(b, X, rid, win, nr):
    u = X @ b
    mx = np.full(nr, -np.inf); np.maximum.at(mx, rid, u)
    e = np.exp(u - mx[rid])
    s = np.zeros(nr); np.add.at(s, rid, e)
    lp = u - mx[rid] - np.log(s[rid])
    per = np.zeros(nr); np.add.at(per, rid, -lp * win)
    return per


def _nll_and_grad(b, X, rid, win, nr):
    """Mean winner negative log-likelihood and its exact gradient,
    d/db = sum_i (P_i - win_i) x_i / n_races."""
    u = X @ b
    mx = np.full(nr, -np.inf); np.maximum.at(mx, rid, u)
    e = np.exp(u - mx[rid])
    s = np.zeros(nr); np.add.at(s, rid, e)
    lp = u - mx[rid] - np.log(s[rid])
    return -(lp * win).sum() / nr, ((np.exp(lp) - win)[:, None] * X).sum(0) / nr


def fit_nonneg(X, rid, win, nr):
    k = X.shape[1]
    return minimize(_nll_and_grad, np.full(k, 1.0 / k), args=(X, rid, win, nr), jac=True,
                    method='L-BFGS-B', bounds=[(0, None)] * k).x


def bag(races: Races, n_bags: int, seed: int = 7):
    """Bootstrap over cards. Returns (weights[B,k], full_oob[B], drop_oob[B,k])."""
    k = len(races.feats)
    rng = np.random.default_rng(seed)
    W, full, drop = [], [], []
    for _ in range(n_bags):
        inbag = list(rng.choice(races.cards, size=len(races.cards), replace=True))
        oob = [c for c in races.cards if c not in set(inbag)]
        if not oob:
            continue
        Xi, Ri, Wi, ni = races.stack(inbag, list(range(k)))
        Xo, Ro, Wo, no = races.stack(oob, list(range(k)))
        b = fit_nonneg(Xi, Ri, Wi, ni)
        W.append(b)
        full.append(race_losses(b, Xo, Ro, Wo, no).mean())
        d = []
        for j in range(k if k > 1 else 0):   # a lone signal has nothing to drop to
            cols = [c for c in range(k) if c != j]
            bj = fit_nonneg(Xi[:, cols], Ri, Wi, ni)
            d.append(race_losses(bj, Xo[:, cols], Ro, Wo, no).mean())
        drop.append(d)
    return np.array(W), np.array(full), np.array(drop).reshape(len(W), -1)


def loco_roi(races: Races, taus, n_boot: int = 2000, seed: int = 3):
    """Leave-one-card-out ROI: fit on the other cards, bet the held-out one.

    Returns {tau: (roi, ci_lo, ci_hi, n_bets, n_winners)}; the CI resamples cards.
    """
    cols = list(range(len(races.feats)))
    per = {t: [] for t in taus}
    for c in races.cards:
        Xi, Ri, Wi, ni = races.stack([x for x in races.cards if x != c], cols)
        b = fit_nonneg(Xi, Ri, Wi, ni)
        Xo, Ro, _, no = races.stack([c], cols)
        mld, paid = races.stack_money([c])
        for t in taus:
            per[t].append(win_bets(b, Xo, Ro, no, mld, paid, t))
    rng = np.random.default_rng(seed)
    out = {}
    for t, rows in per.items():
        a = np.array(rows)
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        st = a[idx, 1].sum(1)
        boot = a[idx, 0].sum(1)[st > 0] / st[st > 0]
        roi = a[:, 0].sum() / a[:, 1].sum() if a[:, 1].sum() else float('nan')
        lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (float('nan'),) * 2
        out[t] = (roi, lo, hi, int(a[:, 2].sum()), int(a[:, 3].sum()))
    return out


def bag_roi(races: Races, n_bags: int, tau: float, seed: int = 7):
    """Bootstrap over cards; OOB profit per race ($2 win bets) of the full set
    and of each drop-one set. Profit per race, not ROI, is compared: per-bag
    ROI explodes when a set makes only a few bets, and betting nothing is a
    legitimate $0 baseline.

    Returns (profit_full[B], profit_drop[B,k], pooled_full_roi).
    """
    k = len(races.feats)
    rng = np.random.default_rng(seed)
    full, drop, tot_p, tot_s = [], [], 0.0, 0.0
    for _ in range(n_bags):
        inbag = list(rng.choice(races.cards, size=len(races.cards), replace=True))
        oob = [c for c in races.cards if c not in set(inbag)]
        if not oob:
            continue
        Xi, Ri, Wi, ni = races.stack(inbag, list(range(k)))
        Xo, Ro, _, no = races.stack(oob, list(range(k)))
        mld, paid = races.stack_money(oob)
        p, s, _, _ = win_bets(fit_nonneg(Xi, Ri, Wi, ni), Xo, Ro, no, mld, paid, tau)
        full.append(p / no); tot_p += p; tot_s += s
        d = []
        for j in range(k if k > 1 else 0):
            cols = [c for c in range(k) if c != j]
            pj, _, _, _ = win_bets(fit_nonneg(Xi[:, cols], Ri, Wi, ni), Xo[:, cols], Ro, no, mld, paid, tau)
            d.append(pj / no)
        drop.append(d)
    return np.array(full), np.array(drop).reshape(len(full), -1), (tot_p / tot_s if tot_s else 0.0)


def loco_bet_log(races: Races, tau: float) -> pd.DataFrame:
    """Every leave-one-card-out bet with its ML odds, model prob and payoff."""
    cols = list(range(len(races.feats)))
    rows = []
    for c in races.cards:
        Xi, Ri, Wi, ni = races.stack([x for x in races.cards if x != c], cols)
        b = fit_nonneg(Xi, Ri, Wi, ni)
        Xo, Ro, _, no = races.stack([c], cols)
        mld, paid = races.stack_money([c])
        u = Xo @ b
        mx = np.full(no, -np.inf); np.maximum.at(mx, Ro, u)
        e = np.exp(u - mx[Ro]); s = np.zeros(no); np.add.at(s, Ro, e)
        p = e / s[Ro]
        top = np.zeros(len(p), bool)
        best = np.full(no, -1.0)
        for i in range(len(p)):
            if p[i] > best[Ro[i]]:
                best[Ro[i]] = p[i]
        top = p >= best[Ro]
        rows.append(pd.DataFrame({'card': c, 'p': p, 'ml_dec': mld, 'paid': paid, 'top': top,
                                  'value_bet': p * (mld + 1) - 1 >= tau}))
    return pd.concat(rows, ignore_index=True)


def roi_main(H, alive, args):
    RULE.update(name=args.rule, min_ml=args.min_ml)
    taus = (0.0, 0.10, 0.20) if args.rule == 'value' else (args.tau,)
    D = H.groupby('race_id').filter(lambda g: g[['ml'] + alive].notna().all().all())
    rule_txt = ("$2 win on every horse with P x (ML+1) - 1 >= tau" if args.rule == 'value' else
                f"$2 win on the model's top pick when its ML is longer than {args.min_ml:g}-1")
    print(f"\n{D['race_id'].nunique()} races / {D['card'].nunique()} cards. Rule: {rule_txt}; "
          f"settled at HRN win payoffs.")
    every = D['paid'].sum() / (2.0 * len(D)) - 1
    fav = D.loc[D.groupby('race_id')['ml'].idxmax()]
    print(f"Baselines: bet every horse {every:+.1%} ROI | bet every ML favorite "
          f"{fav['paid'].sum() / (2.0 * len(fav)) - 1:+.1%} ROI ({len(fav)} bets, {int(fav['win'].sum())} won)")

    sets = [['ml'], ['ml', 'geminiflash_blind'], ['ml', 'opus_blind'], ['ml', 'deepseek_blind'],
            ['ml', 'geminiflash_blind', 'opus_blind'], ['ml'] + alive, ['geminiflash_blind'], ['opus_blind']]
    sets = [s for s in sets if all(f in ['ml'] + alive for f in s)]
    print(f"\nLeave-one-card-out ROI (95% CI over cards):")
    labels = ([f"tau={t:.0%}" for t in taus] if args.rule == 'value' else [f"top pick, ML>{args.min_ml:g}-1"])
    print(f"  {'signals':<46}" + "".join(f"{lab:>28}" for lab in labels))
    for feats in sets:
        res = loco_roi(Races(D, feats), taus)
        cells = "".join(f"  {r:+6.1%} [{lo:+5.0%},{hi:+5.0%}] n={n:<3}" for r, lo, hi, n, w in res.values())
        print(f"  {'+'.join(f.replace('_blind', '') for f in feats):<46}{cells}")

    if args.rule == 'value':
        roi_buckets(D, alive, args)
    print(f"\nBagged pruning on OOB profit per race ({args.rule} rule, {args.bags} bags; ML always kept;"
          f" betting nothing = $0):")
    prune_roi(D, alive, args)


def roi_buckets(D, alive, args):
    """Where the value-rule money goes, by ML odds range (leave-one-card-out)."""
    print("\nWhere the money goes (leave-one-card-out, $2 win bets):")
    buckets = [(0, 2.01, '<= 2-1'), (2.01, 4.51, '5/2-9/2'), (4.51, 9.51, '5-1..9-1'), (9.51, 999, '10-1+')]
    for feats in (['ml', 'geminiflash_blind'], ['ml', 'opus_blind'], ['geminiflash_blind']):
        if not all(f in ['ml'] + alive for f in feats):
            continue
        L = loco_bet_log(Races(D, feats), args.tau)
        name = '+'.join(f.replace('_blind', '') for f in feats)
        for rule, mask in (('value bets', L['value_bet']), ('top pick', L['top'])):
            B = L[mask]
            parts = []
            for lo, hi, lab in buckets:
                x = B[(B['ml_dec'] >= lo) & (B['ml_dec'] < hi)]
                if len(x):
                    parts.append(f"{lab}: {len(x)} bets, {int((x['paid'] > 0).sum())} won, "
                                 f"{x['paid'].sum() / (2 * len(x)) - 1:+.0%}")
            tot = B['paid'].sum() / (2 * len(B)) - 1 if len(B) else float('nan')
            print(f"  {name:<18} {rule:<10} total {len(B):>3} bets {tot:+5.0%} | " + " | ".join(parts))


def prune_roi(D, alive, args):
    """Backward elimination on bagged out-of-bag profit per race."""
    while True:
        feats = ['ml'] + alive
        full, drop, pooled = bag_roi(Races(D, feats), args.bags, args.tau)
        dROI = full[:, None] - drop if drop.size else np.zeros((len(full), 0))
        print(f"  set {'+'.join(f.replace('_blind', '') for f in feats)}: pooled OOB ROI {pooled:+.1%}, "
              f"profit/race ${full.mean():+.2f}")
        stats = {f: (dROI[:, j + 1].mean(), (dROI[:, j + 1] > 0).mean()) for j, f in enumerate(alive)}
        for f, (m, ph) in stats.items():
            print(f"    {f:<20} adds ${m:+.2f}/race   better in {ph:4.0%} of bags")
        failing = [f for f in alive if stats[f][0] <= 0 or stats[f][1] < KEEP_FRACTION]
        if not failing:
            break
        worst = min(failing, key=lambda f: stats[f][0])
        print(f"    -> cutting {worst}")
        alive.remove(worst)
        if not alive:
            break
    print(f"  Kept on ROI: {alive or 'none (ML only)'}")


def bet_rule_cutoff(D, feats, weights, min_ml, keep):
    """Score cutoff for "bet the top pick when its field-adjusted ML > min_ml,
    but only the top `keep` share of those by P x (adjusted ML + 1)", from the
    saved weights on all historical cards. Frozen into the blend JSON.

    Field-adjusted, because the raw ML is stale after scratches: in gutted
    fields the survivors looked like overlays the public never offers (raw-ML
    version: ML-only rule ~0%, but -58% once repriced)."""
    R = Races(D.assign(ml_dec=D['ml_dec_adj']), feats)
    X, rid, _, nr = R.stack(R.cards, list(range(len(feats))))
    mld, _ = R.stack_money(R.cards)
    u = X @ np.array([weights[f] for f in feats])
    mx = np.full(nr, -np.inf); np.maximum.at(mx, rid, u)
    e = np.exp(u - mx[rid]); s = np.zeros(nr); np.add.at(s, rid, e)
    p = e / s[rid]
    best = np.full(nr, -1.0); np.maximum.at(best, rid, p)
    pick = (p >= best[rid]) & (mld > min_ml)
    score = p[pick] * (mld[pick] + 1)
    return float(np.quantile(score, 1 - keep)), int(pick.sum())


def main():
    parser = argparse.ArgumentParser(description='Bagged stage-2 fit with pruning of odds-blind LLMs')
    parser.add_argument('--candidates', default=None, help='Comma list (default: every *_blind provider)')
    parser.add_argument('--bags', type=int, default=200)
    parser.add_argument('--save', action='store_true', help='Write data/openrouter_blend.json')
    parser.add_argument('--metric', choices=['logloss', 'roi'], default='logloss',
                        help='roi: judge models by pre-race win-bet ROI at HRN payoffs instead')
    parser.add_argument('--tau', type=float, default=0.10,
                        help='ROI mode, value rule: bet when P x (ML+1) - 1 >= tau (edge at the morning line)')
    parser.add_argument('--rule', choices=['value', 'toppick'], default='value',
                        help='ROI mode: value (edge at the ML) or toppick (model top pick when ML > --min-ml)')
    parser.add_argument('--min-ml', type=float, default=2.0,
                        help='toppick rule: only bet when the top pick is longer than this (2.0 = 5/2+)')
    parser.add_argument('--keep', type=float, default=0.40,
                        help='--save: keep this share of qualifying top picks by P x (ML+1); the cutoff '
                             'is stored as bet_rule in the blend JSON')
    args = parser.parse_args()

    # Candidate signals per blind model: win-prob prompt ("{p}"), Beyer prompt
    # via the t-dist sim ("{p}_bs"), lengths-vs-median prompt ("{p}_len").
    if args.candidates:
        signals = [c.strip() for c in args.candidates.split(',')]
        base = sorted({re.sub(r'_(bs|len)$', '', c) for c in signals})
    else:
        base = discover_providers()
        signals = base + [f'{p}_bs' for p in base] + [f'{p}_len' for p in base]
    H = load_history(base)
    providers = signals
    coverage = {p: H.groupby('race_id')[p].apply(lambda s: s.notna().all()).mean() for p in providers}
    print("Coverage of races with results:", ", ".join(f"{p} {c:.0%}" for p, c in coverage.items()))
    alive = [p for p in providers if coverage[p] >= 0.8]
    for p in providers:
        if p not in alive:
            print(f"  {p}: skipped (backfill covers only {coverage[p]:.0%} of races)")
    if args.metric == 'roi':
        roi_main(H, alive, args)
        return

    rounds = 0
    while True:
        rounds += 1
        feats = ['ml'] + alive
        D = H.groupby('race_id').filter(lambda g: g[feats].notna().all().all())
        races = Races(D, feats)
        W, full, drop = bag(races, args.bags)
        dOOB = drop - full[:, None]           # > 0: removing the model hurts, i.e. it helps
        print(f"\nRound {rounds}: {D['race_id'].nunique()} races / {len(races.cards)} cards, "
              f"{len(W)} bags, OOB log-loss {full.mean():.4f}")
        print(f"  {'signal':<20} {'bagged w':>8} {'w>0.02':>7} {'mean dOOB':>10} {'helps in':>9}")
        stats = {}
        for j, f in enumerate(feats):
            stats[f] = (W[:, j].mean(), (W[:, j] > 0.02).mean(), dOOB[:, j].mean(), (dOOB[:, j] > 0).mean())
            w, pw, dm, ph = stats[f]
            print(f"  {f:<20} {w:>8.3f} {pw:>7.0%} {dm:>+10.4f} {ph:>9.0%}")
        failing = [f for f in alive if stats[f][2] <= 0 or stats[f][3] < KEEP_FRACTION]
        if not failing:
            break
        worst = min(failing, key=lambda f: stats[f][2])
        print(f"  -> cutting {worst} (mean dOOB {stats[worst][2]:+.4f}, helps in {stats[worst][3]:.0%} of bags)")
        alive.remove(worst)
        if not alive:
            break

    ml_only = Races(D, ['ml'])
    _, ml_full, _ = bag(ml_only, args.bags)
    print(f"\nKept: {alive or 'none'}. OOB log-loss {full.mean():.4f} vs ML-only {ml_full.mean():.4f} "
          f"({full.mean() - ml_full.mean():+.4f})")

    if args.save:
        feats = ['ml'] + alive
        weights = {f: float(stats[f][0]) for f in feats}
        out = {'variant': 'bagged_blind_ml', 'ml_weight': weights['ml'],
               'weights': {p: weights[p] for p in alive}, 'providers': alive,
               'oob_log_loss': float(full.mean()), 'oob_log_loss_ml_only': float(ml_full.mean()),
               'diagnostics': {f: dict(zip(['bagged_w', 'p_w_gt_0.02', 'mean_dOOB', 'p_helps'],
                                           map(float, stats[f]))) for f in feats},
               'n_races': int(D['race_id'].nunique()), 'n_cards': len(races.cards), 'bags': int(len(W))}
        cutoff, n_pool = bet_rule_cutoff(D, feats, weights, args.min_ml, args.keep)
        out['bet_rule'] = {'description': f"$2 win on the top pick when its field-adjusted ML > {args.min_ml:g}-1 "
                                          f"and P x (adj ML+1) >= score_cutoff (top {args.keep:.0%} historically)",
                           'price': 'field_adjusted_ml',
                           'min_ml': args.min_ml, 'keep': args.keep, 'score_cutoff': cutoff,
                           'n_historical_qualifiers': n_pool}
        print(f"Bet rule: top pick with ML > {args.min_ml:g}-1 and P x (ML+1) >= {cutoff:.3f} "
              f"(top {args.keep:.0%} of {n_pool} historical qualifiers)")
        (DATA_PATH / 'openrouter_blend.json').write_text(json.dumps(out, indent=2))
        print(f"Saved to {DATA_PATH / 'openrouter_blend.json'}")


if __name__ == '__main__':
    main()
