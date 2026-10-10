"""
diagnose_segments.py

Where does the blend beat, or lose to, the morning line? This script splits held-out
log-loss by race type:
  - 2-year-old races
  - races with first-time starters (FTS)
  - maidens
  - field size
  - how open the race looked (ML entropy)
  - form layout
  - season
For each segment it also reports the ML favorite's strike rate and the average
$2 payoff of the winner. It then checks calibration on FTS: expected wins
(sum of P) against actual wins, under the model and under the ML.

Model probabilities are leave-one-card-out: the weights are refit without the
card being scored. The feature set is the saved blend's (ML + Gemini Beyer sim +
Gemini lengths). It is compared against a leave-one-card-out ML-only fit
(softmax of b * log ML), so both get the same calibration treatment. Log-loss
grows with how open a race is, so read the delta (blend - ML), not the levels.

First-time starters come from each horse's "Life" record in the chopped forms.
The DRF, Equibase official and Brisnet layouts are all handled.

Usage:
    python diagnose_segments.py                # every card with Gemini picks (spring + fall)
    python diagnose_segments.py --spring-only  # just the cards the blend was fit on
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
import fitz

import fit_openrouter_blend as F

DATA_PATH = F.DATA_PATH
BLEND = json.loads((DATA_PATH / 'openrouter_blend.json').read_text())
FEATS = ['ml'] + BLEND['providers']


def name_key(name):
    n = re.sub(r'\(.*?\)', '', str(name).lower())
    return re.sub(r'[^a-z0-9]', '', n)


def form_layout(pdf):
    text = ''.join(p.get_text() for p in pdf)
    if 'Daily Racing Form' in text:
        return 'DRF'
    return 'official' if 'Owner:' in text else 'Brisnet'


def life_starts(pdf_path, names):
    """{name_key: lifetime starts} for the horses in one race's form.

    Each horse block has one "Life" row, whose first number is starts. It is
    tied to the horse whose header name (a large bold span) is nearest
    vertically on the same page. In DRF the header is on the same row; in the
    official layout it is about three rows below.
    """
    keys = {name_key(n) for n in names}
    out = {}
    pdf = fitz.open(str(pdf_path))
    for page in pdf:
        # Header names: a line of large text (names are split into word spans
        # in the official layout) whose key starts with an entry's key.
        heads = []
        for b in page.get_text('dict')['blocks']:
            for l in b.get('lines', []):
                big = [s for s in l['spans'] if s['size'] >= 10]   # Brisnet headers are 10pt
                k = name_key(''.join(s['text'] for s in big))
                if len(k) >= 3:
                    hit = max((x for x in keys if k.startswith(x)), key=len, default=None)
                    if hit:
                        heads.append((hit, (big[0]['bbox'][1] + big[0]['bbox'][3]) / 2))
        # "Life" rows: the first number to its right on the same baseline.
        words = page.get_text('words')
        lives = []
        for w in words:
            if w[4] not in ('Life', 'Life:'):
                continue
            yc = (w[1] + w[3]) / 2
            right = [v for v in words if abs((v[1] + v[3]) / 2 - yc) < 3 and 0 <= v[0] - w[2] < 30]
            if right:
                v = min(right, key=lambda v: v[0])
                if v[4].isdigit():
                    lives.append((yc, int(v[4])))
        for k, y in heads:
            if lives and k not in out:
                dy, starts = min((abs(ly - y), st) for ly, st in lives)
                if dy < 60:
                    out[k] = starts
    return out, form_layout(pdf)


def attach_race_info(H):
    """Adds horse_name, starts, fts and race-level attributes."""
    parts = []
    for (date, track), g in H.groupby(['date', 'track']):
        ent = pd.read_csv(DATA_PATH / 'entries' / f'{track}_{date}.csv')
        ent['horse_number'] = pd.to_numeric(ent['horse'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        ent = ent[ent['running'] == 1].drop_duplicates(['race', 'horse_number'])
        g = g.merge(ent[['race', 'horse_number', 'horse_name']], on=['race', 'horse_number'], how='left')
        meta = pd.read_csv(DATA_PATH / 'meta' / f'{track}_{date}.csv')
        g = g.merge(meta[['race', 'Restriction', 'Type', 'Surface']], on='race', how='left')
        g['starts'] = np.nan
        g['layout'] = None
        for race, r in g.groupby('race'):
            pdf = DATA_PATH / 'chopped_forms' / date / track / f'race_{race}.pdf'
            if not pdf.exists():
                continue
            starts, layout = life_starts(pdf, r['horse_name'].dropna())
            g.loc[r.index, 'starts'] = r['horse_name'].map(lambda n: starts.get(name_key(n), np.nan))
            g.loc[r.index, 'layout'] = layout
        parts.append(g)
    H = pd.concat(parts, ignore_index=True)
    H['fts'] = H['starts'] == 0
    H['two_yo'] = H['Restriction'].astype(str).str.contains('2 Year')
    H['maiden'] = H['Type'].astype(str).str.contains('Maiden')
    H['season'] = np.where(H['date'] < '2026-07-01', 'spring', 'fall')
    return H


def loco_probs(H, feats):
    """Leave-one-card-out win probabilities for every starter."""
    R = F.Races(H, feats)
    cols = list(range(len(feats)))
    out = pd.Series(np.nan, index=H.index)
    for c in R.cards:
        Xi, Ri, Wi, ni = R.stack([x for x in R.cards if x != c], cols)
        b = F.fit_nonneg(Xi, Ri, Wi, ni)
        g = H[H['card'] == c]
        rid = pd.factorize(g['race_id'])[0]
        u = np.log(g[feats].values) @ b
        e = np.exp(u - pd.Series(u).groupby(rid).transform('max').values)
        out.loc[g.index] = e / pd.Series(e).groupby(rid).transform('sum').values
    return out


def race_table(H):
    """One row per race: losses, payoff, entropy and segment flags."""
    rows = []
    for rid, g in H.groupby('race_id', sort=False):
        w = g[g['win'] == 1].iloc[0]
        ml = g['ml'].values
        top = g['p_blend'].idxmax()
        fav = g['ml'].idxmax()
        pick = g.loc[top]
        bet = (pick['ml_dec_adj'] > BLEND['bet_rule']['min_ml'] and
               pick['p_blend'] * (pick['ml_dec_adj'] + 1) >= BLEND['bet_rule']['score_cutoff'])
        rows.append({
            'race_id': rid, 'card': g['card'].iat[0], 'season': g['season'].iat[0],
            'layout': g['layout'].iat[0], 'two_yo': g['two_yo'].iat[0], 'maiden': g['maiden'].iat[0],
            'surface': g['Surface'].iat[0], 'field': len(g),
            'n_fts': int(g['fts'].sum()), 'starts_known': g['starts'].notna().mean(),
            'fts_ml_share': g.loc[g['fts'], 'ml'].sum(),
            'eff_field': float(np.exp(-(ml * np.log(ml)).sum())),
            'll_ml': -np.log(w['p_ml']), 'll_blend': -np.log(w['p_blend']), 'll_ml_raw': -np.log(w['ml']),
            'win_paid': w['paid'], 'winner_fts': bool(w['fts']),
            'fav_won': bool(g.loc[fav, 'win']), 'top_won': bool(pick['win']),
            'bet': bet, 'bet_paid': pick['paid'] if bet else 0.0,
        })
    return pd.DataFrame(rows)


def ci(x, n_boot=4000, seed=1):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def segment_report(T):
    entropy_cut = T['eff_field'].quantile([1 / 3, 2 / 3]).values
    segs = [
        ('All races', T.index == T.index),
        ('2yo races', T['two_yo']),
        ('3yo+ races', ~T['two_yo']),
        ('Maiden races', T['maiden']),
        ('  maiden, no FTS', T['maiden'] & (T['n_fts'] == 0)),
        ('  maiden, 1-2 FTS', T['maiden'] & T['n_fts'].between(1, 2)),
        ('  maiden, 3+ FTS', T['maiden'] & (T['n_fts'] >= 3)),
        ('Non-maiden races', ~T['maiden']),
        ('No FTS in field', T['n_fts'] == 0),
        ('Any FTS in field', T['n_fts'] > 0),
        ('FTS hold 25%+ of ML', T['fts_ml_share'] >= 0.25),
        ('Dirt', T['surface'] == 'Dirt'),
        ('Turf', T['surface'] == 'Turf'),
        ('Field <= 7', T['field'] <= 7),
        ('Field 8-10', T['field'].between(8, 10)),
        ('Field 11+', T['field'] >= 11),
        (f'ML eff. field < {entropy_cut[0]:.1f} (chalky)', T['eff_field'] < entropy_cut[0]),
        ('ML eff. field middle third', T['eff_field'].between(*entropy_cut)),
        (f'ML eff. field > {entropy_cut[1]:.1f} (open)', T['eff_field'] > entropy_cut[1]),
        ('Layout: official', T['layout'] == 'official'),
        ('Layout: DRF', T['layout'] == 'DRF'),
        ('Layout: Brisnet', T['layout'] == 'Brisnet'),
        ('Spring cards', T['season'] == 'spring'),
        ('Fall cards (live)', T['season'] == 'fall'),
    ]
    print(f"\n{'segment':<34}{'races':>6}{'fav%':>6}{'top%':>6}{'avg $':>7}{'med $':>7}{'effN':>6}"
          f"{'LL ML':>7}{'LL us':>7}{'us-ML':>8}  {'95% CI':<16}{'rule bets':>10}{'ROI':>7}")
    for name, mask in segs:
        S = T[mask]
        if len(S) < 5:
            continue
        d = S['ll_blend'] - S['ll_ml']
        lo, hi = ci(d)
        nb = int(S['bet'].sum())
        roi = f"{S['bet_paid'].sum() / (2 * nb) - 1:+.0%}" if nb else ''
        print(f"{name:<34}{len(S):>6}{S['fav_won'].mean():>6.0%}{S['top_won'].mean():>6.0%}"
              f"{S['win_paid'].mean():>7.2f}{S['win_paid'].median():>7.2f}{S['eff_field'].mean():>6.1f}"
              f"{S['ll_ml'].mean():>7.3f}{S['ll_blend'].mean():>7.3f}{d.mean():>+8.3f}  "
              f"[{lo:+.3f},{hi:+.3f}]{nb:>8}{roi:>9}")


def fts_calibration(H):
    """Expected vs actual wins for FTS and raced horses, in races with any FTS."""
    M = H[H.groupby('race_id')['fts'].transform('any')]
    print(f"\nCalibration in the {M['race_id'].nunique()} races with at least one FTS "
          f"(expected wins = sum of win probabilities):")
    print(f"  {'group':<14}{'horses':>7}{'won':>5}{'E[ML]':>7}{'E[us]':>7}{'E[Gem B]':>9}{'E[Gem L]':>9}"
          f"{'avg ML%':>8}{'avg us%':>8}")
    for lab, g in (('FTS', M[M['fts']]), ('raced before', M[~M['fts'] & M['starts'].notna()])):
        print(f"  {lab:<14}{len(g):>7}{int(g['win'].sum()):>5}{g['p_ml'].sum():>7.1f}{g['p_blend'].sum():>7.1f}"
              f"{g[FEATS[1]].sum():>9.1f}{g[FEATS[2]].sum():>9.1f}"
              f"{g['ml'].mean():>8.1%}{g['p_blend'].mean():>8.1%}")
    fts = M[M['fts']]
    if len(fts):
        print(f"  FTS winners paid avg ${fts.loc[fts['win'] == 1, 'paid'].mean():.2f} "
              f"(all winners in these races ${M.loc[M['win'] == 1, 'paid'].mean():.2f})")


def main():
    parser = argparse.ArgumentParser(description='Held-out blend vs ML log-loss by race segment')
    parser.add_argument('--spring-only', action='store_true', help='Only the cards the blend was fit on')
    parser.add_argument('--out', default=None, help='Optional CSV of the per-race table')
    args = parser.parse_args()

    base = sorted({re.sub(r'_(bs|len)$', '', p) for p in BLEND['providers']})
    marker = 'ai_combined.csv' if args.spring_only else f'{base[0]}.csv'
    H = F.load_history(base, marker=marker)
    H = H.groupby('race_id').filter(lambda g: g[FEATS].notna().all().all()).reset_index(drop=True)
    H = attach_race_info(H)
    H['p_ml'] = loco_probs(H, ['ml'])
    H['p_blend'] = loco_probs(H, FEATS)
    T = race_table(H)

    known = H['starts'].notna().mean()
    print(f"{T['race_id'].nunique()} races / {T['card'].nunique()} cards. Blend = {' x '.join(FEATS)}, "
          f"refit leave-one-card-out; 'ML' = leave-one-card-out ML-only fit.")
    print(f"Lifetime starts parsed for {known:.0%} of starters "
          f"({', '.join(f'{k} {v:.0%}' for k, v in H.groupby('layout')['starts'].apply(lambda s: s.notna().mean()).items())}).")
    print("fav% = ML favorite won, top% = our top pick won, avg/med $ = winner's $2 payoff, "
          "effN = exp(ML entropy), LL = mean winner log-loss (lower is better), us-ML < 0 = we beat the ML. "
          "Rule bets use the frozen cutoff on held-out probabilities.")
    segment_report(T)
    fts_calibration(H)
    if args.out:
        T.to_csv(args.out, index=False)
        print(f"\nPer-race table: {args.out}")


if __name__ == '__main__':
    main()
