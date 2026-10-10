"""
project_openrouter.py

Turn per-provider LLM picks (deepseek.csv, glm.csv, ... from send_openrouter.py)
into blended win projections for one card.

Blend (log-linear / Benter-style):
    P  ∝  ML_norm ** ml_weight  ×  Π_m  p_m ** w_m
where p_m is provider m's win_prob normalized over the horses still running.

Two stages. Stage 1 is the LLMs reading ML-redacted forms (the *_blind
providers), so they handicap from the record instead of echoing the odds
(sighted picks correlated 0.93 with the ML on 2026-10-08, blind 0.67).
Stage 2 adds the market back with a fitted weight: providers and ml_weight
come from data/openrouter_blend.json (fit_openrouter_blend.py --save).
Without that file: blind providers, equal weights summing to 1, no ML.

Usage:
    python project_openrouter.py --date 2026-10-08 --track keeneland
    python project_openrouter.py --date 2026-10-08 --track keeneland --scratch "5:3,8:7"
    python project_openrouter.py --date 2026-10-08 --track keeneland --providers deepseek,glm --ml-weight 0  # sighted

Reads:   {DATA_PATH}/ai_picks/{date}/{track}/{provider}.csv
         {DATA_PATH}/upc_entries/{track}.csv   (re-run download_upc.py for late scratches)
         {DATA_PATH}/openrouter_blend.json     (optional fitted weights)
Writes:  {DATA_PATH}/predictions/{date}/{track}/projections_openrouter.csv
"""

import os, re, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

from fit_openrouter_blend import beyer_sim, lengths_softmax  # same transforms the blend was fit with
from diagnose_segments import life_starts, name_key

DATA_PATH = Path(os.environ['DATA_PATH'])
BLEND_PATH = DATA_PATH / 'openrouter_blend.json'
FLOOR = 0.003


def ml_to_prob(s):
    m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*[/-]\s*(\d+(?:\.\d+)?)\s*$', str(s))
    if not m:
        return np.nan
    return 1.0 / (float(m.group(1)) / float(m.group(2)) + 1.0)


def normalize(x):
    x = np.clip(np.asarray(x, float), FLOOR, None)
    return x / x.sum()


def fmt_odds(p):
    """Probability -> fair odds string ('3.5-1', '12-1')."""
    o = 1.0 / p - 1.0
    return f"{o:.1f}-1" if o < 10 else f"{o:.0f}-1"


def short(p):
    """Column label: 'geminiflash_blind_bs' -> 'gem-B', '..._len' -> 'gem-L'."""
    name = re.sub(r'_blind', '', re.sub(r'_(bs|len)$', '', p))
    name = {'geminiflash': 'gem', 'deepseek': 'ds', 'haiku': 'hk', 'luna': 'lu', 'opus': 'op'}.get(name, name[:4])
    return name + ('-B' if p.endswith('_bs') else '-L' if p.endswith('_len') else '')


def parse_races(spec):
    out = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def load_weights(providers, ml_weight_arg, weights_arg):
    """Resolve (ml_weight, {provider: weight}) from CLI > blend JSON > defaults."""
    fitted = json.loads(BLEND_PATH.read_text()) if BLEND_PATH.exists() else {}
    ml_weight = ml_weight_arg if ml_weight_arg is not None else fitted.get('ml_weight', 0.0)
    if weights_arg:
        weights = {k: float(v) for k, v in (kv.split('=') for kv in weights_arg.split(','))}
    elif fitted.get('weights') and all(p in fitted['weights'] for p in providers):
        weights = {p: fitted['weights'][p] for p in providers}
    else:
        weights = {p: 1.0 / len(providers) for p in providers}
    return ml_weight, weights, ('fitted' if fitted and not weights_arg else 'default/CLI')


def main():
    parser = argparse.ArgumentParser(description='Blend OpenRouter LLM picks into win projections')
    parser.add_argument('--date', required=True)
    parser.add_argument('--track', required=True)
    parser.add_argument('--providers', default=None,
                        help='Comma list; default from blend JSON, else deepseek_blind,glm_blind')
    parser.add_argument('--races', default=None, help='e.g. "1-3" or "2,5"')
    parser.add_argument('--ml-weight', type=float, default=None,
                        help='Weight on log(ML prob); default from blend JSON, else 0 (ML out)')
    parser.add_argument('--weights', default=None,
                        help='Provider weights, e.g. "deepseek=0.6,glm=0.4" (overrides blend JSON)')
    parser.add_argument('--scratch', default=None,
                        help='Extra scratches as race:horse pairs, e.g. "5:3,8:7"')
    parser.add_argument('--draw-in', default=None,
                        help='Also-eligibles that drew into the field, as race:horse pairs, e.g. "10:15"')
    parser.add_argument('--out', default='projections_openrouter.csv',
                        help='Output filename under predictions/{date}/{track}/')
    parser.add_argument('--starters-from-results', action='store_true',
                        help='Grading: take the field from entries/{track}_{date}.csv (actual starters, '
                             'including AEs that drew in) instead of the pre-race HRN entries')
    parser.add_argument('--fts-cutoff', type=float, default=0.30,
                        help='Rule bets in races where first-time starters are at least this share of the '
                             'runners are tracked, not played (default 0.30; above 1 disables)')
    args = parser.parse_args()

    if args.providers:
        providers = [p.strip() for p in args.providers.split(',') if p.strip()]
    elif BLEND_PATH.exists() and json.loads(BLEND_PATH.read_text()).get('providers'):
        providers = json.loads(BLEND_PATH.read_text())['providers']
    else:
        providers = ['deepseek_blind', 'glm_blind']
    ml_weight, weights, source = load_weights(providers, args.ml_weight, args.weights)
    print(f"Blend ({source}): ML^{ml_weight:g} × " +
          " × ".join(f"{p}^{w:.3g}" for p, w in weights.items()))
    pick_dir = DATA_PATH / 'ai_picks' / args.date / args.track

    ent = pd.read_csv(DATA_PATH / 'upc_entries' / f'{args.track}.csv')
    ent = ent[ent['date'] == args.date].copy()
    if ent.empty and args.starters_from_results:
        # download_upc.py rewrites upc_entries each day; the results file has
        # the same ML, AE and trainer/jockey columns for grading old cards.
        ent = pd.read_csv(DATA_PATH / 'entries' / f'{args.track}_{args.date}.csv')
    if ent.empty:
        raise SystemExit(f"upc_entries/{args.track}.csv has no rows for {args.date} — run download_upc.py")
    ent['horse_number'] = pd.to_numeric(ent['horse'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    if args.starters_from_results:
        rf = DATA_PATH / 'entries' / f'{args.track}_{args.date}.csv'
        res_e = pd.read_csv(rf)
        res_e['horse_number'] = pd.to_numeric(res_e['horse'].astype(str).str.extract(r'(\d+)')[0],
                                              errors='coerce')
        ran = res_e.set_index(['race', 'horse_number'])['running']
        ent['running'] = [int(ran.get((r, h), 0)) for r, h in zip(ent['race'], ent['horse_number'])]
        print(f"Field from results: {int(ent['running'].sum())} starters")
    for flag, value, running in (('--scratch', args.scratch, 0), ('--draw-in', args.draw_in, 1)):
        for pair in (value.split(',') if value else []):
            r, h = (int(x) for x in pair.split(':'))
            hit = (ent['race'] == r) & (ent['horse_number'] == h)
            if not hit.any():
                print(f"  {flag} {r}:{h} matches no entry")
            ent.loc[hit, 'running'] = running

    # A provider named "{p}_bs" is model p's Beyer prompt turned into win
    # probabilities by the t-dist sim; "{p}_len" is its lengths-vs-median
    # prompt as a within-race softmax. Both read p's CSV.
    base_of = {p: re.sub(r'_(bs|len)$', '', p) for p in providers}
    df = ent
    for b in dict.fromkeys(base_of.values()):
        pdf = pd.read_csv(pick_dir / f'{b}.csv')
        pdf['horse_number'] = pd.to_numeric(pdf['horse_number'].astype(str).str.extract(r'(\d+)')[0],
                                            errors='coerce')
        # Coupled entries (1/1A) share a betting interest — keep the first row.
        pdf = pdf.drop_duplicates(['race', 'horse_number'])
        keep = ['race', 'horse_number'] + [c for c in pdf.columns if c.startswith(b + '_')]
        df = df.merge(pdf[keep], on=['race', 'horse_number'], how='left')

    if args.races:
        df = df[df['race'].isin(parse_races(args.races))]

    out = []
    for rn, g in df.groupby('race'):
        # ML chance summed over the field the morning line was set for (main
        # body, scratches included, AEs excluded), for field-adjusted pricing.
        q_all = g['ml_odds'].map(ml_to_prob)
        main = (g['also_eligible'] == 0) if 'also_eligible' in g else pd.Series(True, index=g.index)
        main_q = q_all[main].sum()
        g = g[g['running'] == 1].copy()
        # Lifetime starts from the form's "Life" row. The blend has no edge
        # over the ML in FTS-heavy races (diagnose_segments.py), so rule bets
        # there are tracked, not played.
        form = DATA_PATH / 'chopped_forms' / args.date / args.track / f'race_{rn}.pdf'
        starts = life_starts(form, g['horse_name'].dropna())[0] if form.exists() else {}
        g['starts'] = g['horse_name'].map(lambda n: starts.get(name_key(n), np.nan))
        g['fts_share'] = (g['starts'] == 0).mean()
        for p in providers:
            col = (f'{base_of[p]}_expected_beyer' if p.endswith('_bs') else
                   f'{base_of[p]}_lengths' if p.endswith('_len') else f'{p}_win_prob')
            if g[col].isna().any():
                # A provider skipped a running horse: give it that provider's
                # field-min so the horse isn't silently dropped.
                print(f"  R{rn}: {p} missing #{g.loc[g[col].isna(), 'horse_number'].tolist()} — using field min")
                g[col] = g[col].fillna(g[col].min())

        g['ml_prob'] = normalize(g['ml_odds'].map(ml_to_prob).fillna(0.02))
        u = ml_weight * np.log(g['ml_prob'].values)
        logs = []
        for p in providers:
            if p.endswith('_bs'):
                g[f'{p}_p'] = normalize(beyer_sim(g[f'{base_of[p]}_expected_beyer'].values))
            elif p.endswith('_len'):
                g[f'{p}_p'] = normalize(lengths_softmax(g[f'{base_of[p]}_lengths'].values))
            else:
                g[f'{p}_p'] = normalize(g[f'{p}_win_prob'])
            logs.append(np.log(g[f'{p}_p'].values))
            u = u + weights[p] * logs[-1]
        g['llm_prob'] = normalize(np.exp(np.mean(logs, axis=0)))
        e = np.exp(u - u.max())
        g['proj_prob'] = e / e.sum()

        bases = list(dict.fromkeys(base_of.values()))
        beyer_cols = [f'{b}_expected_beyer' for b in bases if f'{b}_expected_beyer' in g]
        g['proj_beyer'] = g[beyer_cols].mean(axis=1).round(0)
        close_cols = [f'{b}_closing_prob' for b in bases if f'{b}_closing_prob' in g]
        g['pred_close_prob'] = normalize(g[close_cols].mean(axis=1).fillna(FLOOR)) if close_cols else np.nan

        g['ml_decimal'] = 1.0 / g['ml_odds'].map(ml_to_prob) - 1.0
        # Field-adjusted ML: rescaled to the horses actually running. The raw
        # ML is stale after scratches (5-horse fields made 7/2 shots "overlays").
        q = g['ml_odds'].map(ml_to_prob).fillna(1 / 31)
        g['ml_adj_decimal'] = 1.0 / (q * main_q / q.sum()) - 1.0
        # $1 EV if the horse goes off at its morning line
        g['ev_at_ml'] = g['proj_prob'] * (g['ml_decimal'] + 1.0) - 1.0
        g['fair_odds'] = g['proj_prob'].map(fmt_odds)
        g['rank'] = g['proj_prob'].rank(ascending=False, method='first').astype(int)
        out.append(g)

    res = pd.concat(out).sort_values(['race', 'rank'])

    # Pre-race bet rule frozen in the blend JSON (fit_openrouter_blend --save --keep):
    # top pick, ML longer than min_ml, and P x (ML+1) at or above the historical cutoff.
    rule = (json.loads(BLEND_PATH.read_text()).get('bet_rule') if BLEND_PATH.exists() else None)
    adjusted = bool(rule) and rule.get('price') == 'field_adjusted_ml'
    price = res['ml_adj_decimal'] if adjusted else res['ml_decimal']
    res['score'] = res['proj_prob'] * (price + 1.0)
    res['bet'] = False
    if rule and not args.weights and args.ml_weight is None:
        res['bet'] = ((res['rank'] == 1) & (price > rule['min_ml'])
                      & (res['score'] >= rule['score_cutoff']))
    # 'bet' is the frozen rule (graded on every race); 'play' is what gets wagered.
    res['play'] = res['bet'] & (res['fts_share'] < args.fts_cutoff)

    out_dir = DATA_PATH / 'predictions' / args.date / args.track
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ['race', 'rank', 'horse_number', 'horse_name', 'ml_odds', 'ml_prob',
            *[f'{p}_p' for p in providers], 'llm_prob', 'proj_prob', 'fair_odds',
            'ev_at_ml', 'ml_adj_decimal', 'score', 'bet', 'play', 'starts', 'fts_share',
            'proj_beyer', 'pred_close_prob', 'trainer', 'jockey']
    res[cols].to_csv(out_dir / args.out, index=False)

    for rn, g in res.groupby('race'):
        n_fts = int((g['starts'] == 0).sum())
        fts_note = f", {n_fts} first-time starter{'s' if n_fts != 1 else ''}" if n_fts else ''
        if g['fts_share'].iat[0] >= args.fts_cutoff:
            fts_note += f" -- {g['fts_share'].iat[0]:.0%} first-time starters: track only"
        if g['starts'].isna().any():
            fts_note += f" (starts unknown for #{g.loc[g['starts'].isna(), 'horse_number'].astype(int).tolist()})"
        print(f"\nRace {rn}  ({len(g)} running{fts_note})")
        print(f"  {'#':>3} {'Horse':<20} {'Sts':>3} {'ML':>5} {'Adj':>5} {'ML%':>5} " +
              ' '.join(f'{short(p):>6}' for p in providers) +
              f" {'Proj%':>6} {'Fair':>7} {'EV@ML':>6} {'Beyer':>5}")
        for _, r in g.iterrows():
            flag = '  <- BET' if r['play'] else '  <- rule bet, FTS-heavy race: TRACK ONLY' if r['bet'] else ''
            sts = '?' if pd.isna(r['starts']) else 'FTS' if r['starts'] == 0 else f"{int(r['starts'])}"
            print(f"  {int(r['horse_number']):>3} {str(r['horse_name'])[:20]:<20} {sts:>3} {r['ml_odds']:>5} "
                  f"{r['ml_adj_decimal']:>4.1f} "
                  f"{r['ml_prob']:>5.0%} " +
                  ' '.join(f"{r[f'{p}_p']:>6.0%}" for p in providers) +
                  f" {r['proj_prob']:>6.1%} {r['fair_odds']:>7} {r['ev_at_ml']:>+6.0%} "
                  f"{r['proj_beyer']:>5.0f}{flag}")
    print(f"\nSaved {len(res)} rows to {out_dir / args.out}")
    if rule:
        bets = res[res['bet']]
        print(f"\nBet rule: {rule['description']} (cutoff {rule['score_cutoff']:.3f})")
        print(f"  Played only when first-time starters are under {args.fts_cutoff:.0%} of the field; "
              f"the rest are tracked.")
        if bets.empty:
            print("  No qualifying bets.")
        for _, r in bets.iterrows():
            action = '$2 win' if r['play'] else f"TRACK ONLY ({r['fts_share']:.0%} first-time starters)"
            print(f"  R{int(r['race'])} #{int(r['horse_number'])} {r['horse_name']} (ML {r['ml_odds']}, "
                  f"field-adjusted {r['ml_adj_decimal']:.1f}-1): "
                  f"model {r['proj_prob']:.1%}, score {r['score']:.2f} -> {action}")


if __name__ == '__main__':
    main()
