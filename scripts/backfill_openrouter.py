"""
backfill_openrouter.py

Run the OpenRouter models over every historical card that has results, so
the blend can be fit on their own track record (fit_openrouter_blend.py).

Variants:
    sighted   normal forms, all three prompts          -> {model}.csv
    blind     ML-redacted forms (make_blind_forms.py),  -> {model}_blind.csv
              win-prob prompt only

Model settings match race day: DeepSeek at medium reasoning; GLM at high
effort pinned to Z.AI with --deliberate (otherwise it barely reasons).

Usage:
    python backfill_openrouter.py                              # all results cards, both models, both variants
    python backfill_openrouter.py --variants blind --cards 2026-10-08/keeneland
    python backfill_openrouter.py --skip-existing              # resume after an interruption

Writes: {DATA_PATH}/ai_picks/{date}/{track}/{name}.csv (+ _raw.jsonl);
        never touches claude/gemini/openai/ai_combined files.
"""

import asyncio, argparse, json, time
from pathlib import Path

from send_openrouter import run_card, make_client, build_extra_body, DATA_PATH

DATA = Path(DATA_PATH)

MODELS = {
    'deepseek': dict(model='deepseek/deepseek-v4.1-flash', concurrency=40, deliberate=False,
                     extra=build_extra_body('medium', 'throughput')),
    'glm': dict(model='z-ai/glm-5.3-flash', concurrency=30, deliberate=True,
                extra=build_extra_body('high', provider_order=['z-ai'])),
    # Native-PDF models read the redacted PDF directly (no text extraction).
    'geminiflash': dict(model='google/gemini-3.8-flash', concurrency=30, deliberate=False, pdf=True,
                        extra=build_extra_body('high', 'throughput')),
    'haiku': dict(model='anthropic/claude-haiku-5.5', concurrency=30, deliberate=False, pdf=True,
                  extra=build_extra_body('high', 'throughput')),
    'luna': dict(model='openai/gpt-6-luna-pro', concurrency=30, deliberate=False, pdf=True,
                 extra=build_extra_body('high', 'throughput')),
}
# Opus 5.5 runs through the Anthropic Batch API instead: send_anthropic.py batch
VARIANTS = {
    'sighted': dict(forms='chopped_forms', prompts=('beyer', 'winprob', 'closing'), suffix=''),
    'blind': dict(forms='chopped_forms_blind', prompts=('winprob',), suffix='_blind'),
}
# Forms folder names that differ from the picks/results track name.
FORMS_ALIAS = {'gulfstream-park': ['gulfstream']}


def results_cards():
    """(date, track) for every card with AI picks AND a results file."""
    cards = []
    for comb in sorted((DATA / 'ai_picks').glob('2026-*/*/ai_combined.csv')):
        date, track = comb.parent.parent.name, comb.parent.name
        if (DATA / 'entries' / f'{track}_{date}.csv').exists():
            cards.append((date, track))
    return cards


def forms_track_for(date, track, forms):
    for t in [track] + FORMS_ALIAS.get(track, []):
        if (DATA / forms / date / t).exists():
            return t
    return None


async def main():
    parser = argparse.ArgumentParser(description='Backfill OpenRouter model picks on historical cards')
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--variants', default='blind',
                        help='blind (default; models never see the ML) and/or sighted')
    parser.add_argument('--cards', default=None,
                        help='Comma list of DATE/TRACK (default: every card with results)')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--prompts', default=None,
                        help="Override the variant's prompts, e.g. beyer (results merge into the existing CSV)")
    args = parser.parse_args()

    cards = ([tuple(c.split('/')) for c in args.cards.split(',')] if args.cards else results_cards())
    models = [m.strip() for m in args.models.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]

    client = make_client()
    sems = {m: asyncio.Semaphore(MODELS[m]['concurrency']) for m in models}
    jobs = []
    for date, track in cards:
        for v in variants:
            vcfg = VARIANTS[v]
            ft = forms_track_for(date, track, vcfg['forms'])
            if ft is None:
                print(f"  {date}/{track}: no {vcfg['forms']} folder — skipped")
                continue
            for m in models:
                cfg = MODELS[m]
                name = m + vcfg['suffix']
                if args.skip_existing and (DATA / 'ai_picks' / date / track / f'{name}.csv').exists():
                    continue
                prompts = tuple(args.prompts.split(',')) if args.prompts else vcfg['prompts']
                jobs.append(run_card(date, track, cfg['model'], name, client=client, semaphore=sems[m],
                                     forms=vcfg['forms'], prompts=prompts, extra_body=cfg['extra'],
                                     deliberate=cfg['deliberate'], forms_track=ft,
                                     pdf=cfg.get('pdf', False), verbose=False))

    print(f"{len(jobs)} card-runs across {len(cards)} cards ({'+'.join(models)} × {'+'.join(variants)})",
          flush=True)
    t0 = time.time()
    totals = {'tok_in': 0, 'tok_out': 0, 'failed': 0, 'done': 0}
    for fut in asyncio.as_completed(jobs):
        s = await fut
        totals['done'] += 1
        for k in ('tok_in', 'tok_out', 'failed'):
            totals[k] += s.get(k, 0)
        print(f"[{totals['done']:>3}/{len(jobs)} {time.time()-t0:6.0f}s] {s['card']:<28} {s.get('name',''):<15} "
              f"rows={s['rows']:>3} failed={s['failed']} "
              f"reasoning~{s.get('median_reasoning_tokens', 0):.0f} ({s.get('seconds', 0)}s)", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s. Tokens {totals['tok_in']:,} in / {totals['tok_out']:,} out; "
          f"{totals['failed']} failed calls", flush=True)


if __name__ == '__main__':
    asyncio.run(main())
