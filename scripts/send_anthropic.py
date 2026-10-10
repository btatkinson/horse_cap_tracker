"""
send_anthropic.py

Send race PDFs to Claude through the Anthropic API directly (your own key),
for Opus 5.5 at high effort. Same prompts, CSV format and raw jsonl as
send_openrouter.py, and the same default: ML-redacted (blind) forms. Claude
reads the PDF natively (page images + text), so no glyph legend is needed.

Modes:
    card    race day — one card through the Messages API (streaming, with
            the server-side refusal fallback enabled)
    batch   backfill — every historical card with results in ONE Message
            Batch (50% off; usually done within the hour, max 24h)

Usage:
    python send_anthropic.py card --date 2026-10-08 --track keeneland
    python send_anthropic.py batch                         # submit, wait, write CSVs
    python send_anthropic.py batch --resume msgbatch_...   # collect an already-submitted batch

Writes: {DATA_PATH}/ai_picks/{date}/{track}/{name}.csv   (default name opus_blind)
        {DATA_PATH}/ai_picks/_batches/{batch_id}.json    (batch index, for --resume)

Requires: .env with CLAUDE_API_KEY and DATA_PATH
"""

import argparse, asyncio, base64, json, os, random, time
from pathlib import Path

import anthropic

from send_openrouter import prepare_card, build_systems, write_outputs, parse_races, DATA_PATH

MODEL = 'claude-opus-5-5'
MAX_TOKENS = 32000
# Opus 5.5 defaults to medium effort; handicapping a field is worth more thought.
DEFAULT_EFFORT = 'high'
BATCH_DIR = Path(DATA_PATH) / 'ai_picks' / '_batches'


def user_content(race_num: int, pdf_bytes: bytes) -> list:
    return [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                        "data": base64.standard_b64encode(pdf_bytes).decode()}},
        {"type": "text", "text": f"Here is the DRF past performance PDF for Race {race_num}. "
                                 f"Provide your estimates for each horse."},
    ]


def message_text(msg) -> str:
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def usage_dict(u) -> dict:
    return {'input_tokens': u.input_tokens, 'output_tokens': u.output_tokens,
            'cache_read_input_tokens': getattr(u, 'cache_read_input_tokens', 0) or 0}


async def call_claude(client, semaphore, race_num, pdf_bytes, system, label, model, effort,
                      max_retries: int = 4):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with client.beta.messages.stream(
                    model=model, max_tokens=MAX_TOKENS, system=system,
                    messages=[{"role": "user", "content": user_content(race_num, pdf_bytes)}],
                    betas=["server-side-fallback-2026-07-01"],
                    extra_body={"fallbacks": "default", "output_config": {"effort": effort}},
                ) as stream:
                    msg = await stream.get_final_message()
                if msg.stop_reason == "refusal":
                    print(f"  R{race_num} ({label}) refused by the whole fallback chain")
                    return (race_num, label, "", {'stop_reason': 'refusal'})
                return (race_num, label, message_text(msg),
                        {'usage': usage_dict(msg.usage), 'stop_reason': msg.stop_reason,
                         'served_by': msg.model})
            except anthropic.RateLimitError as e:
                wait = float(e.response.headers.get("retry-after", "30")) + random.uniform(0, 5)
                print(f"  R{race_num} ({label}) rate limited; waiting {wait:.0f}s")
                await asyncio.sleep(wait)
            except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                wait = min(2 ** attempt * 5, 60) + random.uniform(0, 2)
                print(f"  R{race_num} ({label}) attempt {attempt+1} failed: {e}; retrying in {wait:.0f}s")
                await asyncio.sleep(wait)
        print(f"  R{race_num} ({label}) FAILED after {max_retries} attempts")
        return (race_num, label, "", {})


async def run_card(date, track, name, *, model=MODEL, effort=DEFAULT_EFFORT, forms='chopped_forms_blind',
                   prompts=('winprob',), concurrency=4, races_filter=None, forms_track=None,
                   deliberate=True):
    races, track_prompt = prepare_card(date, track, forms, forms_track, True, races_filter, True)
    if not races:
        return None
    systems = build_systems(prompts, b'', 'blind' in forms, deliberate, track_prompt)
    client = anthropic.AsyncAnthropic(api_key=os.environ['CLAUDE_API_KEY'], max_retries=0)
    sem = asyncio.Semaphore(concurrency)
    print(f"{model} effort={effort}: {len(races)} races × {'+'.join(prompts)} "
          f"(server-side refusal fallback on)")
    t0 = time.time()
    results = await asyncio.gather(*[call_claude(client, sem, rn, pdf, systems[label], label, model, effort)
                                     for rn, pdf in sorted(races.items()) for label in prompts])
    return write_outputs(date, track, name, model, results, elapsed=time.time() - t0)


def build_batch(cards, model, effort, prompts, forms='chopped_forms_blind', deliberate=True):
    """Return (requests, index) for every race on every card."""
    requests, index = [], {}
    for date, track, forms_track in cards:
        races, track_prompt = prepare_card(date, track, forms, forms_track, True, None, False)
        systems = build_systems(prompts, b'', 'blind' in forms, deliberate, track_prompt)
        for rn, pdf in sorted(races.items()):
            for label in prompts:
                cid = f"{date}_{track}_{rn}_{label}"
                index[cid] = [date, track, rn, label]
                requests.append({"custom_id": cid, "params": {
                    "model": model, "max_tokens": MAX_TOKENS, "system": systems[label],
                    "messages": [{"role": "user", "content": user_content(rn, pdf)}],
                    "output_config": {"effort": effort},
                }})
    return requests, index


def collect_batch(client, batch_id, index, name, model):
    """Wait for the batch to end, then write one CSV per card."""
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"  {time.strftime('%H:%M:%S')} {b.processing_status}: processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored}", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(60)

    by_card = {}
    for r in client.messages.batches.results(batch_id):
        date, track, rn, label = index[r.custom_id]
        if r.result.type == "succeeded":
            msg = r.result.message
            item = (rn, label, message_text(msg),
                    {'usage': usage_dict(msg.usage), 'stop_reason': msg.stop_reason})
        else:
            print(f"  {r.custom_id}: {r.result.type}")
            item = (rn, label, "", {'batch_result': r.result.type})
        by_card.setdefault((date, track), []).append(item)
    for (date, track), results in sorted(by_card.items()):
        s = write_outputs(date, track, name, model, results, verbose=False)
        print(f"  {date}/{track:<16} rows={s['rows']:>3} failed={s['failed']} "
              f"tokens {s['tok_in']:,} in / {s['tok_out']:,} out", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Send race PDFs to Claude (Opus 5.5) via the Anthropic API')
    sub = parser.add_subparsers(dest='mode', required=True)
    pc = sub.add_parser('card', help='Race day: one card, standard Messages API')
    pc.add_argument('--date', required=True)
    pc.add_argument('--track', required=True)
    pc.add_argument('--races', default=None)
    pb = sub.add_parser('batch', help='Backfill every historical card with results (Batch API, 50% off)')
    pb.add_argument('--cards', default=None, help='Comma list of DATE/TRACK (default: all with results)')
    pb.add_argument('--resume', default=None, help='Collect an already-submitted batch id')
    for p in (pc, pb):
        p.add_argument('--name', default='opus_blind')
        p.add_argument('--model', default=MODEL)
        p.add_argument('--effort', default=DEFAULT_EFFORT, choices=['low', 'medium', 'high', 'xhigh', 'max'])
        p.add_argument('--prompts', default='winprob')
    args = parser.parse_args()
    prompts = tuple(p.strip() for p in args.prompts.split(','))

    if args.mode == 'card':
        asyncio.run(run_card(args.date, args.track, args.name, model=args.model, effort=args.effort,
                             prompts=prompts, races_filter=parse_races(args.races) if args.races else None))
        return

    client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'])
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = json.loads((BATCH_DIR / f'{args.resume}.json').read_text())
        collect_batch(client, args.resume, state['index'], state['name'], state['model'])
        return

    from backfill_openrouter import results_cards, forms_track_for
    cards = ([tuple(c.split('/')) for c in args.cards.split(',')] if args.cards else results_cards())
    cards = [(d, t, forms_track_for(d, t, 'chopped_forms_blind')) for d, t in cards]
    cards = [c for c in cards if c[2]]
    requests, index = build_batch(cards, args.model, args.effort, prompts)
    batch = client.messages.batches.create(requests=requests)
    (BATCH_DIR / f'{batch.id}.json').write_text(json.dumps(
        {'name': args.name, 'model': args.model, 'effort': args.effort, 'index': index}))
    print(f"Submitted {batch.id}: {len(requests)} requests across {len(cards)} cards", flush=True)
    collect_batch(client, batch.id, index, args.name, args.model)


if __name__ == '__main__':
    main()
