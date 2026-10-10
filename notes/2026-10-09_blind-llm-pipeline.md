# Session notes — 2026-10-08 / 10-09: blind LLM pipeline, data fixes, frozen bet rule

## TL;DR

- **Moved off Opus/Gemini/GPT direct calls** to cheap models, mostly via OpenRouter.
  - The models never see the morning line: they read **ML-redacted PDFs**.
  - The market enters only in a fitted **stage 2** (conditional logit).
- **Two data bugs fixed:**
  - `download_results.py` was inventing starters: 390 of 2,094 spring "starters" were scratches or AEs.
  - HRN's footnote row was shifting finish positions.
- **The current probability model is Gemini 3.8 Flash Beyer + lengths + ML** (bagged, pruned).
  - Every model's "win probability" answer was cut. Asking for *performance* (Beyers, margins) works better.
  - Five other models were tested and cut: Opus 5.5, DeepSeek, GLM, GPT-6 Luna, Haiku 5.5.
- **Betting is pre-race only**: no closing odds (CAW) and no live trading.
  - "Bet anything that beats the ML" loses about as much as random.
  - The frozen rule is the **top pick, field-adjusted ML 5/2+, top 40% by edge**.
- **Error bars are huge:** about 200 races and about 40 rule bets. Treat everything as a hypothesis being tracked.

## Race-day workflow

Use `C:\Users\Blake\anaconda3\python.exe` with `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`. The default `python` lacks pymupdf. Run from `scripts/`:

```bash
python chop_form.py --pdf KEE--10-10-2026.pdf --date 2026-10-10 --track keeneland
python make_blind_forms.py --date 2026-10-10 --track keeneland
python download_upc.py --track keeneland --date 2026-10-10
python track_condition.py --track keeneland --date 2026-10-10
python backfill_openrouter.py --models geminiflash --variants blind --prompts winprob,beyer,lengths --cards 2026-10-10/keeneland
python download_scratches.py --date 2026-10-10 --apply --wait 150
python project_openrouter.py --date 2026-10-10 --track keeneland
```

**Reading the projector output:**
- It prints every race with the model's win %, fair odds and field-adjusted ML ("Adj").
- `<- BET` marks horses that meet the frozen rule, and a bet list prints at the end.
- **Re-pull scratches before each race:** `download_scratches.py --date ... --apply`. Late scratches happen after the list is posted.
- **Manual overrides:** `--scratch "5:3,7:2"` and `--draw-in "9:13"` (race:program).
- **Optional tracking runs:** the other models are useful for future refits but aren't used by the rule.
  - `--models deepseek,glm,haiku,luna`, about $1.50 per card.
  - Opus: `send_anthropic.py card`, about $1 per prompt per card.

## The frozen bet rule (as of 2026-10-09 ~11:45)

> $2 win on the blend's top pick when its **field-adjusted ML > 2-1** and **P × (adj ML + 1) ≥ 1.104**

- **Field-adjusted ML:** the horse's ML-implied chance rescaled from the field the ML was set for (main body, scratches included, AEs excluded) to the horses actually running.
  - The raw ML goes stale after scratches. In 5-horse fields, 7/2 shots looked like overlays the public would never offer.
- **Cutoff:** the top 40% of 89 historical qualifiers, stored in `data/openrouter_blend.json` → `bet_rule`.
- **Expect about 2 bets per card;** some days there will be none.

**Backtest** (leave-one-card-out: weights and cutoff from the other 21 cards, $2 win at HRN payoffs):

| Version | ML only | Gemini Beyer + lengths + ML |
|---|---|---|
| Raw ML, top 40% | −0.4% (41 bets) | +23.5% (46 bets) |
| **Field-adjusted ML, top 40%** | **−58% (29 bets)** | **+29.5% (38 bets, 15 won; 95% CI −16% to +79%)** |

- Under the raw-ML version, most bets came from scratch-gutted races: +7% with 2+ scratches vs +52% with 0–1.
- The rule was arrived at through several choices made on the same 210 races, so expect shrinkage. Don't tune it further on the spring sample. Re-evaluate after 10–20 new cards.

**Live record so far:**
- **10/08:** raw-ML rule with real fields: 3 bets, 1 winner (Facteur Cheval $9.18), +53%.
- **10/09:** field-adjusted rule: R2 #16 Amazing Max, R5 #8 Rogue Meteor, R7 #5 Tiger Man. Results not yet graded.

## Findings

### Baselines (spring 2026, 22 cards / 210 races, corrected data)

- **Random pick:** −25.7%.
- **Every ML favorite:** −25.7%, winning 32%.
- **By ML range:**
  - 2-1 or shorter: −41%.
  - 5/2–9/2: −25%.
  - 5-1..9-1: −6%, the best range this spring.
  - 10-1..19-1: −25%.
  - 20-1+: −44%.
- **Winners pay ~35–40% under their ML** in every range: live horses get bet down, and CAW is sharp.

### Probability models: held-out log-loss vs ML alone (lower is better)

| Signals (each + ML) | vs ML |
|---|---|
| Gemini Flash win-prob prompt | −0.024 |
| Gemini Flash lengths prompt | −0.031 |
| Gemini Flash Beyer prompt (t-dist sim) | −0.041 |
| **Gemini Flash Beyer + lengths (saved blend)** | **−0.042** |
| Opus 5.5 Beyer | −0.009 |
| DeepSeek lengths | −0.005 |
| Spring frontier trio, win-prob only | −0.012 |
| Spring frontier trio, Beyer only | −0.018 |

- **Saved blend:** ML^0.53 × GeminiBeyer^0.63 × GeminiLengths^0.35. Out-of-bag 1.7351 vs 1.7751 for ML only.
- **Bagged pruning cut every win-prob view first.**
- **Opus 5.5 at high effort** split the weight evenly with Gemini but was always cut, because it's redundant. It also thinks only ~600 tokens per race even when told to deliberate.
- **The earlier "Beyer adds nothing" conclusion was an artifact** of the phantom starters.
- **Blind vs sighted:** sighted projections correlated 0.93 with the ML (blind 0.67). Sighted models mostly echo the odds.
- **Contamination check:** the backfilled April races predate model release, so I tested for memorized results.
  - No model claimed to remember any result.
  - Forced to pick from names alone, Gemini hit 12.4% (random 10.9%, ML favorite 26%).

### ROI (pre-race, HRN payoffs)

- **"Bet any horse with P × (ML+1) − 1 ≥ 10%" (value rule):** −18% to −30% for every blend, no better than random. It loads up on 10-1+ shots.
- **Top pick only, when its ML is 5/2+:**
  - Spring trio + ML: +3% (103 bets).
  - Gemini blend: −8%.
  - ML favorite at 5/2+: −12%.
  - Bagged ROI pruning on this rule cut every model. 100 bets can't rank models.
- **Better probabilities ≠ better bets on this sample.** Pick models by log-loss and use ROI to check rules.

## Code changes

**New scripts:**
- `send_openrouter.py`: any OpenRouter model.
  - Text mode, with a DRF glyph legend, or `--pdf` native mode.
  - `--forms chopped_forms_blind` (default) and prompts `winprob, beyer, lengths, closing`.
  - Writes merge into the existing CSV, and raw responses append to `*_raw.jsonl`.
- `send_anthropic.py`: Opus 5.5 at high effort, `card` (race day) or `batch` (backfill at 50% off, `--resume`).
- `make_blind_forms.py`: true PDF redaction of the ML for the DRF, older DRF, Equibase and Brisnet layouts. Each race gets exactly one box per horse.
- `backfill_openrouter.py`: runs models over historical cards. Model configs, including GLM's settings, live here.
- `fit_openrouter_blend.py`: bagged non-negative conditional logit with backward-elimination pruning.
  - Signals: `{model}`, `{model}_bs` and `{model}_len`.
  - `--metric roi`, with `--rule value|toppick`, and `--save --keep 0.40` to freeze the bet rule.
- `project_openrouter.py`: blend + bet flags.
  - Options: `--scratch`, `--draw-in`, `--starters-from-results` (grading) and `--out`.
- `ticket_ab.py`: A/B part-wheel Pick-N ticket builder.
- `download_scratches.py`: Keeneland's official scratches & changes page → `upc_entries` (matches by name first).

**Fixed:**
- `download_results.py`:
  - A starter is now any horse named in the payout table or the also-rans (matched by name, with fuzzy matching).
  - It no longer invents last-place finishes.
  - It skips the "* Preliminary Speed Figures" footnote row.
  - It records `also_eligible`.
  - The 22 spring results files were regenerated; the originals are in `data/_backup_results_2026-10-08/`.
- `download_upc.py`: reads HRN's "Scratched?" column and marks AEs as not running.
- `.env`: `OPENROUTER_KEY` had a duplicate `OPENROUTER_KEY=` pasted inside the quotes.

## Data sources

| Need | Source | Notes |
|---|---|---|
| Entries / ML | HRN entries page | `download_upc.py`. Shows AEs, but often **no scratches** |
| Scratches / AE draw-ins | keeneland.com/racing/scratches-changes | `download_scratches.py`. Friday's list appeared ~11:05 am ET. The page can have typos (wrong program numbers), so it matches by name |
| Results / payoffs | HRN results page | `download_results.py` (fixed). Payoffs for the top 3 only |
| Track condition | NWS forecast fallback | Equibase is bot-walled, so this is an estimate |

**Ruled out:**
- Equibase: bot wall. Don't circumvent it.
- Racing TV API: 403 without the app's credentials. Don't lift tokens.
- DRF: 403.
- m.keeneland.com: SSL handshake failure.
- KHRC PDFs: stewards' reports, posted after racing.

## Pick 5 notes

- **Tickets are A/B part-wheels:** a main line plus "backup" lines that swap in B horses in one leg at a time.
- **On 10/08 the A/B structure hit ~15% more often than the best single ticket** for the same $150.
- **Two-ticket pairs are chosen together:** ticket 1 is the best in budget, and ticket 2 is the structure adding the most *new* coverage.
  - Early 10/09 pair: 28.6% for $159.
  - Late 10/09 pair: 13.4% for $160.
- **Capping each ticket at 3 lines** cost nothing on the late card and roughly halves entry time.
- **Hit rates are model probabilities, not a tested edge.** Horizontals haven't been backtested yet.
- **Not yet in the repo:** the pair optimizer is in this session's scratchpad (`p5_pair2.py`) and should be ported into `ticket_ab.py`.

## Costs & accounts (as of 10/09 ~1:45 pm)

- **OpenRouter:** $60.86 left of $100. Backfills cost about $16 for Beyers and $8 for lengths across 5 models × 22 cards.
- **Anthropic:**
  - The old key ran out of credit.
  - The new key ($20) has about $9.50 left after the Opus Beyer batch (~$9.40, which sat queued for 3 hours) and today's card (~$1).
- **Race-day cost for the rule:** about $0.15 per card (Gemini Flash, 3 prompts).

## Open items

1. **Grade every new card on the frozen rule** (`project_openrouter.py --starters-from-results`) and keep a running tally. Grade 10/09 tonight.
2. **Pick 4/5 backtest at HRN payoffs.** A proposal session card exists, and it needs 2026 exotic payoffs in `data/meta`.
3. **Port the pair optimizer and a `--slip` export into `ticket_ab.py`;** default to 3 lines per ticket.
4. **Refit the blend after 10–20 new cards,** including the tracking models, and decide then whether to refreeze the rule.
5. **Consider adding 5/01 and 5/02 Churchill** (Derby days) to the history. Results are on HRN, but there are no entries files yet.
6. **No programmatic betting:** retail ADWs generally lack a betting API. Ask the ADW about API or bulk upload access.

## Gotchas

- **GLM barely reasons by default** (12 tokens). Run it with `--reasoning-effort high --provider-order z-ai --deliberate`.
- **Gemini reads each PDF page as a compact image,** about 1,400 input tokens per race.
- **Anthropic batches can sit queued for hours** on a new account.
- **Bash heredocs mangle backslashes in Python regexes.** Use the Write tool for scripts containing regexes.
