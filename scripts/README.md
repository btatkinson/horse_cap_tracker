# Horse Racing AI Handicapping Pipeline

Ensemble AI + human expert handicapping system for thoroughbred racing.
Built around Claude and Gemini Beyer speed figure estimates, t-distribution
Monte Carlo probability simulation, expert logit adjustments, and Elo-rated
human handicapper integration.

## Quick Start — Race Day Workflow

```bash
# 1. AI pipeline: chop PDF → track conditions → Claude + Gemini
#    (confirms PDF, date, and track before spending API credits)
python run_ai_pipeline.py --track keeneland

# 2. Expert pipeline: scrape → LLM extraction → predictions
python run_expert_pipeline.py --track keeneland

# 3. Pick ticket optimizer
python pick_optimizer.py --date 2026-04-09 --track keeneland --races 1-5
python pick_optimizer.py --date 2026-04-09 --track keeneland --races 6-8

# --- After races are final ---

# 4. Grade the day
python daily_summary.py --date 2026-04-09 --track keeneland --bankroll 500

# 5. Update expert Elo ratings (replays all history)
python expert_elo.py --force

# 6. Re-run backtest with new data
python backtest_ai.py --optimize
```

Or run individual steps manually:
```bash
python chop_form.py --pdf GP--04-09-2026.pdf
python track_condition.py --track keeneland --date 2026-04-09
python send_claude.py --date 2026-04-09 --track keeneland
python send_gemini.py --date 2026-04-09 --track keeneland
python scrape_experts.py --track keeneland --date 2026-04-09
python send_experts.py --date 2026-04-09 --track keeneland
python predict_races.py --date 2026-04-09 --track keeneland
```

---

## Scripts Reference

### Data Collection

#### `download_upc.py`
Download upcoming entries with ML odds and scratches from Horse Racing Nation.

```
--track TRACK     Track name (e.g. keeneland)       [required]
--date DATE       Date YYYY-MM-DD                   [default: today]
```
Writes: `{DATA_PATH}/upc_entries/{track}.csv`, `{DATA_PATH}/upc_meta/{track}.csv`

#### `chop_form.py`
Split a DRF past-performance PDF into per-race PDFs (preserves PDF encoding).

```
--pdf FILENAME    PDF filename in data/forms/       [default: latest]
--date DATE       Override date folder (YYYY-MM-DD) [default: auto-detect]
--track TRACK     Override track name               [default: auto-detect]
```
Writes: `{DATA_PATH}/chopped_forms/{date}/{track}/race_N.pdf`

#### `track_condition.py`
Fetch track conditions from Equibase (race day) or NWS forecast (future dates).

```
--track TRACK     Track name                        [required]
--date DATE       Target date YYYY-MM-DD            [default: today]
```
Writes: `{DATA_PATH}/historical_meta.pkl`

### AI Estimates

#### `send_claude.py`
Send chopped race PDFs to Claude for Beyer + win probability estimates.
Each race gets two prompts (beyer + winprob). Injects track conditions.

```
--date DATE       Date folder (YYYY-MM-DD)          [required]
--track TRACK     Track folder                      [required]
--model MODEL     Claude model                      [default: claude-opus-4-6]
--yes, -y         Skip confirmation prompts
```
Writes: `{DATA_PATH}/ai_picks/{date}/{track}/claude.csv`

#### `send_gemini.py`
Same as send_claude but for Gemini.

```
--date DATE       Date folder (YYYY-MM-DD)          [required]
--track TRACK     Track folder                      [required]
--model MODEL     Gemini model                      [default: gemini-3.1-pro-preview]
--yes, -y         Skip confirmation prompts
```
Writes: `{DATA_PATH}/ai_picks/{date}/{track}/gemini.csv`

### Expert Picks

#### `scrape_experts.py`
Fetch raw expert pick text from track websites. Auto-discovers new experts
from the Keeneland index page.

```
--track TRACK     Track name                        [required]
--date DATE       Target date YYYY-MM-DD            [default: today]
```
Writes: `{DATA_PATH}/raw_expert_picks/{date}/{track}/{expert_name}.txt`

Skips experts already in `raw_expert_picks/` or `already_sent_eps/`.

#### `send_experts.py`
Send raw expert text to Claude for logit delta extraction. Includes a horse
name→number reference table from entries so the LLM can resolve names.

```
--date DATE       Date (YYYY-MM-DD)                 [required]
--track TRACK     Track name                        [required]
--model MODEL     Claude model for extraction       [default: claude-sonnet-4-20250514]
```
Reads: `{DATA_PATH}/raw_expert_picks/{date}/{track}/*.txt`
Writes: `{DATA_PATH}/expert_picks/{date}/{track}/{expert_name}.csv`

After successful extraction, moves the raw .txt to `already_sent_eps/` to
prevent double-charging on re-runs.

**Expert CSV format:**
```
race,horse_number,logit_delta,sentiment,expert
1,6,0.7,best_bet,scott_hazelton
1,2,0.3,mentioned,scott_hazelton
```

**Logit delta scale:**
| Delta | Meaning |
|-------|---------|
| +0.7  | Best bet / top pick |
| +0.5  | Strong 2nd pick |
| +0.3  | Mentioned favorably / "Use" |
| +0.15 | Contender / "Consider" / "Backup" |
| -0.3  | Mentioned negatively |
| -0.5  | Explicit avoid |

#### `expert_elo.py`
Elo rating system for expert handicappers. Full replay from initial ratings
every run — safe to re-run after tuning parameters.

```
--force           Replay all history from scratch, overwrite ratings
--report          Print current ratings table
--weights         Print derived ensemble weights
--quiet           Suppress per-day output
```
Reads: `{DATA_PATH}/expert_picks/`, `{DATA_PATH}/entries/`
Writes: `{DATA_PATH}/expert_ratings.json`

**Elo parameters** (top of file):
- `K_INITIAL = 16` — starting K factor
- `K_FLOOR = 8` — minimum K after decay
- `K_DECAY_RATE = 0.01` — K decay per race scored
- `ELO_SCALE = 3000` — expected improvement scaling
- `EXPERT_INITIAL_RATINGS` — per-expert starting ratings

**How it works:**
- Scores each expert per race: did their logit adjustments assign more
  probability to the actual winner vs raw ML?
- Update: `new_rating = old + K * (actual_improvement - expected_improvement)`
- Expected improvement scales with rating distance from 1500
- Negative ratings (below 1500) mean the expert's picks are *faded*

#### `view_expert_history.py`
Detailed per-race history viewer for one expert. Shows the logit math:
ML%, adjusted%, finish, winner identification.

```
EXPERT            Expert name (e.g. kyle_herron)    [positional]
--date DATE       Filter to one date
--list            List all tracked experts
```

### Predictions & Betting

#### `predict_races.py`
Generate race predictions and bet recommendations from AI picks + fitted
params + expert adjustments.

```
--date DATE       Date (YYYY-MM-DD)                 [required]
--track TRACK     Track name                        [required]
--sims N          MC simulations                    [default: 100000]
--min-roi PCT     Min EV ROI to bet                 [default: 0.0]
--takeout PCT     Win pool takeout rate              [default: 0.17]
```
Reads: `ai_picks/`, `upc_entries/`, `fitted_params.json`, `expert_ratings.json`, `expert_picks/`
Writes: `{DATA_PATH}/predictions/{date}/{track}/predictions.csv`, `bet_log.csv`

**Probability engine:**
1. Per AI source: t-dist MC sim from Beyers → `sim_prob`
2. Regularize: `reg_sim = 0.75 * sim + 0.25 * ML`
3. Blend: `source_prob = 0.5 * reg_sim + 0.5 * direct_win_prob`
4. Ensemble: logit-space weighted average of AI sources (ML not in ensemble)
5. Expert adjustments: apply Elo-weighted logit deltas
6. Edge: `EV_ROI = (ensemble_prob / ml_prob) * (1 - takeout) - 1`

**Betting rules:**
- Bet any horse with positive EV after takeout
- ML ≤ 9/2: Win only
- ML 5/1–9/1: Win + Place
- ML ≥ 10/1: Win + Place + Show
- 2+ positive EV horses in same race: Exacta box
- 3+ positive EV horses: Trifecta box

#### `pick_optimizer.py`
Pareto frontier ticket optimizer for Pick 3/4/5/6 multi-race bets.
Reads from predict_races output — no re-computation.

```
--date DATE       Date (YYYY-MM-DD)                 [required]
--track TRACK     Track name                        [required]
--races RANGE     Race range: "1-5" or "3,4,5,6,7"  [required]
--pool DOLLARS    Estimated pool size ($)            [default: 100000]
--takeout PCT     Track takeout rate                 [default: 0.22]
--max-cost N      Max ticket cost ($)                [default: none]
--blend PCT       Model vs ML blend (1.0=all model)  [default: 0.5]
--min-prob PCT    Min coverage per race               [default: none]
--min-total PCT   Min total hit probability           [default: none]
--top-n N         Number of tickets to show           [default: 10]
```

**EV formula:** `EV = (1 - takeout) × ∏(Σ our_i/mkt_i per race) - cost`

### Backtesting & Grading

#### `backtest_ai.py`
Backtest AI handicapping estimates against actual results. Optimizes
t-dist sim params, ML regularization weight, and source blend weights.

```
--optimize        Run Nelder-Mead optimization
--std FLOAT       Initial std_dev                    [default: 5.0]
--df FLOAT        Initial degrees_of_freedom         [default: 1.0]
--ml-reg FLOAT    Initial ML regularization          [default: 0.25]
--sims N          MC simulations                     [default: 50000]
```
Reads: `ai_picks/`, `entries/`
Writes: `{DATA_PATH}/fitted_params.json`, `ai_picks/backtest_results.csv`

**Optimized parameters:**
- `std_dev` — t-distribution scale for Beyer noise
- `degrees_of_freedom` — t-distribution tail heaviness
- `ml_regularization` — how much ML anchors the Beyer sim (0–1)
- Source weights (claude vs gemini split)

#### `daily_summary.py`
Grade a day's predictions and bets against actual results. Shows model
accuracy, race-by-race results, bet P&L, and expert pick grading.

```
--date DATE       Date (YYYY-MM-DD)                 [required]
--track TRACK     Track name                        [required]
--bankroll FLOAT  Starting bankroll for P&L tracking [default: none]
```
Writes: `{DATA_PATH}/daily_summaries/{date}_{track}.txt`

Expert grading scales bet size by confidence: best bets get $2 base,
contenders get $0.43. W/WP/WPS rules follow ML odds thresholds.

---

## Data Layout

All paths relative to `DATA_PATH` from `.env`:

```
{DATA_PATH}/
├── forms/                          # Raw DRF PP PDFs
├── chopped_forms/{date}/{track}/   # Per-race PDFs
│   └── race_N.pdf
├── ai_picks/{date}/{track}/        # AI estimates
│   ├── claude.csv
│   └── gemini.csv
├── upc_entries/{track}.csv         # Today's entries + ML odds
├── upc_meta/{track}.csv            # Race metadata
├── entries/                        # Historical results (ground truth)
│   └── {track}_{date}.csv
├── raw_expert_picks/{date}/{track}/ # Scraped expert text
│   └── {expert}.txt
├── already_sent_eps/{date}/{track}/ # Processed expert text (moved here)
│   └── {expert}.txt
├── expert_picks/{date}/{track}/    # LLM-extracted expert logit deltas
│   └── {expert}.csv
├── predictions/{date}/{track}/     # Forward predictions
│   ├── predictions.csv
│   └── bet_log.csv
├── daily_summaries/                # Daily grade reports
│   └── {date}_{track}.txt
├── historical_meta.pkl             # Track conditions cache
├── fitted_params.json              # Optimized model parameters
└── expert_ratings.json             # Expert Elo ratings + history
```

## Environment

`.env` file in repo root:
```
DATA_PATH=C:\Users\Blake\Code\Sports\horse_cap_tracker\data
CLAUDE_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

## Key Dates

All dates throughout the pipeline use **YYYY-MM-DD** format.

## Dependencies

```
pip install pymupdf anthropic google-genai python-dotenv
pip install requests beautifulsoup4 tqdm
pip install numpy pandas scipy scikit-learn
```