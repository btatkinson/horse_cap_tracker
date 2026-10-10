"""
predict_races.py

Generate race predictions and bet recommendations from AI picks + fitted params.

Usage:
    python predict_races.py --date 2026-04-08 --track keeneland
    python predict_races.py --date 2026-04-08 --track keeneland --sims 200000

Reads from:
    {DATA_PATH}/ai_picks/{date}/{track}/claude.csv   (and gemini.csv, openai.csv)
    {DATA_PATH}/upc_entries/{track}.csv               (scratches + ML odds)
    {DATA_PATH}/fitted_params.json                    (from backtest)
    {DATA_PATH}/regression_model.json                 (logistic CV model)
    {DATA_PATH}/ridge_value_model.json                (log-ridge, 4·W+2·P+1·S)
    {DATA_PATH}/raw_value_model.json                  (raw-ridge, W+P+S)

Writes to:
    {DATA_PATH}/predictions/{date}/{track}/predictions.csv
    {DATA_PATH}/predictions/{date}/{track}/bet_log.csv

Three models, each with its own ATB-derived break-even:
    1. Logistic → EV on a $2 win bet (ev_per_2), break-even = $0, bet ≥ $0.10
    2. Log-ridge (4·W+2·P+1·S, normalized to $18.30/race):
         break-even per horse = (18.30/N) × 1/(1-takeout)
    3. Raw-ridge (W+P+S, normalized to $6·(1-takeout)/race):
         break-even per horse = (6·(1-takeout)/N) × 1/(1-takeout) = 6/N

A horse is FLAGGED (shown in report) if ANY single model has it above its
break-even threshold × edge multiplier — the model sees value.

A horse is a BET if at least one of the three models clears its edge
threshold AND no model is below its qualification floor. The triggered_by
column logs which model(s) fired so you can see in the history which
signal is actually making money: LOG (logistic), RDG (log-ridge),
RAW (raw-ridge), or combinations.

A bet is marked as "AGREE" if at least 2 of 3 models are above their
break-even (positive value, even if not a full bet trigger). This is the
high-confidence tier — multi-model agreement.
"""

import os, json, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])


# ── Defaults (overridden by fitted_params.json if present) ──────────
DEFAULT_STD = 5.0
DEFAULT_DF = 1.0
DEFAULT_WEIGHTS = {'claude': 0.35, 'gemini': 0.30, 'ml': 0.35}

# Minimum edge (probability units) to consider a bet
MIN_EDGE = 0.065

# Win pool takeout rate (Keeneland ~17%, varies by track)
WIN_TAKEOUT = 0.17

# ML odds thresholds for bet type
#   <= 4.5 (9/2):  Win only
#   5 to 9:        Win + Place
#   >= 10:         Win + Place + Show
ML_WIN_ONLY_MAX = 4.5
ML_WP_MAX = 9.0

# ── Log-Ridge value target (4·W + 2·P + 1·S) ────────────────────────
# Target = 4*win_paid + 2*place_paid + 1*show_paid (raw dollars).
#
# Expected sum of target across a whole race (constant, independent of
# field size): ~$18.30. Derivation — on a $2-base ticket in a pari-mutuel
# pool with 17% takeout, the weighted sum of payouts across all three
# board-hitters averages:
#   4 * win_paid   -> 6.64  (one winner, 83% of $8 fair return)
#   2 * place_paid -> 6.64  (one 2nd, same math via place pool)
#   1 * show_paid  -> 4.98  (one 3rd, same math via show pool)
#   total          -> 18.26
#
# We normalize every race's ridge predictions to sum to this constant,
# so break-even per horse = RIDGE_RACE_TOTAL / N, and edge threshold
# scales with field size automatically (8-horse field: $2.29 break-even,
# 12-horse field: $1.52 break-even).
RIDGE_RACE_TOTAL = 18.3
# Floor = 0.70× BE means a horse needs to be meaningfully below field-average
# for this model to veto a bet another model is making. If floor = 1.0× BE,
# the average horse would block itself; tri-model logic would be strictly
# worse than logistic-alone for any horse below the field-average ridge.
RIDGE_MIN_MULTIPLIER = 0.70
RIDGE_EDGE_MULTIPLIER = 1.05  # 5% over per-horse fair share (BET trigger)

# ── Raw-Ridge value target (W + P + S, unweighted) ──────────────────
# Target = win_paid + place_paid + show_paid (raw dollars, no log).
#
# This is the "value hunter" model. Unlike log-ridge, we do NOT
# normalize per-race to a theoretical total. The raw-ridge output is
# heavily regularized and tends to cluster around the training-set
# mean (intercept dominates). Normalizing flattens the already-small
# spread and makes the signal unusable.
#
# Instead, predict_v2 reads the training-set prediction mean/std from
# raw_value_model.json (cv_pred_mean, cv_pred_std) and defines:
#   break-even threshold = cv_pred_mean
#   bet edge threshold   = cv_pred_mean + RAW_EDGE_K × cv_pred_std
#
# A horse whose raw prediction is above the training mean has some
# positive value; one that's K std above is a bet. This is field-size
# independent — the model's calibration handles it. If the model
# weakens and predictions narrow, K*std shrinks automatically.
#
# RAW_EDGE_K = 1.0 → ~16% of horses on average will clear bet (tail).
# RAW_EDGE_K = 1.5 → stricter, ~7%.
# RAW_EDGE_K = 0.5 → looser, ~31%.
RAW_EDGE_K = 1.0   # std-devs above CV prediction mean for bet trigger
RAW_MIN_K  = -0.5  # below which raw-ridge vetoes another model's bet
                   # (0.5σ below training mean = "actively bad horse")

# Small positive floor when normalizing a race (avoids div-by-zero and
# stops a single negative prediction from inverting everything).
# Only used by log-ridge (which still normalizes). Raw-ridge uses raw
# model outputs directly, so it has no floor.
RIDGE_FLOOR = 0.5

# Qualification floor (each horse must clear its active floor to be considered).
# Edge threshold (at least one model must clear its edge threshold to fire).
# Logistic defaults: non-negative EV floor, $0.05 edge.
# Ridge thresholds are computed per-race from RIDGE_RACE_TOTAL / N.
# Raw thresholds are computed from calibration stats in raw_value_model.json.
LOGISTIC_MIN = 0.00           # $ per $2 (ev_per_2 floor)
LOGISTIC_EDGE_THRESHOLD = 0.05  # $ per $2


# =============================================================================
# 1. LOADING
# =============================================================================

def load_fitted_params():
    """Load fitted_params.json from DATA_PATH. Returns dict or None."""
    fp = DATA_PATH / 'fitted_params.json'
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return None


def load_ai_picks(date_str, track):
    """Load the combined AI picks CSV for one date/track.

    Reads `ai_combined.csv` produced by `combine_ai_picks.py`, which
    merges claude/gemini/openai CSVs and runs IterativeImputer to fill
    per-provider NaN gaps. Horses that couldn't be recovered (no signal
    from any provider) have been dropped at combine time.

    Why read pre-combined rather than merge-at-runtime: the combined file
    is auditable — you can inspect ai_combined.csv to see which values
    were imputed, and the drop decisions are visible in the combine log.
    Runtime merging obscured both.
    """
    picks_dir = DATA_PATH / 'ai_picks' / date_str / track
    combined_path = picks_dir / 'ai_combined.csv'

    if not combined_path.exists():
        # Fall back: try flat layout (no track subdir)
        alt = DATA_PATH / 'ai_picks' / date_str / 'ai_combined.csv'
        if alt.exists():
            combined_path = alt
        else:
            raise FileNotFoundError(
                f"{combined_path} not found. Run `combine_ai_picks.py` to "
                f"generate the combined file from per-provider CSVs:\n"
                f"    python combine_ai_picks.py --date {date_str} --track {track}"
            )

    df = pd.read_csv(combined_path)

    # Normalize types — combine_ai_picks writes Int64 race/horse_number
    # but the CSV round-trip may lose this
    df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
    df['horse_number'] = pd.to_numeric(df['horse_number'],
                                        errors='coerce').astype('Int64')

    if 'date' not in df.columns:
        df['date'] = date_str
    if 'track' not in df.columns:
        df['track'] = track

    return df


def load_upc_entries(track):
    """Load upcoming entries with ML odds and running status."""
    upc_dir = DATA_PATH / 'upc_entries'
    # Try exact name, then with hyphens/underscores
    candidates = [
        upc_dir / f'{track}.csv',
        upc_dir / f'{track.replace("-", "_")}.csv',
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Normalize column names
            if 'horse' in df.columns and 'horse_number' not in df.columns:
                df['horse_number'] = pd.to_numeric(df['horse'], errors='coerce').astype('Int64')
            if 'race' in df.columns:
                df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
            return df
    return None


def _parse_ml_odds(odds_str):
    """Parse fractional/decimal ML odds to implied probability."""
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '':
            return np.nan
        s = str(odds_str).strip()
        if '/' in s:
            num, den = map(float, s.split('/'))
            return den / (num + den)
        val = float(s)
        if val >= 1:
            return 1 / (val + 1)
        return val
    except (ValueError, ZeroDivisionError):
        return np.nan


def _ml_odds_to_decimal(odds_str):
    """Parse fractional ML odds to decimal odds (e.g. '9/2' → 4.5)."""
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '':
            return np.nan
        s = str(odds_str).strip()
        if '/' in s:
            num, den = map(float, s.split('/'))
            return num / den
        return float(s)
    except (ValueError, ZeroDivisionError):
        return np.nan


# =============================================================================
# 2. PROBABILITY ENGINE (mirrors backtest_ai.py logic)
# =============================================================================

def sim_beyer_to_prob(df, beyer_col, std_dev, degrees_of_freedom, num_sims, seed=42):
    """T-dist MC sim per race. Returns Series of win probs."""
    result = pd.Series(np.nan, index=df.index)
    rng = np.random.RandomState(seed)
    t_dist = stats.t(df=degrees_of_freedom)

    for _, race_df in df.groupby('race'):
        idx = race_df.index
        n = len(idx)
        vals = race_df[beyer_col].values.copy()
        if n <= 1:
            result.loc[idx] = 1.0
            continue
        mean_b = np.nanmean(vals)
        if np.isnan(mean_b):
            continue
        vals = np.where(np.isnan(vals), mean_b, vals)
        noise = t_dist.rvs(size=(n, num_sims), random_state=rng) * std_dev
        winners = np.argmax(vals.reshape(n, 1) + noise, axis=0)
        counts = np.bincount(winners, minlength=n)
        probs = np.maximum(counts / num_sims, 1e-6)
        result.loc[idx] = probs / probs.sum()

    return result


def build_ensemble(df, std_dev, degrees_of_freedom, num_sims, weights,
                   ml_regularization=0.25):
    """Build 4 independent probability sources and blend into ensemble_prob.

    Sources:
      - claude_beyer: t-dist MC sim, ML-regularized
      - claude_prob:  direct LLM win probability
      - gemini_beyer: t-dist MC sim, ML-regularized
      - gemini_prob:  direct LLM win probability

    Each gets its own weight from fitted_params.json.
    Fallbacks: beyer backs up prob if missing, and vice versa.
    ML only regularizes the beyer sims.
    """
    AI_PROVIDERS = ['claude', 'gemini']

    df = df.copy()
    available = []

    has_ml = 'ml_win_prob_norm' in df.columns and df['ml_win_prob_norm'].notna().any()

    for provider in AI_PROVIDERS:
        bc = f'{provider}_beyer'
        pc = f'{provider}_win_prob'
        sim_col = f'{provider}_sim_prob'
        reg_col = f'{provider}_reg_sim'
        beyer_source = f'{provider}_beyer'
        prob_source = f'{provider}_prob'

        # Beyer path: sim → regularize
        if bc in df.columns and df[bc].notna().any():
            df[sim_col] = sim_beyer_to_prob(df, bc, std_dev, degrees_of_freedom, num_sims)
        else:
            df[sim_col] = np.nan

        if has_ml and df[sim_col].notna().any():
            ml_fill = df['ml_win_prob_norm'].fillna(df[sim_col])
            df[reg_col] = (1 - ml_regularization) * df[sim_col] + ml_regularization * ml_fill
        else:
            df[reg_col] = df[sim_col]

        # Beyer source (fallback to direct prob if beyer missing)
        beyer_col = f'{beyer_source}_src'
        df[beyer_col] = df[reg_col]
        beyer_missing = df[beyer_col].isna()
        if pc in df.columns and beyer_missing.any():
            df.loc[beyer_missing, beyer_col] = df.loc[beyer_missing, pc]
        if df[beyer_col].notna().any():
            available.append(beyer_source)

        # Prob source (fallback to reg sim if direct prob missing)
        prob_col = f'{prob_source}_src'
        if pc in df.columns:
            df[prob_col] = df[pc]
        else:
            df[prob_col] = np.nan
        prob_missing = df[prob_col].isna()
        if prob_missing.any() and df[reg_col].notna().any():
            df.loc[prob_missing, prob_col] = df.loc[prob_missing, reg_col]
        if df[prob_col].notna().any():
            available.append(prob_source)

    # Logit-space weighted ensemble
    if not available:
        df['ensemble_prob'] = np.nan
        return df

    w = {s: weights.get(s, 0) for s in available}
    tw = sum(w.values())
    if tw == 0:
        tw = len(w)
        w = {s: 1.0 / tw for s in w}
    else:
        w = {s: v / tw for s, v in w.items()}

    clip_lo, clip_hi = 1e-4, 1 - 1e-4
    logit_sum = np.zeros(len(df))
    for s, wt in w.items():
        p = df[f'{s}_src'].fillna(0).clip(clip_lo, clip_hi)
        logit_sum += np.log(p / (1 - p)) * wt

    df['ensemble_prob'] = 1.0 / (1.0 + np.exp(-logit_sum))

    # Normalize within each race
    for _, race_df in df.groupby('race'):
        idx = race_df.index
        total = df.loc[idx, 'ensemble_prob'].sum()
        if total > 0:
            df.loc[idx, 'ensemble_prob'] /= total

    return df


# =============================================================================
# 3. EXPERT ADJUSTMENTS
# =============================================================================

def load_expert_ratings():
    """Load expert_ratings.json."""
    path = DATA_PATH / 'expert_ratings.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def apply_expert_adjustments(df, date_str, track, skip_tipmeerkat=True):
    """
    Apply expert logit deltas to ensemble_prob, weighted by Elo rating.

    For each expert with picks for this date/track:
      1. Load their logit deltas from expert_picks/{date}/{track}/{expert}.csv
      2. Look up their Elo weight from expert_ratings.json
      3. Apply: logit(ensemble) += weight * logit_delta per horse
      4. Renormalize per race

    Experts without ratings get a small default weight based on INITIAL_RATING
    from expert_elo.py (effectively near-zero influence).

    Args:
      skip_tipmeerkat: if True (the new default), exclude any expert pick
        CSV whose name starts with 'tm_' (the prefix used by
        scrape_tipmeerkat.py). Their Elo ratings still update via
        expert_elo.py replay — this only excludes them from THIS run's
        adjustment. Default is True because tipmeerkat data is currently
        bugged; pass False to opt back in.
    """
    expert_dir = DATA_PATH / 'expert_picks' / date_str / track
    if not expert_dir.exists():
        return df

    csv_files = sorted(expert_dir.glob('*.csv'))
    if skip_tipmeerkat:
        n_before = len(csv_files)
        csv_files = [p for p in csv_files if not p.stem.startswith('tm_')]
        n_skipped = n_before - len(csv_files)
        if n_skipped:
            print(f"  Skipped {n_skipped} tipmeerkat (tm_*) expert(s) "
                  f"(default; pass --include-tipmeerkat to override)")
    if not csv_files:
        return df

    ratings = load_expert_ratings()

    # Derive weights: (rating - 1500) / scale_factor
    # Scale so that a rating of 1535 (proven sharp) gives weight ~0.035
    # and 1510 (mildly useful) gives ~0.01
    BASELINE = 1500
    SCALE = 1000  # rating points per unit of logit influence

    print(f"\nApplying expert adjustments:")

    df = df.copy()
    clip = 1e-4

    # Convert ensemble to logit space
    ens_probs = df['ensemble_prob'].clip(clip, 1 - clip)
    logits = np.log(ens_probs / (1 - ens_probs)).values

    any_applied = False
    for csv_path in csv_files:
        expert_name = csv_path.stem
        expert_df = pd.read_csv(csv_path)

        if expert_df.empty:
            continue

        # Get weight from rating
        if expert_name in ratings:
            rating = ratings[expert_name]['rating']
        else:
            rating = BASELINE  # unknown expert, neutral weight

        weight = (rating - BASELINE) / SCALE
        if abs(weight) < 0.0001:
            print(f"  {expert_name:<22s} rating={rating:.0f}  weight≈0, skipping")
            continue

        # Build delta map: {(race, horse_number): logit_delta}
        n_applied = 0
        for _, row in expert_df.iterrows():
            race = int(row['race'])
            hn = int(row['horse_number'])
            delta = float(row['logit_delta'])

            mask = (df['race'] == race) & (df['horse_number'] == hn)
            idx = df.index[mask]
            if len(idx) > 0:
                logits[idx[0]] += weight * delta
                n_applied += 1

        print(f"  {expert_name:<22s} rating={rating:.1f}  weight={weight:+.4f}  "
              f"adjusted {n_applied} horses")
        any_applied = True

    if not any_applied:
        print("  No expert adjustments applied")
        return df

    # Convert back from logit
    df['ensemble_prob'] = 1.0 / (1.0 + np.exp(-logits))

    # Renormalize per race
    for _, race_df in df.groupby('race'):
        idx = race_df.index
        total = df.loc[idx, 'ensemble_prob'].sum()
        if total > 0:
            df.loc[idx, 'ensemble_prob'] /= total

    return df


# =============================================================================
# 4. EDGE CALCULATION
# =============================================================================

def calculate_edges(df, takeout=WIN_TAKEOUT):
    """
    Compute edge and expected value accounting for track takeout.

    For a $2 win bet on a horse at ML odds of X/1:
      - Payout if win: $2 * (1 + X) but pool has takeout
      - Approximate payout: $2 / ml_prob * (1 - takeout)
      - EV per $2 = 2 * (ensemble_prob / ml_prob) * (1 - takeout) - 2
      - ROI = (ensemble_prob / ml_prob) * (1 - takeout) - 1

    Positive ROI = positive expected value bet.
    Breakeven requires ensemble_prob > ml_prob / (1 - takeout).

    Kelly uses the takeout-adjusted odds.
    """
    df = df.copy()
    if 'ml_win_prob_norm' not in df.columns:
        df['edge'] = np.nan
        df['ev_roi'] = np.nan
        df['ev_per_2'] = np.nan
        df['kelly'] = np.nan
        return df

    proj = df['ensemble_prob']
    ml = df['ml_win_prob_norm']

    # Raw probability edge (before takeout)
    df['edge'] = proj - ml

    # Expected ROI on a $2 win bet (after takeout)
    # ROI = (our_prob / ml_prob) * (1 - takeout) - 1
    df['ev_roi'] = np.where(ml > 0, (proj / ml) * (1 - takeout) - 1, np.nan)

    # EV in dollars per $2 bet
    df['ev_per_2'] = np.where(ml > 0, 2 * df['ev_roi'], np.nan)

    # Kelly on takeout-adjusted odds
    # Effective decimal odds = (1 / ml) * (1 - takeout)
    # Kelly = (p * odds - 1) / (odds - 1)
    eff_odds = np.where(ml > 0, (1 / ml) * (1 - takeout), 0)
    df['kelly'] = np.where(
        eff_odds > 1,
        np.clip((proj * eff_odds - 1) / (eff_odds - 1), 0, 1),
        0
    )
    df['half_kelly'] = df['kelly'] / 2

    return df


# =============================================================================
# 5. BET RECOMMENDATIONS
# =============================================================================

def generate_bets(df,
                  logistic_min=LOGISTIC_MIN,
                  logistic_edge=LOGISTIC_EDGE_THRESHOLD,
                  ridge_edge_multiplier=RIDGE_EDGE_MULTIPLIER,
                  ridge_min_multiplier=RIDGE_MIN_MULTIPLIER,
                  raw_edge_k=RAW_EDGE_K,
                  raw_min_k=RAW_MIN_K,
                  takeout=WIN_TAKEOUT):
    """
    Generate bet recommendations using tri-model agreement.

    Three independent signals, each with its own break-even:

      1. LOG (logistic): ev_per_2 on a $2 win bet, post-takeout
         - min floor: logistic_min ($0)
         - bet edge:  logistic_edge ($0.05 = 2.5% edge)

      2. RDG (log-ridge on 4·W+2·P+1·S, normalized to $18.30/race):
         - fair share per horse = RIDGE_RACE_TOTAL / N
         - break-even = fair share × 1/(1-takeout)   [gross-up of takeout]
         - min floor: break_even × ridge_min_multiplier
         - bet edge:  break_even × ridge_edge_multiplier

      3. RAW (raw-ridge on W+P+S, NOT normalized):
         Break-even is calibrated from the training CV prediction distribution
         (μ and σ are saved in raw_value_model.json and surfaced on
         df.attrs['raw_cv_mean'], df.attrs['raw_cv_std']).
         - positive (above BE):  raw_value ≥ cv_mean
         - bet edge:             raw_value ≥ cv_mean + raw_edge_k × cv_std
         - floor veto:           raw_value < cv_mean + raw_min_k × cv_std
                                 (default raw_min_k = -0.5 → "actively bad")

    Bet logic:
      • FLAGGED: any model at its edge threshold (report-only)
      • BET: ≥1 model at edge AND no active model below its floor
      • AGREE: BET + ≥2/3 models above their break-even (high-confidence)

    triggered_by examples:
        "LOG+RDG+RAW+AGREE", "LOG+AGREE", "RAW" (solo), "RDG"

    Bet type by ML odds:  ML ≤ 4.5 → W,  ≤ 9 → WP,  else WPS.
    """
    bets = []
    race_summaries = {}

    has_logistic = 'ev_per_2' in df.columns and df['ev_per_2'].notna().any()
    has_ridge = 'ridge_value' in df.columns and df['ridge_value'].notna().any()
    has_raw = 'raw_value' in df.columns and df['raw_value'].notna().any()

    takeout_gross = 1.0 / max(1e-6, 1.0 - takeout)

    # Raw-ridge calibration stats from df.attrs (set by apply_raw_value_model)
    raw_cv_mean = df.attrs.get('raw_cv_mean') if has_raw else None
    raw_cv_std  = df.attrs.get('raw_cv_std')  if has_raw else None

    if has_raw and (raw_cv_mean is None or raw_cv_std is None):
        # Shouldn't happen in the normal flow, but be defensive:
        # compute from the data itself. Warns once at top of generate_bets.
        print(f"  WARNING: raw_cv_mean/std missing on df — "
              f"using empirical card stats (less stable)")
        raw_cv_mean = float(df['raw_value'].mean())
        raw_cv_std  = float(df['raw_value'].std())

    # Raw-ridge thresholds are card-wide constants (not per-race) because
    # they're derived from the model's training distribution, not from
    # a per-race normalization. Same for every race in this card.
    if has_raw:
        raw_break_even = raw_cv_mean
        raw_edge       = raw_cv_mean + raw_edge_k * raw_cv_std
        raw_min        = raw_cv_mean + raw_min_k  * raw_cv_std
    else:
        raw_break_even = raw_edge = raw_min = None

    for race_num, race_df in df.groupby('race'):
        n_horses = len(race_df)

        # Log-ridge break-evens ──────────────────────────────────────
        ridge_raw_fair = RIDGE_RACE_TOTAL / n_horses if n_horses > 0 else 0
        ridge_break_even = ridge_raw_fair * takeout_gross
        ridge_min = ridge_break_even * ridge_min_multiplier
        ridge_edge = ridge_break_even * ridge_edge_multiplier

        # Per-horse model flags ──────────────────────────────────────
        def _model_flags(row):
            flags = {
                'log_active': False, 'log_pos': False, 'log_bet': False, 'log_qual': True,
                'rdg_active': False, 'rdg_pos': False, 'rdg_bet': False, 'rdg_qual': True,
                'raw_active': False, 'raw_pos': False, 'raw_bet': False, 'raw_qual': True,
            }

            if has_logistic and pd.notna(row.get('ev_per_2')):
                ev = row['ev_per_2']
                flags['log_active'] = True
                flags['log_pos']    = ev > 0
                flags['log_bet']    = ev >= logistic_edge
                flags['log_qual']   = ev >= logistic_min

            if has_ridge and pd.notna(row.get('ridge_value')):
                rv = row['ridge_value']
                flags['rdg_active'] = True
                flags['rdg_pos']    = rv >= ridge_break_even
                flags['rdg_bet']    = rv >= ridge_edge
                flags['rdg_qual']   = rv >= ridge_min

            if has_raw and pd.notna(row.get('raw_value')):
                rw = row['raw_value']
                flags['raw_active'] = True
                flags['raw_pos']    = rw >= raw_break_even
                flags['raw_bet']    = rw >= raw_edge
                flags['raw_qual']   = rw >= raw_min

            return flags

        race_flags = {idx: _model_flags(row) for idx, row in race_df.iterrows()}

        def _qualified(f):
            if f['log_active'] and not f['log_qual']:
                return False
            if f['rdg_active'] and not f['rdg_qual']:
                return False
            if f['raw_active'] and not f['raw_qual']:
                return False
            return f['log_bet'] or f['rdg_bet'] or f['raw_bet']

        qualified_mask = race_df.index.map(lambda i: _qualified(race_flags[i]))
        edge_horses = race_df[qualified_mask].copy()

        def _rank_score(idx):
            f = race_flags[idx]
            score = 0.0
            if f['log_active']:
                ev = edge_horses.loc[idx, 'ev_per_2']
                score += ev / max(logistic_edge, 1e-6)
            if f['rdg_active']:
                rv = edge_horses.loc[idx, 'ridge_value']
                score += (rv - ridge_break_even) / max(ridge_break_even, 1e-6)
            if f['raw_active']:
                rw = edge_horses.loc[idx, 'raw_value']
                # Measure in units of std-above-mean — consistent scale
                # with the edge threshold (edge is at +K·σ).
                score += (rw - raw_break_even) / max(raw_cv_std, 1e-6)
            return score

        if len(edge_horses) > 0:
            edge_horses['_rank_score'] = edge_horses.index.map(_rank_score)
            edge_horses = edge_horses.sort_values('_rank_score', ascending=False)

        if len(edge_horses) == 0:
            flagged = []
            for idx, row in race_df.iterrows():
                f = race_flags[idx]
                if f['log_bet'] or f['rdg_bet'] or f['raw_bet']:
                    flagged.append(int(row['horse_number']))
            race_summaries[race_num] = {
                'action': 'PASS',
                'horses': [],
                'flagged': flagged,
                'ridge_break_even': round(ridge_break_even, 2) if has_ridge else None,
                'raw_break_even':   round(raw_break_even, 2)   if has_raw   else None,
            }
            continue

        race_bets = []
        for idx, row in edge_horses.iterrows():
            hn = int(row['horse_number'])
            ml_dec = row.get('ml_decimal', np.nan)
            prob = row['ensemble_prob']
            kelly = row.get('half_kelly', 0)
            f = race_flags[idx]

            if pd.notna(ml_dec) and ml_dec <= ML_WIN_ONLY_MAX:
                bet_type = 'W'
            elif pd.notna(ml_dec) and ml_dec <= ML_WP_MAX:
                bet_type = 'WP'
            else:
                bet_type = 'WPS'

            triggers = []
            if f['log_bet']: triggers.append('LOG')
            if f['rdg_bet']: triggers.append('RDG')
            if f['raw_bet']: triggers.append('RAW')

            n_positive = sum([f['log_pos'], f['rdg_pos'], f['raw_pos']])
            is_agree = n_positive >= 2
            if is_agree:
                triggers.append('AGREE')

            triggered_by = '+'.join(triggers) if triggers else ''

            bet_record = {
                'race': race_num,
                'horse_number': hn,
                'bet_type': bet_type,
                'ensemble_prob': round(prob, 4),
                'ml_prob': round(row.get('ml_win_prob_norm', np.nan), 4),
                'edge': round(row.get('edge', np.nan), 4),
                'ev_per_2': round(row.get('ev_per_2', np.nan), 2),
                'ev_roi': round(row.get('ev_roi', np.nan), 3),
                'kelly': round(kelly, 4),
                'ml_odds': row.get('ml_odds_raw', ''),
                'triggered_by': triggered_by,
                'n_models_positive': n_positive,
                'is_agree': is_agree,
            }

            if has_ridge:
                rv = row.get('ridge_value', np.nan)
                bet_record['ridge_value']      = round(rv, 2) if pd.notna(rv) else np.nan
                bet_record['ridge_break_even'] = round(ridge_break_even, 2)
                bet_record['ridge_edge']       = round(rv - ridge_break_even, 2) if pd.notna(rv) else np.nan

            if has_raw:
                rw = row.get('raw_value', np.nan)
                bet_record['raw_value']      = round(rw, 3) if pd.notna(rw) else np.nan
                bet_record['raw_break_even'] = round(raw_break_even, 3)
                bet_record['raw_sigma']      = round((rw - raw_break_even) / raw_cv_std, 2) if pd.notna(rw) else np.nan

            race_bets.append(bet_record)

        bets.extend(race_bets)
        race_summaries[race_num] = {
            'action': 'BET',
            'n_edge_horses': len(edge_horses),
            'horses': [int(h) for h in edge_horses['horse_number']],
            'ridge_break_even': round(ridge_break_even, 2) if has_ridge else None,
            'raw_break_even':   round(raw_break_even, 2)   if has_raw   else None,
        }

    bet_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    return bet_df, race_summaries


# =============================================================================
# 6. REPORTING
# =============================================================================

def print_race_report(df, race_num, bet_df=None, race_summary=None,
                      takeout=WIN_TAKEOUT, extra_banner_info=''):
    """
    Print formatted report for one race.

    Markers shown next to each horse:
        ◀ BET   — at least one model fired as a bet (qualified)
        ◀ AGREE — BET + 2+ models above break-even (high-confidence)
        ◀ flag  — at least one model above its edge threshold but
                   another model's floor vetoed it (shown for visibility)

    extra_banner_info: optional string appended to the race banner inside
        the parentheses (after raw μ/σ, before the closing paren). Used by
        downstream callers — e.g. predict_v3 injects per-race entropy and
        gate-eligibility tags here without needing to monkey-patch.
    """
    race = df[df['race'] == race_num].sort_values('ensemble_prob', ascending=False)
    if race.empty:
        return

    has_ridge = 'ridge_value' in race.columns and race['ridge_value'].notna().any()
    has_raw = 'raw_value' in race.columns and race['raw_value'].notna().any()

    n_horses = len(race)
    takeout_gross = 1.0 / max(1e-6, 1.0 - takeout)
    ridge_be = (RIDGE_RACE_TOTAL / n_horses) * takeout_gross if n_horses > 0 else 0

    # Raw-ridge BE is card-wide (calibrated from training), not per-race.
    # Pulled from df.attrs set by apply_raw_value_model.
    raw_cv_mean = df.attrs.get('raw_cv_mean') if has_raw else None
    raw_cv_std  = df.attrs.get('raw_cv_std')  if has_raw else None
    raw_be = raw_cv_mean if raw_cv_mean is not None else 0

    print(f"\n{'='*88}")
    banner = f"RACE {race_num}   (N={n_horses}, ridge BE=${ridge_be:.2f}"
    if has_raw and raw_cv_mean is not None:
        banner += f", raw μ=${raw_cv_mean:.3f} σ=${raw_cv_std:.3f}"
    if extra_banner_info:
        banner += f", {extra_banner_info}"
    banner += ")"
    print(banner)
    print(f"{'='*88}")

    # Build header
    src_cols = [c for c in race.columns if c.endswith('_beyer') and not c.startswith('ensemble')]
    header = f"{'#':>3} "
    for col in src_cols:
        header += f"{col.replace('_beyer','')[:6]:>6} "
    header += f"{'Prob':>7} "
    if 'ml_win_prob_norm' in race.columns:
        header += f"{'ML%':>6} {'Edge':>7} {'EV$2':>6} "
    if has_ridge:
        header += f"{'Ridge$':>7} "
    if has_raw:
        header += f"{'Raw$':>6} "
    header += f"{'½K':>5} {'ML':>6}"
    print(header)
    print("-" * len(header))

    # Precompute bet-status lookup from bet_df (if any)
    bet_lookup = {}
    if bet_df is not None and not bet_df.empty:
        race_bets_df = bet_df[bet_df['race'] == race_num]
        for _, b in race_bets_df.iterrows():
            bet_lookup[int(b['horse_number'])] = {
                'is_agree': bool(b.get('is_agree', False)),
                'triggered_by': b.get('triggered_by', ''),
            }

    # FLAG set: horses where some single model saw edge but the horse
    # didn't make the qualified bet list (vetoed by another model's floor)
    flagged_set = set()
    if race_summary is not None:
        flagged_set = set(race_summary.get('flagged', []))

    for _, row in race.iterrows():
        hn = int(row['horse_number'])
        line = f"{hn:>3} "
        for col in src_cols:
            v = row.get(col, np.nan)
            line += f"{v:>6.0f} " if pd.notna(v) else f"{'—':>6} "
        line += f"{row['ensemble_prob']:>6.1%} "
        if 'ml_win_prob_norm' in race.columns:
            ml = row.get('ml_win_prob_norm', np.nan)
            edge = row.get('edge', np.nan)
            ev2 = row.get('ev_per_2', np.nan)
            line += f"{ml:>5.1%} " if pd.notna(ml) else f"{'—':>5} "
            line += f"{edge:>+6.1%} " if pd.notna(edge) else f"{'—':>7} "
            line += f"{ev2:>+5.2f} " if pd.notna(ev2) else f"{'—':>6} "
        if has_ridge:
            rv = row.get('ridge_value', np.nan)
            line += f"{rv:>6.2f} " if pd.notna(rv) else f"{'—':>7} "
        if has_raw:
            rw = row.get('raw_value', np.nan)
            line += f"{rw:>5.2f} " if pd.notna(rw) else f"{'—':>6} "
        hk = row.get('half_kelly', np.nan)
        ml_raw = row.get('ml_odds_raw', '')
        line += f"{hk:>5.3f}" if pd.notna(hk) else f"{'—':>5}"
        line += f" {ml_raw:>6}" if ml_raw else ""

        # Bet-status marker hierarchy:
        #   AGREE > BET > flag
        if hn in bet_lookup:
            info = bet_lookup[hn]
            if info['is_agree']:
                line += f"  ◀ AGREE [{info['triggered_by']}]"
            else:
                line += f"  ◀ BET [{info['triggered_by']}]"
        elif hn in flagged_set:
            line += "  ◀ flag"

        print(line)

    # Print bet recommendations for this race
    if bet_df is not None and not bet_df.empty:
        race_bets_out = bet_df[bet_df['race'] == race_num]
        if not race_bets_out.empty:
            print(f"\n  Bets:")
            for _, b in race_bets_out.iterrows():
                hn = str(b['horse_number'])
                parts = [f"    {b['bet_type']:<5s} #{hn:<4s}"]
                if pd.notna(b.get('ev_per_2')):
                    parts.append(f"LogEV=${b['ev_per_2']:+.2f}")
                if 'ridge_value' in b and pd.notna(b.get('ridge_value')):
                    parts.append(f"Ridge=${b['ridge_value']:.2f}")
                if 'raw_value' in b and pd.notna(b.get('raw_value')):
                    # Show raw_value + σ-distance from break-even. The σ
                    # is what determines the bet trigger, so it's more
                    # informative than the raw dollar alone (which clusters
                    # narrowly around μ).
                    sig_str = f"{b['raw_sigma']:+.1f}σ" if 'raw_sigma' in b and pd.notna(b.get('raw_sigma')) else ""
                    parts.append(f"Raw=${b['raw_value']:.2f}{'('+sig_str+')' if sig_str else ''}")
                if 'triggered_by' in b and b.get('triggered_by'):
                    parts.append(f"[{b['triggered_by']}]")
                print("  ".join(parts))


def percentage_to_american_odds(p):
    """Convert probability to American odds string."""
    if pd.isna(p) or p <= 0 or p >= 1:
        return '—'
    if p >= 0.5:
        return f"{int(-100 * p / (1 - p)):+d}"
    return f"+{int(100 * (1 - p) / p)}"


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def run_predictions(date_str, track, num_sims=100000, use_regression=False,
                    skip_tipmeerkat=True, suppress_trigger_explanation=False,
                    race_banner_extras=None):
    """Full forward-looking prediction pipeline.

    Args:
      skip_tipmeerkat: if True (the new default), exclude tm_* expert CSVs
        from both the ensemble's expert adjustment and the regression's
        human_pool feature. Their Elo ratings still update via
        expert_elo.py — this only affects the bet recommendations for
        THIS run. Default is True because tipmeerkat data is currently
        bugged; pass False to opt back in.
      suppress_trigger_explanation: if True, skip the "Bet trigger:..."
        through "AGREE marker:..." block in the summary. Used by callers
        (e.g. predict_v3) that have their own trigger logic and don't
        want the v2 explanation cluttering output.
      race_banner_extras: optional {race_num: str} dict. Each value is
        appended to that race's banner inside the parentheses. Lets
        callers inject per-race context (e.g. predict_v3 puts entropy
        and gate-eligibility tags here).
    """

    # --- Load fitted params ---
    params = load_fitted_params()
    if params:
        std_dev = params['std_dev']
        degrees_of_freedom = params['degrees_of_freedom']
        ml_reg = params.get('ml_regularization', 0.25)
        weights = params['weights']
        print(f"Loaded fitted params (std={std_dev:.2f}, df={degrees_of_freedom:.2f}, "
              f"ml_reg={ml_reg:.2f})")
        print(f"  Weights: {weights}")
    else:
        std_dev = DEFAULT_STD
        degrees_of_freedom = DEFAULT_DF
        ml_reg = 0.25
        weights = DEFAULT_WEIGHTS
        print(f"No fitted_params.json found, using defaults")

    # --- Load AI picks ---
    print(f"\nLoading AI picks for {date_str} / {track}...")
    df = load_ai_picks(date_str, track)
    print(f"  {len(df)} horses across {df['race'].nunique()} races")

    # --- Load UPC entries (scratches + ML) ---
    upc = load_upc_entries(track)
    if upc is not None:
        print(f"Loading entries from upc_entries/{track}.csv...")

        # Filter to running horses.
        # Two things count as scratched:
        #   1. Explicitly marked running != 1 in entries (official scratch)
        #   2. In AI picks but not in entries at all (late scratch that never
        #      got an entries row, or AI dreamed up a horse that's not racing)
        #
        # Previously only (1) was filtered, so late-scratch horses lingered
        # with NaN ML odds, NaN edges, and zero ev_per_2 — visible in the
        # report but polluting ensemble_prob normalization. Treating them
        # as scratched and dropping them triggers the normal downstream
        # renormalization (per-race ensemble_prob, ML%, ridge, raw all get
        # recomputed after the drop), so the remaining horses' probabilities
        # correctly absorb the scratched mass.
        entries_keys = set(zip(upc['race'], upc['horse_number']))
        if 'running' in upc.columns:
            entries_keys = set(zip(
                upc[upc['running'] == 1]['race'],
                upc[upc['running'] == 1]['horse_number'],
            ))

        before = len(df)
        ai_keys = set(zip(df['race'], df['horse_number']))
        scratched_or_missing = ai_keys - entries_keys

        if scratched_or_missing:
            # Break down the reasons so it's clear in the log why a horse
            # was dropped — helpful when scratches happen last-minute.
            in_entries_not_running = {
                (r, h) for (r, h) in scratched_or_missing
                if 'running' in upc.columns and (r, h) in
                set(zip(upc[upc['running'] != 1]['race'],
                        upc[upc['running'] != 1]['horse_number']))
            }
            not_in_entries = scratched_or_missing - in_entries_not_running

            df = df[df.apply(
                lambda r: (r['race'], r['horse_number']) not in scratched_or_missing,
                axis=1,
            )]
            print(f"  Removed {before - len(df)} non-running horses:")
            if in_entries_not_running:
                print(f"    {len(in_entries_not_running)} scratched "
                      f"(running=0 in entries)")
            if not_in_entries:
                print(f"    {len(not_in_entries)} missing from entries "
                      f"(late scratch or AI-only horse):")
                for r, h in sorted(not_in_entries):
                    print(f"      R{r} #{h}")
        else:
            print(f"  No scratches")

        # Merge ML odds
        if 'ml_odds' in upc.columns:
            upc_ml = upc[['race', 'horse_number', 'ml_odds']].drop_duplicates()

            # Diagnostic: check for merge mismatches
            ai_keys = set(zip(df['race'], df['horse_number']))
            upc_keys = set(zip(upc_ml['race'], upc_ml['horse_number']))
            ai_only = ai_keys - upc_keys
            if ai_only:
                print(f"  WARNING: {len(ai_only)} AI horses not in UPC entries (no ML odds):")
                for r, h in sorted(ai_only):
                    print(f"    R{r} #{h}")

            df = df.merge(upc_ml, on=['race', 'horse_number'], how='left')
            df['ml_odds_raw'] = df['ml_odds']
            df['ml_decimal'] = df['ml_odds'].apply(_ml_odds_to_decimal)
            df['ml_win_prob'] = df['ml_odds'].apply(_parse_ml_odds)

            # Normalize ML within each race
            df['ml_win_prob_norm'] = np.nan
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                raw = race_df['ml_win_prob'].values
                valid = ~np.isnan(raw)
                if valid.any():
                    probs = raw[valid]
                    probs = probs / probs.sum()
                    df.loc[idx[valid], 'ml_win_prob_norm'] = probs
            print(f"  ML odds loaded for {df['ml_win_prob'].notna().sum()} horses")
    else:
        print(f"  No upc_entries/{track}.csv found — running without ML odds")

    # --- Build probabilities ---
    if use_regression:
        print(f"\nBuilding regression probabilities...")
        df = build_regression_probs(df, date_str, track,
                                    skip_tipmeerkat=skip_tipmeerkat)
    else:
        print(f"\nBuilding ensemble (sims={num_sims:,})...")
        df = build_ensemble(df, std_dev, degrees_of_freedom, num_sims, weights,
                            ml_regularization=ml_reg)
        # Apply expert adjustments (only in ensemble mode — regression includes them)
        df = apply_expert_adjustments(df, date_str, track,
                                      skip_tipmeerkat=skip_tipmeerkat)

    # --- Edge calculation ---
    df = calculate_edges(df)

    # --- Ridge value predictions (log-ridge: 4·W+2·P+1·S) ---
    df = apply_ridge_value_model(df, date_str, track,
                                 skip_tipmeerkat=skip_tipmeerkat)

    # --- Raw value predictions (raw-ridge: W+P+S, no transform) ---
    # Third independent signal. The value-hunter — lets the heavy tail of
    # big payouts pull coefficients toward overlay signals. Compare its
    # triggers to logistic in the history to see which is finding actual
    # value vs just picking favorites.
    df = apply_raw_value_model(df, date_str, track,
                               skip_tipmeerkat=skip_tipmeerkat)

    # --- Bet recommendations ---
    bet_df, race_summaries = generate_bets(
        df,
        logistic_min=LOGISTIC_MIN,
        logistic_edge=LOGISTIC_EDGE_THRESHOLD,
        ridge_edge_multiplier=RIDGE_EDGE_MULTIPLIER,
        ridge_min_multiplier=RIDGE_MIN_MULTIPLIER,
        raw_edge_k=RAW_EDGE_K,
        raw_min_k=RAW_MIN_K,
        takeout=WIN_TAKEOUT,
    )

    # --- Print reports ---
    extras_map = race_banner_extras or {}
    for race_num in sorted(df['race'].unique()):
        print_race_report(df, race_num, bet_df,
                          race_summary=race_summaries.get(race_num),
                          takeout=WIN_TAKEOUT,
                          extra_banner_info=extras_map.get(race_num, ''))

    # --- Summary ---
    n_bet_races = sum(1 for v in race_summaries.values() if v['action'] == 'BET')
    n_pass_races = sum(1 for v in race_summaries.values() if v['action'] == 'PASS')

    print(f"\n{'='*80}")
    print(f"SUMMARY: {date_str} / {track}")
    print(f"{'='*80}")
    print(f"  Races: {df['race'].nunique()}")
    print(f"  Bet races: {n_bet_races}  |  Pass: {n_pass_races}")

    has_ridge = 'ridge_value' in df.columns and df['ridge_value'].notna().any()
    has_raw   = 'raw_value'   in df.columns and df['raw_value'].notna().any()

    field_sizes = df.groupby('race').size()
    takeout_gross = 1.0 / max(1e-6, 1.0 - WIN_TAKEOUT)

    # Print bet trigger rules + per-race break-even ranges
    # (skipped when suppress_trigger_explanation=True — used by predict_v3,
    # which has its own trigger rules and prints them up-front instead.)
    if not suppress_trigger_explanation:
        print(f"\n  Bet trigger: at least one of three models above its edge,")
        print(f"               AND no active model below its qualification floor.")
        print(f"")
        print(f"  Logistic: bet ≥ ${LOGISTIC_EDGE_THRESHOLD:.2f} EV, floor ≥ ${LOGISTIC_MIN:.2f}")
        if has_ridge:
            rd_min = (RIDGE_RACE_TOTAL / field_sizes.max()) * takeout_gross
            rd_max = (RIDGE_RACE_TOTAL / field_sizes.min()) * takeout_gross
            print(f"  Log-ridge: break-even this card ${rd_min:.2f}–${rd_max:.2f}, "
                  f"bet ≥ {RIDGE_EDGE_MULTIPLIER:.2f}× BE")
        if has_raw:
            raw_cv_mean = df.attrs.get('raw_cv_mean')
            raw_cv_std  = df.attrs.get('raw_cv_std')
            if raw_cv_mean is not None and raw_cv_std is not None:
                raw_edge = raw_cv_mean + RAW_EDGE_K * raw_cv_std
                raw_min  = raw_cv_mean + RAW_MIN_K  * raw_cv_std
                print(f"  Raw-ridge: μ=${raw_cv_mean:.3f}, σ=${raw_cv_std:.3f}  "
                      f"bet ≥ ${raw_edge:.3f} (μ+{RAW_EDGE_K}σ), "
                      f"floor ${raw_min:.3f} (μ{RAW_MIN_K:+}σ)")
        print(f"  AGREE marker: ≥2 of 3 models above their break-even (positive value)")

    if not bet_df.empty:
        print(f"\n  Total bets: {len(bet_df)}")
        if 'is_agree' in bet_df.columns:
            n_agree = int(bet_df['is_agree'].sum())
            print(f"    AGREE bets (≥2 models positive): {n_agree}")

        # Breakdown by which model(s) triggered
        if 'triggered_by' in bet_df.columns:
            print(f"\n  Triggered-by breakdown:")
            trig_counts = bet_df['triggered_by'].value_counts()
            for trig, n in trig_counts.items():
                print(f"    {trig:<24s} {n}")

        print(f"\n  Bet Log:")
        for _, b in bet_df.iterrows():
            parts = [
                f"    R{int(b['race']):>2}",
                f"{b['bet_type']:<4s}",
                f"#{str(b['horse_number']):<4s}",
                f"prob={b['ensemble_prob']:.1%}",
            ]
            if pd.notna(b.get('ev_per_2')):
                parts.append(f"LogEV=${b['ev_per_2']:+.2f}")
            if 'ridge_value' in b and pd.notna(b.get('ridge_value')):
                parts.append(f"Rdg=${b['ridge_value']:.2f}")
            if 'raw_value' in b and pd.notna(b.get('raw_value')):
                sig_str = f"{b['raw_sigma']:+.1f}σ" if 'raw_sigma' in b and pd.notna(b.get('raw_sigma')) else ""
                parts.append(f"Raw=${b['raw_value']:.2f}{'('+sig_str+')' if sig_str else ''}")
            if 'triggered_by' in b and b.get('triggered_by'):
                parts.append(f"[{b['triggered_by']}]")
            print("  ".join(parts))

    # --- Save outputs ---
    out_dir = DATA_PATH / 'predictions' / date_str / track
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / 'predictions.csv'
    df.to_csv(pred_path, index=False, encoding='utf-8')
    print(f"\n  Predictions → {pred_path}")

    if not bet_df.empty:
        bet_path = out_dir / 'bet_log.csv'
        bet_df.to_csv(bet_path, index=False, encoding='utf-8')
        print(f"  Bet log     → {bet_path}")

    return df, bet_df


def _apply_linear_model(df, date_str, track, model_path, output_col,
                        race_total, floor, label, skip_tipmeerkat=True):
    """
    Shared implementation for both apply_ridge_value_model (log-ridge, 4·W+2·P+1·S)
    and apply_raw_value_model (raw-ridge, W+P+S). Handles feature construction,
    transform inversion (log1p ↔ expm1), and per-race normalization to race_total.

    Args:
        model_path: Path to the model JSON
        output_col: Name of the column to populate (e.g. 'ridge_value', 'raw_value')
        race_total: Per-race sum to normalize predictions to. Can be a callable
                    `f(takeout) -> float` for takeout-dependent totals (raw),
                    or a constant (log-ridge).
        floor: Minimum positive prediction before normalizing (prevents a
               single negative from blowing up the scale)
        label: Human-readable label for the print banner

    Returns df with `output_col` populated. If the model file doesn't exist,
    returns df unchanged (the bet logic falls back to whatever models are
    present).
    """
    if not model_path.exists():
        print(f"\n  No {model_path.name} — skipping {label} model")
        return df

    with open(model_path) as f:
        model = json.load(f)

    feature_cols = model['feature_cols']
    feature_is_prob = model.get('feature_is_prob', [True] * len(feature_cols))
    coefs = np.array(model['coefficients'])
    intercept = model['intercept']

    print(f"\n  Loading {label} model (α={model.get('regularization_alpha')}, "
          f"CV top-pick=${model.get('cv_top_pick_value')})")

    df = df.copy()
    df = _ensure_regression_features(df, date_str, track,
                                     skip_tipmeerkat=skip_tipmeerkat)

    col_map = {
        'ml_prob_norm': 'ml_win_prob_norm',
        'claude_prob': 'claude_win_prob',
        'gemini_prob': 'gemini_win_prob',
        'openai_prob': 'openai_win_prob',
        'claude_beyer_sim': 'claude_beyer_sim',
        'gemini_beyer_sim': 'gemini_beyer_sim',
        'openai_beyer_sim': 'openai_beyer_sim',
        'openai_closing': 'openai_closing_prob',
        'claude_closing': 'claude_closing_prob',
        'gemini_closing': 'gemini_closing_prob',
        'human_prob': 'human_prob',
        'inv_field_size': 'inv_field_size',
        'ml_x_fieldsize': 'ml_x_fieldsize',
    }

    # Normalize probability features per race (to match training)
    for fc, is_prob in zip(feature_cols, feature_is_prob):
        if not is_prob:
            continue
        df_col = col_map.get(fc, fc)
        if df_col in df.columns:
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                vals = df.loc[idx, df_col].values
                valid = ~np.isnan(vals)
                if valid.any() and np.nansum(vals[valid]) > 0:
                    df.loc[idx, df_col] = np.where(
                        valid, vals / np.nansum(vals), np.nan
                    )

    # Build feature matrix
    X_cols = []
    for fc in feature_cols:
        df_col = col_map.get(fc, fc)
        if df_col not in df.columns:
            print(f"    WARNING: {label} feature '{fc}' not found, using 0")
            df[df_col] = 0
        X_cols.append(df_col)

    X_raw = df[X_cols].fillna(0).values
    clip_lo, clip_hi = 1e-4, 1 - 1e-4

    input_space = model.get('input_space', 'raw')
    if input_space == 'mixed':
        X = np.zeros_like(X_raw, dtype=float)
        for i, is_prob in enumerate(feature_is_prob):
            if is_prob:
                clipped = np.clip(X_raw[:, i], clip_lo, clip_hi)
                X[:, i] = np.log(clipped / (1 - clipped))
            else:
                X[:, i] = X_raw[:, i]
    elif input_space == 'logit':
        X_clipped = np.clip(X_raw, clip_lo, clip_hi)
        X = np.log(X_clipped / (1 - X_clipped))
    else:
        X = X_raw

    raw_col = f'{output_col}_raw'
    df[raw_col] = X @ coefs + intercept

    # Invert target transform if the model used one (only log-ridge currently)
    target_transform = model.get('target_transform')
    if target_transform == 'log1p':
        df[raw_col] = np.expm1(df[raw_col])

    # Resolve race_total (callable for takeout-dependent raw-ridge, else const)
    if callable(race_total):
        total_per_race = race_total(WIN_TAKEOUT)
    else:
        total_per_race = race_total

    # Normalize each race to sum to total_per_race. Floor prevents a single
    # strongly negative prediction from making the race sum tiny (or zero)
    # and blowing up the scale factor for every other horse.
    df[output_col] = np.nan
    for race_num, race_df in df.groupby('race'):
        idx = race_df.index
        raw = df.loc[idx, raw_col].values.astype(float)
        floored = np.where(np.isnan(raw), floor, np.maximum(raw, floor))
        total = floored.sum()
        if total > 0:
            df.loc[idx, output_col] = floored * (total_per_race / total)

    return df


def apply_ridge_value_model(df, date_str, track, skip_tipmeerkat=True):
    """
    Apply ridge_value_model.json — the log-ridge on 4·win + 2·place + 1·show.
    Populates `ridge_value` column normalized to RIDGE_RACE_TOTAL (~$18.30)/race.
    """
    return _apply_linear_model(
        df, date_str, track,
        model_path=DATA_PATH / 'ridge_value_model.json',
        output_col='ridge_value',
        race_total=RIDGE_RACE_TOTAL,
        floor=RIDGE_FLOOR,
        label='log-ridge value',
        skip_tipmeerkat=skip_tipmeerkat,
    )


def apply_raw_value_model(df, date_str, track, skip_tipmeerkat=True):
    """
    Apply raw_value_model.json — the raw-ridge on win + place + show.

    Unlike log-ridge, we do NOT normalize raw predictions per race. Heavy
    regularization compresses the model's output around the training mean,
    and re-normalizing to a fixed total just flattens whatever signal is
    left. Instead, break-evens are defined relative to the model's own
    CV-predicted distribution (cv_pred_mean ± K·cv_pred_std from the JSON).

    Populates:
        raw_value_raw: raw model output (same as raw_value — kept for
                       parity with log-ridge's {raw, normalized} pair)
        raw_value:     same as raw_value_raw (no normalization)

    Also stashes the calibration stats on df.attrs for generate_bets:
        df.attrs['raw_cv_mean']
        df.attrs['raw_cv_std']
    """
    model_path = DATA_PATH / 'raw_value_model.json'
    if not model_path.exists():
        print(f"\n  No raw_value_model.json — skipping raw-ridge value model")
        return df

    with open(model_path) as f:
        model = json.load(f)

    feature_cols = model['feature_cols']
    feature_is_prob = model.get('feature_is_prob', [True] * len(feature_cols))
    coefs = np.array(model['coefficients'])
    intercept = model['intercept']

    cv_mean = model.get('cv_pred_mean')
    cv_std  = model.get('cv_pred_std')

    print(f"\n  Loading raw-ridge value model (α={model.get('regularization_alpha')}, "
          f"CV top-2=${model.get('cv_top2_value')})")
    if cv_mean is not None and cv_std is not None:
        print(f"    Calibration: μ=${cv_mean:.3f}, σ=${cv_std:.3f} "
              f"→ bet ≥ μ+{RAW_EDGE_K}σ = ${cv_mean + RAW_EDGE_K*cv_std:.3f}")
    else:
        print(f"    WARNING: no cv_pred_mean/std in JSON — re-run regression_analysis.py")
        print(f"    Falling back to empirical per-card stats (may be unstable on small cards)")

    df = df.copy()
    df = _ensure_regression_features(df, date_str, track,
                                     skip_tipmeerkat=skip_tipmeerkat)

    col_map = {
        'ml_prob_norm': 'ml_win_prob_norm',
        'claude_prob': 'claude_win_prob',
        'gemini_prob': 'gemini_win_prob',
        'openai_prob': 'openai_win_prob',
        'claude_beyer_sim': 'claude_beyer_sim',
        'gemini_beyer_sim': 'gemini_beyer_sim',
        'openai_beyer_sim': 'openai_beyer_sim',
        'openai_closing': 'openai_closing_prob',
        'claude_closing': 'claude_closing_prob',
        'gemini_closing': 'gemini_closing_prob',
        'human_prob': 'human_prob',
        'inv_field_size': 'inv_field_size',
        'ml_x_fieldsize': 'ml_x_fieldsize',
    }

    # Normalize probability features per race (same as training)
    for fc, is_prob in zip(feature_cols, feature_is_prob):
        if not is_prob:
            continue
        df_col = col_map.get(fc, fc)
        if df_col in df.columns:
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                vals = df.loc[idx, df_col].values
                valid = ~np.isnan(vals)
                if valid.any() and np.nansum(vals[valid]) > 0:
                    df.loc[idx, df_col] = np.where(
                        valid, vals / np.nansum(vals), np.nan
                    )

    # Build feature matrix
    X_cols = []
    for fc in feature_cols:
        df_col = col_map.get(fc, fc)
        if df_col not in df.columns:
            print(f"    WARNING: raw-ridge feature '{fc}' not found, using 0")
            df[df_col] = 0
        X_cols.append(df_col)

    X_raw = df[X_cols].fillna(0).values
    clip_lo, clip_hi = 1e-4, 1 - 1e-4

    input_space = model.get('input_space', 'raw')
    if input_space == 'mixed':
        X = np.zeros_like(X_raw, dtype=float)
        for i, is_prob in enumerate(feature_is_prob):
            if is_prob:
                clipped = np.clip(X_raw[:, i], clip_lo, clip_hi)
                X[:, i] = np.log(clipped / (1 - clipped))
            else:
                X[:, i] = X_raw[:, i]
    elif input_space == 'logit':
        X_clipped = np.clip(X_raw, clip_lo, clip_hi)
        X = np.log(X_clipped / (1 - X_clipped))
    else:
        X = X_raw

    # Raw linear prediction — used directly, no normalization
    df['raw_value_raw'] = X @ coefs + intercept
    df['raw_value']     = df['raw_value_raw']

    # If the JSON didn't include calibration stats (old training run),
    # fall back to empirical per-card stats. This is a degraded mode —
    # a single-card sample of 80 horses has noisy mean/std and might
    # drift from true training distribution.
    if cv_mean is None or cv_std is None:
        cv_mean = float(df['raw_value'].mean())
        cv_std  = float(df['raw_value'].std())
        print(f"    Empirical fallback: μ=${cv_mean:.3f}, σ=${cv_std:.3f}")

    # Stash calibration on df for generate_bets to pick up. Using
    # df.attrs is cleaner than globals — survives round-trips and
    # doesn't pollute module state.
    df.attrs['raw_cv_mean'] = cv_mean
    df.attrs['raw_cv_std']  = cv_std

    # ALSO write as columns so the calibration persists through
    # df.to_csv(). df.attrs is a runtime convenience but doesn't survive
    # CSV serialization. backtest_walk_forward.py concatenates these
    # per-card CSVs, so without stamped μ/σ the backtest would lose the
    # calibration that was active when each card's predictions were made.
    # Every horse on a card gets the same value — it's a card-level stat.
    df['raw_cv_mean'] = cv_mean
    df['raw_cv_std']  = cv_std

    return df


def _ensure_regression_features(df, date_str, track, skip_tipmeerkat=True):
    """
    Compute human_prob, beyer sims, inv_field_size, ml_x_fieldsize if they
    aren't already present. Idempotent — safe to call more than once.

    This exists so apply_ridge_value_model() can run after either the
    ensemble path (which doesn't build these) or the regression path (which
    does). Either way we get the features the ridge model expects.

    Args:
      skip_tipmeerkat: if True, exclude tm_* expert CSVs from human_pool.
    """
    if 'human_prob' not in df.columns:
        df['human_pool'] = 0.0
        expert_dir = DATA_PATH / 'expert_picks' / date_str / track
        ratings_path = DATA_PATH / 'expert_ratings.json'
        ratings = {}
        if ratings_path.exists():
            with open(ratings_path) as f_r:
                ratings = json.load(f_r)

        if expert_dir.exists():
            n_tm_skipped = 0
            for csv_path in expert_dir.glob('*.csv'):
                expert_name = csv_path.stem
                # Skip tipmeerkat experts when requested. Their Elo
                # ratings still update via expert_elo.py replay; this
                # just excludes them from human_pool for this prediction.
                if skip_tipmeerkat and expert_name.startswith('tm_'):
                    n_tm_skipped += 1
                    continue
                try:
                    edf = pd.read_csv(csv_path)
                except Exception:
                    continue
                if edf.empty:
                    continue

                elo = ratings.get(expert_name, {}).get('rating', 1500)
                elo_mult = max(0.1, (elo - 1350) / 150)

                for _, row in edf.iterrows():
                    race = int(row['race'])
                    hn = int(row['horse_number'])
                    delta = float(row['logit_delta'])
                    pool_amount = (delta / 0.7) * 2.0 * elo_mult
                    mask = (df['race'] == race) & (df['horse_number'] == hn)
                    df.loc[mask, 'human_pool'] += pool_amount

            if skip_tipmeerkat and n_tm_skipped:
                print(f"  Skipped {n_tm_skipped} tipmeerkat (tm_*) expert(s) "
                      f"from human_pool (default; pass --include-tipmeerkat to override)")

        df['human_prob'] = 0.0
        for _, race_df in df.groupby('race'):
            idx = race_df.index
            pool = df.loc[idx, 'human_pool'].values
            shifted = pool - pool.min() + 0.01
            total = shifted.sum()
            if total > 0:
                df.loc[idx, 'human_prob'] = shifted / total

    # Beyer sim columns
    for beyer_col, sim_col in [('claude_beyer', 'claude_beyer_sim'),
                                ('gemini_beyer', 'gemini_beyer_sim'),
                                ('openai_beyer', 'openai_beyer_sim')]:
        if sim_col in df.columns and df[sim_col].notna().any():
            continue
        df[sim_col] = np.nan
        if beyer_col in df.columns:
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                vals = df.loc[idx, beyer_col].values
                valid = ~np.isnan(vals)
                if valid.sum() >= 2:
                    centered = vals[valid] - np.mean(vals[valid])
                    exp_vals = np.exp(centered * 0.1)
                    probs = exp_vals / exp_vals.sum()
                    full = np.full(len(vals), np.nan)
                    full[valid] = probs
                    df.loc[idx, sim_col] = full

    # inv_field_size
    if 'inv_field_size' not in df.columns:
        df['inv_field_size'] = np.nan
        for _, race_df in df.groupby('race'):
            idx = race_df.index
            df.loc[idx, 'inv_field_size'] = 1.0 / len(race_df)

    # ml_x_fieldsize interaction
    if 'ml_x_fieldsize' not in df.columns and 'ml_win_prob_norm' in df.columns:
        clip_lo_fs, clip_hi_fs = 1e-4, 1 - 1e-4
        ml_logit = np.log(
            df['ml_win_prob_norm'].clip(clip_lo_fs, clip_hi_fs) /
            (1 - df['ml_win_prob_norm'].clip(clip_lo_fs, clip_hi_fs))
        )
        df['ml_x_fieldsize'] = ml_logit * df['inv_field_size']

    return df


def build_regression_probs(df, date_str, track, skip_tipmeerkat=True):
    """
    Build probability estimates using the fitted logistic regression model
    (from regression_analysis.py). Uses averaged K-fold coefficients applied
    to ML, Claude/Gemini/OpenAI Prob, Claude/Gemini/OpenAI Beyer, Human Pool,
    plus interaction features.

    Returns df with 'ensemble_prob' column populated.
    """
    model_path = DATA_PATH / 'regression_model.json'
    if not model_path.exists():
        print(f"  No regression_model.json found — run regression_analysis.py first")
        return df

    with open(model_path) as f:
        model = json.load(f)

    feature_cols = model['feature_cols']
    coefs = np.array(model['coefficients'])
    intercept = model['intercept']

    print(f"  Regression model: {model['n_folds']}-fold avg, "
          f"CV ρ={model['cv_spearman']:.4f}")

    df = df.copy()
    df = _ensure_regression_features(df, date_str, track,
                                     skip_tipmeerkat=skip_tipmeerkat)

    col_map = {
        'ml_prob_norm': 'ml_win_prob_norm',
        'claude_prob': 'claude_win_prob',
        'gemini_prob': 'gemini_win_prob',
        'openai_prob': 'openai_win_prob',
        'claude_beyer_sim': 'claude_beyer_sim',
        'gemini_beyer_sim': 'gemini_beyer_sim',
        'openai_beyer_sim': 'openai_beyer_sim',
        'openai_closing': 'openai_closing_prob',
        'claude_closing': 'claude_closing_prob',
        'gemini_closing': 'gemini_closing_prob',
        'human_prob': 'human_prob',
        'inv_field_size': 'inv_field_size',
        'ml_x_fieldsize': 'ml_x_fieldsize',
    }

    feature_is_prob = model.get('feature_is_prob', [True] * len(feature_cols))

    # Normalize probability features per race
    for fc, is_prob in zip(feature_cols, feature_is_prob):
        if not is_prob:
            continue
        df_col = col_map.get(fc, fc)
        if df_col in df.columns:
            for _, race_df in df.groupby('race'):
                idx = race_df.index
                vals = df.loc[idx, df_col].values
                valid = ~np.isnan(vals)
                if valid.any() and np.nansum(vals[valid]) > 0:
                    df.loc[idx, df_col] = np.where(valid, vals / np.nansum(vals), np.nan)

    # Build feature matrix
    X_cols = []
    for fc in feature_cols:
        df_col = col_map.get(fc, fc)
        if df_col in df.columns:
            X_cols.append(df_col)
        else:
            print(f"    WARNING: feature '{fc}' ({df_col}) not found, using 0")
            df[df_col] = 0
            X_cols.append(df_col)

    input_space = model.get('input_space', 'raw')
    clip_lo, clip_hi = 1e-4, 1 - 1e-4
    X_raw = df[X_cols].fillna(0).values

    if input_space == 'mixed':
        X = np.zeros_like(X_raw, dtype=float)
        for i, is_prob in enumerate(feature_is_prob):
            if is_prob:
                clipped = np.clip(X_raw[:, i], clip_lo, clip_hi)
                X[:, i] = np.log(clipped / (1 - clipped))
            else:
                X[:, i] = X_raw[:, i]
    elif input_space == 'logit':
        X_clipped = np.clip(X_raw, clip_lo, clip_hi)
        X = np.log(X_clipped / (1 - X_clipped))
    else:
        X = X_raw

    logits = X @ coefs + intercept
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    df['ensemble_prob'] = raw_probs

    # Normalize per race
    for _, race_df in df.groupby('race'):
        idx = race_df.index
        total = df.loc[idx, 'ensemble_prob'].sum()
        if total > 0:
            df.loc[idx, 'ensemble_prob'] /= total

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate race predictions + bet recommendations")
    parser.add_argument('--date', type=str, required=True, help='Date (e.g. 2026-04-08)')
    parser.add_argument('--track', type=str, required=True, help='Track (e.g. keeneland)')
    parser.add_argument('--sims', type=int, default=100000, help='MC simulations')
    parser.add_argument('--logistic-min', type=float, default=LOGISTIC_MIN,
                        help=f'Logistic EV floor $ per $2 (default {LOGISTIC_MIN}). '
                             f'Horse must clear this on logistic to qualify.')
    parser.add_argument('--logistic-edge', type=float, default=LOGISTIC_EDGE_THRESHOLD,
                        help=f'Logistic EV edge $ per $2 (default {LOGISTIC_EDGE_THRESHOLD}). '
                             f'At least one model must clear its edge.')
    parser.add_argument('--ridge-min-mult', type=float, default=RIDGE_MIN_MULTIPLIER,
                        help=f'Log-ridge qualification floor as multiple of per-race '
                             f'fair share (default {RIDGE_MIN_MULTIPLIER}).')
    parser.add_argument('--ridge-edge-mult', type=float, default=RIDGE_EDGE_MULTIPLIER,
                        help=f'Log-ridge edge threshold as multiple of per-race '
                             f'fair share (default {RIDGE_EDGE_MULTIPLIER}, '
                             f'i.e. 5%% over fair share)')
    parser.add_argument('--raw-edge-k', type=float, default=RAW_EDGE_K,
                        help=f'Raw-ridge edge threshold in std-devs above training '
                             f'mean (default {RAW_EDGE_K}). 1.0 = "~16%% of horses '
                             f'clear bet"; 1.5 stricter, 0.5 looser.')
    parser.add_argument('--raw-min-k', type=float, default=RAW_MIN_K,
                        help=f'Raw-ridge floor threshold in std-devs above training '
                             f'mean (default {RAW_MIN_K}). Negative = "model vetoes '
                             f'only if horse is actively below training mean."')
    parser.add_argument('--takeout', type=float, default=WIN_TAKEOUT,
                        help='Win pool takeout rate (default 0.17)')
    parser.add_argument('--regression', action='store_true',
                        help='Use regression model instead of ensemble for ensemble_prob')
    parser.add_argument('--skip-tipmeerkat', action='store_true',
                        default=None,
                        help='[DEPRECATED — skip is now the default behavior] '
                             'Has no effect; passing this flag is a no-op. '
                             'Pass --include-tipmeerkat to opt back in.')
    parser.add_argument('--include-tipmeerkat', action='store_true',
                        help='Include tipmeerkat (tm_*) experts in the bet '
                             'recommendations and regression human_pool. '
                             'NOT recommended — tipmeerkat data is currently '
                             'bugged and skewing results. Default behavior '
                             'is to exclude tipmeerkat.')
    args = parser.parse_args()

    WIN_TAKEOUT = args.takeout
    LOGISTIC_MIN = args.logistic_min
    LOGISTIC_EDGE_THRESHOLD = args.logistic_edge
    RIDGE_MIN_MULTIPLIER = args.ridge_min_mult
    RIDGE_EDGE_MULTIPLIER = args.ridge_edge_mult
    RAW_EDGE_K = args.raw_edge_k
    RAW_MIN_K  = args.raw_min_k

    # Resolve tipmeerkat behavior. Default is to SKIP tipmeerkat (its data
    # is currently bugged). --include-tipmeerkat opts back in. The legacy
    # --skip-tipmeerkat flag is accepted as a no-op for backward compat.
    skip_tipmeerkat = not args.include_tipmeerkat
    if args.skip_tipmeerkat:
        print("  Note: --skip-tipmeerkat is now the default and is no longer "
              "needed. Future versions may remove this flag.")
    if not skip_tipmeerkat:
        print("!" * 78)
        print("!!  --include-tipmeerkat active: tipmeerkat (tm_*) experts WILL be used.")
        print("!!  This data is currently bugged and may skew results.")
        print("!!  Recommend running without --include-tipmeerkat (the default).")
        print("!" * 78)

    run_predictions(args.date, args.track, num_sims=args.sims,
                    use_regression=args.regression,
                    skip_tipmeerkat=skip_tipmeerkat)