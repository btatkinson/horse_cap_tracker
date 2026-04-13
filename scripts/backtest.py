"""
backtest_ai.py

Backtest AI handicapping estimates against actual results.
Optimizes t-dist sim params and source blend weights.
Exports a fitted equation for forward-looking predictions.

Data layout (all relative to DATA_PATH from .env):
    {DATA_PATH}/ai_picks/{date}/claude.csv
    {DATA_PATH}/ai_picks/{date}/gemini.csv
    {DATA_PATH}/entries/*.csv           — historical results (the truth)

Results CSV schema:
    running, horse, horse_name, sire, trainer, jockey, ml_odds,
    finish, win_paid, place_paid, show_paid, track, date, race

AI pick CSV schema (flexible — missing cols OK):
    race, horse_number, {source}_beyer, {source}_win_prob, track

Usage:
    python backtest_ai.py
    python backtest_ai.py --optimize
    python backtest_ai.py --optimize --sims 100000
"""

import os, json, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
PICKS_DIR = DATA_PATH / 'ai_picks'
ENTRIES_DIR = DATA_PATH / 'entries'

GROUP_COLS = ['date', 'track', 'race']

# ── Source roster ────────────────────────────────────────────────────
# AI providers that supply beyer and win_prob estimates.
AI_PROVIDERS = ['claude', 'gemini']

# Ensemble sources — each gets an independent weight.
# Beyer sources use t-dist sim → ML-regularized prob.
# Prob sources use the LLM's direct win probability estimate.
# ML is NOT an ensemble member (only regularizes beyer sims).
SOURCE_ROSTER = [
    {'name': 'claude_beyer', 'weight': 0.25},
    {'name': 'claude_prob',  'weight': 0.25},
    {'name': 'gemini_beyer', 'weight': 0.25},
    {'name': 'gemini_prob',  'weight': 0.25},
]


# =============================================================================
# 1. LOADING
# =============================================================================

def _normalize_date(d):
    """Convert various date formats to YYYY-MM-DD."""
    d = str(d).strip()
    # Already ISO? (YYYY-MM-DD)
    if len(d) == 10 and d[4] == '-' and d[7] == '-':
        return d
    # "4_3_26" or "4-3-26" or "04-03-2026" (M-D-Y)
    for sep in ['_', '-', '/']:
        parts = d.split(sep)
        if len(parts) == 3:
            m, day, yr = parts
            try:
                yr = int(yr)
                if yr < 100:
                    yr += 2000
                return f"{yr}-{int(m):02d}-{int(day):02d}"
            except ValueError:
                pass
    return d


# Canonical track names — maps variants to the short name used in ai_picks
_TRACK_ALIASES = {
    'gulfstream-park':  'gulfstream',
    'gulfstream park':  'gulfstream',
    'churchill-downs':  'churchill',
    'churchill downs':  'churchill',
    'santa-anita':      'santa-anita',
    'santa anita':      'santa-anita',
    'santa anita park': 'santa-anita',
    'del-mar':          'del-mar',
    'del mar':          'del-mar',
    'fair-grounds':     'fair-grounds',
    'fair grounds':     'fair-grounds',
    'tampa-bay':        'tampa',
    'tampa bay':        'tampa',
    'tampa bay downs':  'tampa',
    'belmont-park':     'belmont',
    'belmont park':     'belmont',
    'belmont at aqueduct': 'aqueduct',
    'laurel-park':      'laurel',
    'laurel park':      'laurel',
    'oaklawn-park':     'oaklawn',
    'oaklawn park':     'oaklawn',
    'monmouth-park':    'monmouth',
    'monmouth park':    'monmouth',
    'parx-racing':      'parx',
    'parx racing':      'parx',
    'turfway-park':     'turfway',
    'turfway park':     'turfway',
    'ellis-park':       'ellis',
    'ellis park':       'ellis',
}


def _normalize_track(t):
    """Normalize track name to canonical short form."""
    if pd.isna(t):
        return t
    t = str(t).strip().lower()
    return _TRACK_ALIASES.get(t, t)


def load_entries():
    """
    Load historical results. These are the ground truth.
    Only 'running == 1' horses are kept (scratches removed).
    """
    if not ENTRIES_DIR.exists():
        return pd.DataFrame()

    dfs = []
    for f in ENTRIES_DIR.glob('*.csv'):
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Remove scratches
    if 'running' in df.columns:
        df = df[df['running'] == 1].copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    if 'track' in df.columns:
        df['track'] = df['track'].apply(_normalize_track)

    if 'finish' in df.columns:
        df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
        df['won'] = (df['finish'] == 1).astype(int)

    if 'horse' in df.columns and 'horse_number' not in df.columns:
        df['horse_number'] = pd.to_numeric(df['horse'], errors='coerce').astype('Int64')

    # Morning line → probability
    if 'ml_odds' in df.columns:
        df['ml_win_prob'] = df['ml_odds'].apply(_parse_ml_odds)

    # Clean payout columns to numeric (strip $ signs, etc.)
    for col in ['win_paid', 'place_paid', 'show_paid']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True),
                errors='coerce'
            )

    return df.reset_index(drop=True)


def _parse_ml_odds(odds_str):
    """Parse fractional/decimal ML odds to implied probability."""
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '':
            return np.nan
        s = str(odds_str).strip()
        if '/' in s:
            num, den = map(float, s.split('/'))
            return den / (num + den)
        if s.upper() in ('EVN', 'EVEN', 'EV'):
            return 0.5
        v = float(s)
        return 1 / (v + 1) if v > 0 else np.nan
    except (ValueError, ZeroDivisionError):
        return np.nan


def _find_source_csvs(date_dir, source_name):
    """
    Find a source CSV under a date folder.  Handles both layouts:
      ai_picks/{date}/claude.csv              (flat)
      ai_picks/{date}/{track}/claude.csv      (track subfolder)
    Returns list of (path, track_name_or_None).
    """
    date_dir = Path(date_dir)
    found = []

    # Flat: {date}/claude.csv
    flat = date_dir / f'{source_name}.csv'
    if flat.exists():
        found.append((flat, None))

    # Track subfolders: {date}/{track}/claude.csv
    for sub in date_dir.iterdir():
        if sub.is_dir():
            nested = sub / f'{source_name}.csv'
            if nested.exists():
                found.append((nested, sub.name))

    return found


def _read_source_csv(path, source_name, track_override=None):
    """Read one source CSV, normalize columns, return DataFrame."""
    df = pd.read_csv(path)

    # Rename columns to {source}_beyer, {source}_win_prob
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if 'beyer' in cl:
            col_map[c] = f'{source_name}_beyer'
        elif 'win_prob' in cl or 'prob' in cl:
            col_map[c] = f'{source_name}_win_prob'
    df = df.rename(columns=col_map)

    df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').astype('Int64')

    keep = ['race', 'horse_number']
    for col in [f'{source_name}_beyer', f'{source_name}_win_prob']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            keep.append(col)

    # Track: use column if present, else folder name, else None
    if 'track' in df.columns:
        keep.append('track')
    elif track_override:
        df['track'] = track_override
        keep.append('track')

    return df[keep]


def load_ai_picks(date_str):
    """
    Load all AI source CSVs for one date.
    Searches both flat and track-subfolder layouts.
    Returns merged DataFrame with {provider}_beyer, {provider}_win_prob columns.
    """
    date_dir = PICKS_DIR / date_str
    if not date_dir.exists():
        raise FileNotFoundError(f"No folder: {date_dir}")

    source_frames = {}  # {provider_name: [df, ...]}

    for provider in AI_PROVIDERS:
        hits = _find_source_csvs(date_dir, provider)
        for path, track_name in hits:
            df = _read_source_csv(path, provider, track_override=track_name)
            source_frames.setdefault(provider, []).append(df)

    if not source_frames:
        raise FileNotFoundError(f"No AI pick files in {date_dir}")

    # Concat per-source (multiple tracks), then merge across sources
    source_dfs = {}
    for name, dfs in source_frames.items():
        source_dfs[name] = pd.concat(dfs, ignore_index=True)

    names = list(source_dfs.keys())
    merged = source_dfs[names[0]]
    for name in names[1:]:
        on = ['race', 'horse_number']
        if 'track' in merged.columns and 'track' in source_dfs[name].columns:
            on.append('track')
        merged = merged.merge(source_dfs[name], on=on, how='outer')

    merged['date'] = date_str
    return merged


def load_all_picks():
    """Load and concat AI picks from ALL date folders under PICKS_DIR."""
    if not PICKS_DIR.exists():
        return pd.DataFrame()

    dfs = []
    for d in sorted(PICKS_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Skip non-date folders (like if backtest_results.csv is here)
        try:
            dfs.append(load_ai_picks(d.name))
        except FileNotFoundError:
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def merge_picks_and_results(picks, entries):
    """
    Inner join picks with entries on (date, race, horse_number).
    This removes scratches — only horses in results survive.
    """
    picks = picks.copy()
    entries = entries.copy()

    # Normalize dates
    picks['date'] = picks['date'].apply(_normalize_date)
    entries['date'] = entries['date'].apply(_normalize_date)

    # Normalize track names
    if 'track' in picks.columns:
        picks['track'] = picks['track'].apply(_normalize_track)
    if 'track' in entries.columns:
        entries['track'] = entries['track'].apply(_normalize_track)

    on = ['date', 'race', 'horse_number']
    if 'track' in picks.columns and 'track' in entries.columns:
        on.append('track')

    df = picks.merge(entries, on=on, how='inner')
    return df


# =============================================================================
# 2. MORNING LINE NORMALIZATION
# =============================================================================

def normalize_ml_probs(df):
    """Normalize raw ML probs to sum to 1 within each race."""
    df = df.copy()
    if 'ml_win_prob' not in df.columns:
        return df

    df['ml_win_prob_norm'] = np.nan
    gcols = [c for c in GROUP_COLS if c in df.columns]

    for _, race_df in df.groupby(gcols):
        idx = race_df.index
        raw = race_df['ml_win_prob'].values
        valid = ~np.isnan(raw)
        if valid.any():
            probs = raw[valid]
            probs = probs / probs.sum()
            df.loc[idx[valid], 'ml_win_prob_norm'] = probs

    return df


# =============================================================================
# 3. IMPUTATION
# =============================================================================

def _fit_prob_to_beyer(df):
    """Fit logit(prob) ~ beyer from all paired observations. Returns (slope, intercept)."""
    bv, pv = [], []
    for provider in AI_PROVIDERS:
        bc, pc = f'{provider}_beyer', f'{provider}_win_prob'
        if bc in df.columns and pc in df.columns:
            mask = df[bc].notna() & df[pc].notna() & (df[pc] > 0) & (df[pc] < 1)
            bv.extend(df.loc[mask, bc].values)
            pv.extend(df.loc[mask, pc].values)
    if len(bv) < 5:
        return None
    bv, pv = np.array(bv), np.array(pv)
    logits = np.log(pv / (1 - pv))
    slope, intercept = np.polyfit(bv, logits, 1)
    return slope, intercept


def impute_missing(df, std_dev, degrees_of_freedom, num_sims=20000):
    """
    Fill gaps per AI provider:
      1. beyer present, prob missing → t-dist sim
      2. prob present, beyer missing → logit inversion
      3. both missing → copy from other provider average
    """
    df = df.copy()
    fit = _fit_prob_to_beyer(df)

    for provider in AI_PROVIDERS:
        bc, pc = f'{provider}_beyer', f'{provider}_win_prob'
        if bc not in df.columns:
            df[bc] = np.nan
        if pc not in df.columns:
            df[pc] = np.nan

        # Case 1: beyer → prob
        mask = df[bc].notna() & df[pc].isna()
        if mask.any():
            simmed = _sim_beyer_to_prob(df.loc[mask], bc, std_dev, degrees_of_freedom, num_sims)
            df.loc[mask, pc] = simmed.values

        # Case 2: prob → beyer
        mask = df[pc].notna() & df[bc].isna()
        if mask.any() and fit is not None:
            slope, intercept = fit
            probs = df.loc[mask, pc].clip(1e-4, 1 - 1e-4)
            df.loc[mask, bc] = (np.log(probs / (1 - probs)) - intercept) / slope

    # Case 3: still missing → average from other provider
    for provider in AI_PROVIDERS:
        others = [p for p in AI_PROVIDERS if p != provider]
        for metric in ['beyer', 'win_prob']:
            col = f'{provider}_{metric}'
            other_cols = [f'{o}_{metric}' for o in others if f'{o}_{metric}' in df.columns]
            if other_cols:
                still_na = df[col].isna()
                avail = df.loc[still_na, other_cols].mean(axis=1)
                df.loc[still_na, col] = avail

    return df


def _sim_beyer_to_prob(df, beyer_col, std_dev, degrees_of_freedom, num_sims, seed=42):
    """T-dist MC sim per race group. Returns Series of win probs."""
    result = pd.Series(np.nan, index=df.index)
    rng = np.random.RandomState(seed)
    t_dist = stats.t(df=degrees_of_freedom)
    gcols = [c for c in GROUP_COLS if c in df.columns] or ['race']

    for _, race_df in df.groupby(gcols):
        idx = race_df.index
        n = len(idx)
        vals = race_df[beyer_col].values
        if n <= 1:
            result.loc[idx] = 1.0
            continue
        mean_b = np.nanmean(vals)
        vals = np.where(np.isnan(vals), mean_b, vals)
        noise = t_dist.rvs(size=(n, num_sims), random_state=rng) * std_dev
        winners = np.argmax(vals.reshape(n, 1) + noise, axis=0)
        counts = np.bincount(winners, minlength=n)
        probs = np.maximum(counts / num_sims, 1e-6)
        result.loc[idx] = probs / probs.sum()

    return result


# =============================================================================
# 4. COMPOSITE PROBABILITIES
# =============================================================================

def build_composite(df, std_dev, degrees_of_freedom, num_sims=50000,
                    weights=None, ml_regularization=0.25):
    """
    Build 4 independent probability sources, then blend into ensemble.

    Sources:
      - claude_beyer: t-dist MC sim from Claude beyers, ML-regularized
      - claude_prob:  Claude's direct win probability estimate
      - gemini_beyer: t-dist MC sim from Gemini beyers, ML-regularized
      - gemini_prob:  Gemini's direct win probability estimate

    Each source is an independent ensemble member with its own weight.
    Claude beyer backs up claude_prob if direct prob is missing (and vice versa).

    ML is NOT an ensemble member — only regularizes beyer sims.
    Ensemble = logit-space weighted average of available sources.
    """
    df = df.copy()
    gcols = [c for c in GROUP_COLS if c in df.columns]

    if weights is None:
        weights = {s['name']: s['weight'] for s in SOURCE_ROSTER}

    has_ml = 'ml_win_prob_norm' in df.columns and df['ml_win_prob_norm'].notna().any()

    available_sources = []

    for provider in AI_PROVIDERS:
        bc = f'{provider}_beyer'
        pc = f'{provider}_win_prob'
        sim_col = f'{provider}_sim_prob'
        reg_col = f'{provider}_reg_sim'
        beyer_source = f'{provider}_beyer'   # ensemble source name
        prob_source = f'{provider}_prob'      # ensemble source name

        # ── Beyer path: sim → regularize with ML ──
        if bc in df.columns and df[bc].notna().any():
            df[sim_col] = _sim_beyer_to_prob_full(df, bc, std_dev, degrees_of_freedom, num_sims)
        else:
            df[sim_col] = np.nan

        if has_ml and df[sim_col].notna().any():
            ml_fill = df['ml_win_prob_norm'].fillna(df[sim_col])
            df[reg_col] = (1 - ml_regularization) * df[sim_col] + ml_regularization * ml_fill
        else:
            df[reg_col] = df[sim_col]

        # Beyer source prob = regularized sim
        # Fallback: if beyer missing, use direct prob
        beyer_col = f'{beyer_source}_src'
        df[beyer_col] = df[reg_col]
        beyer_missing = df[beyer_col].isna()
        if pc in df.columns and beyer_missing.any():
            df.loc[beyer_missing, beyer_col] = df.loc[beyer_missing, pc]

        if df[beyer_col].notna().any():
            available_sources.append(beyer_source)

        # ── Prob path: direct LLM probability ──
        # Fallback: if direct prob missing, use regularized sim
        prob_col = f'{prob_source}_src'
        if pc in df.columns:
            df[prob_col] = df[pc]
        else:
            df[prob_col] = np.nan
        prob_missing = df[prob_col].isna()
        if prob_missing.any() and df[reg_col].notna().any():
            df.loc[prob_missing, prob_col] = df.loc[prob_missing, reg_col]

        if df[prob_col].notna().any():
            available_sources.append(prob_source)

    # --- Ensemble blend in logit space ---
    if not available_sources:
        df['ensemble_prob'] = np.nan
        return df

    w = {s: weights.get(s, 0) for s in available_sources}
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
    for _, race_df in df.groupby(gcols):
        idx = race_df.index
        total = df.loc[idx, 'ensemble_prob'].sum()
        if total > 0:
            df.loc[idx, 'ensemble_prob'] /= total

    return df


def _sim_beyer_to_prob_full(df, beyer_col, std_dev, degrees_of_freedom,
                            num_sims, seed=42):
    """Sim for all rows, grouping by race. Returns Series."""
    result = pd.Series(np.nan, index=df.index)
    rng = np.random.RandomState(seed)
    t_dist = stats.t(df=degrees_of_freedom)
    gcols = [c for c in GROUP_COLS if c in df.columns] or ['race']

    for _, race_df in df.groupby(gcols):
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


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate(df, prob_col='ensemble_prob'):
    """
    Primary: mean Spearman ρ (prob rank vs finish rank per race).
    Secondary: log loss, top-pick win rate, calibration.
    """
    metrics = {}
    gcols = [c for c in GROUP_COLS if c in df.columns]

    # Spearman per race
    rhos = []
    for _, race_df in df.groupby(gcols):
        valid = race_df.dropna(subset=[prob_col, 'finish'])
        if len(valid) < 3:
            continue
        rho, _ = spearmanr(valid[prob_col], -valid['finish'])
        rhos.append(rho)

    metrics['spearman_mean'] = np.mean(rhos) if rhos else np.nan
    metrics['spearman_median'] = np.median(rhos) if rhos else np.nan
    metrics['n_races'] = len(rhos)

    # Log loss
    if 'won' in df.columns:
        valid = df.dropna(subset=[prob_col])
        if len(valid) > 0:
            probs = np.clip(valid[prob_col].values, 1e-6, 1 - 1e-6)
            from sklearn.metrics import log_loss
            metrics['log_loss'] = log_loss(valid['won'].values, probs)

    # Top pick win rate
    if 'won' in df.columns:
        hits = []
        for _, race_df in df.groupby(gcols):
            valid = race_df.dropna(subset=[prob_col])
            if len(valid) == 0:
                continue
            hits.append(valid.loc[valid[prob_col].idxmax(), 'won'])
        if hits:
            metrics['top_pick_win_rate'] = np.mean(hits)

    # Calibration
    if 'won' in df.columns:
        valid = df.dropna(subset=[prob_col]).copy()
        if len(valid) > 0:
            valid['bucket'] = pd.cut(
                valid[prob_col],
                bins=[0, 0.05, 0.10, 0.20, 0.35, 1.0],
                labels=['<5%', '5-10%', '10-20%', '20-35%', '>35%']
            )
            metrics['calibration'] = valid.groupby('bucket', observed=True).agg(
                n=('won', 'count'),
                actual=('won', 'mean'),
                predicted=(prob_col, 'mean')
            )

    return metrics


# =============================================================================
# 5b. HISTORICAL EXPERT APPLICATION (for backtest comparison)
# =============================================================================

def apply_historical_experts(df, base_col, out_col):
    """
    Apply historical expert logit deltas to a base probability column.
    Loads expert_picks CSVs for each date/track in df, applies Elo-weighted
    logit adjustments, renormalizes per race.

    Uses CURRENT ratings from expert_ratings.json (already Elo-optimized).
    """
    gcols = [c for c in GROUP_COLS if c in df.columns]
    ratings_path = DATA_PATH / 'expert_ratings.json'

    if not ratings_path.exists():
        print(f"  No expert_ratings.json found")
        df[out_col] = np.nan
        return df

    with open(ratings_path) as f:
        all_ratings = json.load(f)

    BASELINE = 1500
    SCALE = 1000
    clip = 1e-4

    df = df.copy()
    df[out_col] = df[base_col].copy()

    # Group by date/track to find expert picks
    if 'date' not in df.columns or 'track' not in df.columns:
        print(f"  Need date+track columns for expert application")
        return df

    total_experts = 0
    total_adjusted = 0

    # Debug: show what we're looking for
    date_tracks = df.groupby(['date', 'track']).size().reset_index()
    expert_picks_dir = DATA_PATH / 'expert_picks'
    for _, row in date_tracks.iterrows():
        d, t = row['date'], row['track']
        edir = expert_picks_dir / d / t
        exists = edir.exists()
        n_csv = len(list(edir.glob('*.csv'))) if exists else 0
        if not exists:
            # Try listing what IS in expert_picks for this date
            date_dir = expert_picks_dir / d
            if date_dir.exists():
                subdirs = [s.name for s in date_dir.iterdir() if s.is_dir()]
                print(f"  {d}/{t}: NOT FOUND (available: {subdirs})")
            else:
                print(f"  {d}/{t}: date dir not found")
        else:
            print(f"  {d}/{t}: {n_csv} CSVs")

    for (date_str, track), group_df in df.groupby(['date', 'track']):
        expert_dir = DATA_PATH / 'expert_picks' / date_str / track
        if not expert_dir.exists():
            continue

        csv_files = sorted(expert_dir.glob('*.csv'))
        if not csv_files:
            continue

        # Get logits for this group
        idx = group_df.index
        probs = df.loc[idx, out_col].clip(clip, 1 - clip)
        logits = np.log(probs / (1 - probs)).values

        for csv_path in csv_files:
            expert_name = csv_path.stem
            expert_df = pd.read_csv(csv_path)
            if expert_df.empty:
                continue

            # Get weight from rating
            if expert_name in all_ratings:
                rating = all_ratings[expert_name]['rating']
            else:
                rating = BASELINE

            weight = (rating - BASELINE) / SCALE
            if abs(weight) < 0.0001:
                continue

            # Apply deltas
            for _, row in expert_df.iterrows():
                race = int(row['race'])
                hn = int(row['horse_number'])
                delta = float(row['logit_delta'])

                mask = (group_df['race'] == race) & (group_df['horse_number'] == hn)
                local_idx = group_df.index[mask]
                if len(local_idx) > 0:
                    pos = np.where(idx == local_idx[0])[0]
                    if len(pos) > 0:
                        logits[pos[0]] += weight * delta
                        total_adjusted += 1

            total_experts += 1

        # Convert back and normalize per race
        adjusted_probs = 1.0 / (1.0 + np.exp(-logits))
        df.loc[idx, out_col] = adjusted_probs

        for race_num, race_df in group_df.groupby('race'):
            race_idx = race_df.index
            total = df.loc[race_idx, out_col].sum()
            if total > 0:
                df.loc[race_idx, out_col] /= total

    return df, total_experts, total_adjusted


def print_comparison_report(df):
    """
    Print side-by-side comparison of different model configurations:
      1. ML only (baseline)
      2. ML + experts (human-only)
      3. AI only (ensemble, no experts)
      4. AI + experts (full model)
    """
    gcols = [c for c in GROUP_COLS if c in df.columns]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    configs = []

    # 1. ML baseline
    if 'ml_win_prob_norm' in df.columns and df['ml_win_prob_norm'].notna().any():
        configs.append(('ML only', 'ml_win_prob_norm'))

    # 2. ML + experts
    if 'ml_plus_experts' in df.columns and df['ml_plus_experts'].notna().any():
        configs.append(('ML + experts', 'ml_plus_experts'))

    # 3. AI only (ensemble without expert adjustments)
    if 'ensemble_prob' in df.columns and df['ensemble_prob'].notna().any():
        configs.append(('AI only', 'ensemble_prob'))

    # 4. AI + experts
    if 'ai_plus_experts' in df.columns and df['ai_plus_experts'].notna().any():
        configs.append(('AI + experts', 'ai_plus_experts'))

    if not configs:
        print("  No probability columns available")
        return

    header = f"  {'Config':<20s} {'Spearman':>9s} {'Median':>8s} {'LogLoss':>8s} {'TopPick':>8s} {'Races':>6s}"
    print(header)
    print(f"  {'-'*20} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for label, col in configs:
        m = evaluate(df, prob_col=col)
        rho = m.get('spearman_mean', np.nan)
        med = m.get('spearman_median', np.nan)
        ll = m.get('log_loss', np.nan)
        tp = m.get('top_pick_win_rate', np.nan)
        n = m.get('n_races', 0)

        rho_s = f"{rho:.4f}" if not np.isnan(rho) else "n/a"
        med_s = f"{med:.4f}" if not np.isnan(med) else "n/a"
        ll_s = f"{ll:.4f}" if not np.isnan(ll) else "n/a"
        tp_s = f"{tp:.1%}" if not np.isnan(tp) else "n/a"

        print(f"  {label:<20s} {rho_s:>9s} {med_s:>8s} {ll_s:>8s} {tp_s:>8s} {n:>6d}")

    # Show deltas vs ML
    if len(configs) > 1 and 'ml_win_prob_norm' in df.columns:
        ml_m = evaluate(df, prob_col='ml_win_prob_norm')
        ml_rho = ml_m.get('spearman_mean', np.nan)
        print(f"\n  Edge over ML (Spearman):")
        for label, col in configs:
            if col == 'ml_win_prob_norm':
                continue
            m = evaluate(df, prob_col=col)
            rho = m.get('spearman_mean', np.nan)
            delta = rho - ml_rho if not np.isnan(rho) and not np.isnan(ml_rho) else np.nan
            delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "n/a"
            print(f"    {label:<20s} {delta_s}")


# =============================================================================
# 6. OPTIMIZATION
# =============================================================================

def optimize_params(df, initial_std=5.0, initial_df=1.0, initial_ml_reg=0.25,
                    num_sims=10000):
    """Optimize (std_dev, degrees_of_freedom, ml_regularization) to maximize mean Spearman."""
    if 'finish' not in df.columns:
        raise ValueError("Need 'finish' column")

    def objective(params):
        sd, dof, ml_reg = params
        if sd <= 0.5 or dof <= 0.1 or ml_reg < 0 or ml_reg > 0.95:
            return 1.0
        result = build_composite(df, std_dev=sd, degrees_of_freedom=dof,
                                 num_sims=num_sims, ml_regularization=ml_reg)
        m = evaluate(result)
        rho = m.get('spearman_mean', 0)
        return -rho if not np.isnan(rho) else 1.0

    res = minimize(objective, x0=[initial_std, initial_df, initial_ml_reg],
                   method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 0.05, 'fatol': 0.001, 'disp': True})

    opt_std, opt_df, opt_ml_reg = res.x
    opt_ml_reg = np.clip(opt_ml_reg, 0, 0.95)
    print(f"  Optimal std_dev:         {opt_std:.4f}")
    print(f"  Optimal df:              {opt_df:.4f}")
    print(f"  Optimal ml_regularization: {opt_ml_reg:.4f}")
    print(f"  Best Spearman:           {-res.fun:.4f}")
    return opt_std, opt_df, opt_ml_reg


def optimize_weights(df, std_dev, degrees_of_freedom, num_sims=10000):
    """Optimize source weights via softmax parameterization."""
    available = [s['name'] for s in SOURCE_ROSTER
                 if f"{s['name']}_src" in df.columns and df[f"{s['name']}_src"].notna().any()]

    if len(available) < 2:
        print(f"  Only {len(available)} source(s), skipping weight optimization.")
        return {s['name']: s['weight'] for s in SOURCE_ROSTER}

    def objective(raw_w):
        exp_w = np.exp(raw_w)
        w = dict(zip(available, exp_w / exp_w.sum()))
        # Fill defaults for unavailable
        for s in SOURCE_ROSTER:
            if s['name'] not in w:
                w[s['name']] = 0
        result = build_composite(df, std_dev=std_dev, degrees_of_freedom=degrees_of_freedom,
                                 num_sims=num_sims, weights=w)
        m = evaluate(result)
        rho = m.get('spearman_mean', 0)
        return -rho if not np.isnan(rho) else 1.0

    res = minimize(objective, np.zeros(len(available)),
                   method='Nelder-Mead', options={'maxiter': 200, 'disp': True})

    exp_w = np.exp(res.x)
    opt = dict(zip(available, exp_w / exp_w.sum()))
    print("\n  Optimal weights:")
    for s, w in opt.items():
        print(f"    {s:>15s}: {w:.4f}")
    return opt


# =============================================================================
# 7. EXPORT FITTED EQUATION
# =============================================================================

def export_equation(std_dev, degrees_of_freedom, weights, ml_regularization=0.25,
                    out_path=None):
    """
    Save the fitted parameters so forward-looking scripts can import them.

    The forward equation is:
      1. For each AI provider: sim beyer→prob via t(df, std_dev)
      2. Regularize sim: (1 - ml_reg) * sim + ml_reg * ML
      3. Four independent sources: claude_beyer, claude_prob, gemini_beyer, gemini_prob
      4. Ensemble = logit-space weighted average of all 4 sources

    Saved as JSON: {std_dev, degrees_of_freedom, ml_regularization, weights}
    """
    if out_path is None:
        out_path = DATA_PATH / 'fitted_params.json'

    params = {
        'std_dev': float(std_dev),
        'degrees_of_freedom': float(degrees_of_freedom),
        'ml_regularization': float(ml_regularization),
        'weights': {k: float(v) for k, v in weights.items()},
        'source_roster': [s['name'] for s in SOURCE_ROSTER],
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"\n  Fitted params saved to {out_path}")
    return params


# =============================================================================
# 8. AI-vs-ML DISAGREEMENT ANALYSIS
# =============================================================================

def _safe_logit(p, clip=1e-4):
    """Logit with clipping to avoid inf."""
    p = np.clip(p, clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def _print_disagreement_report(df):
    """
    Classify each horse by AI-vs-ML disagreement in logit space.

    Logit delta = mean(logit(claude_prob), logit(gemini_prob)) - logit(ml_prob)

    Bins:
        AI very high on:  delta > 1.0   (~2.7x more likely than ML thinks)
        AI high on:       0.4 < delta <= 1.0
        Agreement:       -0.4 <= delta <= 0.4
        AI low on:       -1.0 <= delta < -0.4
        AI very low on:   delta < -1.0

    Reports: n, actual win%, AI prob, ML prob, ROI at ML odds.
    """
    # Need at least one AI source + ML
    ai_prob_cols = [f'{s["name"]}_src' for s in SOURCE_ROSTER
                    if f'{s["name"]}_src' in df.columns]
    ml_col = 'ml_win_prob_norm'

    if not ai_prob_cols or ml_col not in df.columns:
        return
    if 'won' not in df.columns:
        return

    valid = df.dropna(subset=ai_prob_cols + [ml_col]).copy()
    if len(valid) < 10:
        return

    # Compute AI average in logit space
    ai_logits = np.column_stack([_safe_logit(valid[c].values) for c in ai_prob_cols])
    ai_mean_logit = ai_logits.mean(axis=1)
    ml_logit = _safe_logit(valid[ml_col].values)

    valid['logit_delta'] = ai_mean_logit - ml_logit

    # Bins
    bins = [
        ('AI very high on', valid['logit_delta'] > 1.0),
        ('AI high on',      (valid['logit_delta'] > 0.4) & (valid['logit_delta'] <= 1.0)),
        ('Agreement',       (valid['logit_delta'] >= -0.4) & (valid['logit_delta'] <= 0.4)),
        ('AI low on',       (valid['logit_delta'] >= -1.0) & (valid['logit_delta'] < -0.4)),
        ('AI very low on',  valid['logit_delta'] < -1.0),
    ]

    print("\n  AI-vs-ML Disagreement (logit space):")
    print(f"    {'bin':<18s} {'n':>5s} {'win%':>7s} {'AI_prob':>8s} {'ML_prob':>8s} {'ROI':>8s}")
    print(f"    {'-'*18} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

    for label, mask in bins:
        subset = valid[mask]
        n = len(subset)
        if n == 0:
            print(f"    {label:<18s} {0:>5d}       -        -        -        -")
            continue

        actual_wr = subset['won'].mean()

        # Average AI prob and ML prob
        ai_prob_avg = np.mean([subset[c].mean() for c in ai_prob_cols])
        ml_prob_avg = subset[ml_col].mean()

        # ROI: bet $1 on every horse in this bin at ML odds
        # win_paid is for a $2 bet, so payout_per_dollar = win_paid / 2
        if 'win_paid' in subset.columns:
            winners = subset[subset['won'] == 1]
            total_bet = n
            if len(winners) > 0:
                wp = pd.to_numeric(
                    winners['win_paid'].astype(str).str.replace(r'[^0-9.]', '', regex=True),
                    errors='coerce'
                )
                total_return = wp.sum() / 2.0
            else:
                total_return = 0.0
            roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0.0
            roi_str = f"{roi:+.1%}"
        else:
            roi_str = "   n/a"

        print(f"    {label:<18s} {n:>5d} {actual_wr:>7.1%} {ai_prob_avg:>8.3f} {ml_prob_avg:>8.3f} {roi_str:>8s}")

    # Summary stats
    print(f"\n    Logit delta stats: mean={valid['logit_delta'].mean():+.3f}  "
          f"std={valid['logit_delta'].std():.3f}  "
          f"range=[{valid['logit_delta'].min():+.2f}, {valid['logit_delta'].max():+.2f}]")


# =============================================================================
# 8b. CONSENSUS ANALYSIS
# =============================================================================

def _print_consensus_report(df):
    """
    Analyze performance when ML, AI, and experts agree on the top pick.

    For each race, identify:
      - ML top pick (highest ml_win_prob_norm)
      - AI top pick (highest ensemble_prob)
      - Expert top pick (highest expert-adjusted prob, if available)

    Then report win rates and ROI by agreement level.
    """
    gcols = [c for c in GROUP_COLS if c in df.columns]

    if 'won' not in df.columns or 'ml_win_prob_norm' not in df.columns:
        return
    if 'ensemble_prob' not in df.columns:
        return

    # Determine which expert-adjusted column to use
    expert_col = None
    for col in ['ai_plus_experts', 'ml_plus_experts']:
        if col in df.columns and df[col].notna().any():
            expert_col = col
            break

    print("\n  Consensus Analysis:")
    print(f"  (When do ML, AI, and experts agree on the top pick?)")

    race_records = []

    for _, race_df in df.groupby(gcols):
        valid = race_df.dropna(subset=['ml_win_prob_norm', 'ensemble_prob', 'finish'])
        if len(valid) < 2:
            continue

        ml_top = int(valid.loc[valid['ml_win_prob_norm'].idxmax(), 'horse_number'])
        ai_top = int(valid.loc[valid['ensemble_prob'].idxmax(), 'horse_number'])

        expert_top = None
        if expert_col and expert_col in valid.columns and valid[expert_col].notna().any():
            expert_top = int(valid.loc[valid[expert_col].idxmax(), 'horse_number'])

        # Who won?
        winner_row = valid[valid['won'] == 1]
        if winner_row.empty:
            continue
        winner = int(winner_row.iloc[0]['horse_number'])

        # Winner's ML odds for ROI calc
        winner_ml = valid.loc[valid['won'] == 1, 'ml_win_prob_norm'].values[0]
        win_paid = winner_row.iloc[0].get('win_paid', np.nan)

        # Consensus level
        if expert_top is not None:
            all_agree = (ml_top == ai_top == expert_top)
            ml_ai_agree = (ml_top == ai_top)
            consensus_pick = ml_top if all_agree else None
            two_of_three = (
                (ml_top == ai_top) or
                (ml_top == expert_top) or
                (ai_top == expert_top)
            )
        else:
            all_agree = False
            ml_ai_agree = (ml_top == ai_top)
            consensus_pick = ml_top if ml_ai_agree else None
            two_of_three = ml_ai_agree

        # Determine the consensus horse (most-agreed-upon)
        if all_agree:
            level = 'all_agree'
            pick = ml_top
        elif two_of_three:
            level = 'two_agree'
            # Pick the horse that 2+ agree on
            from collections import Counter
            picks = [ml_top, ai_top]
            if expert_top is not None:
                picks.append(expert_top)
            pick = Counter(picks).most_common(1)[0][0]
        else:
            level = 'no_agree'
            pick = ml_top  # default to ML

        won = 1 if pick == winner else 0

        # ML odds of the consensus pick
        pick_row = valid[valid['horse_number'] == pick]
        pick_ml_prob = pick_row['ml_win_prob_norm'].values[0] if len(pick_row) > 0 else np.nan
        pick_ml_decimal = (1 / pick_ml_prob - 1) if pick_ml_prob > 0 else np.nan

        race_records.append({
            'level': level,
            'pick': pick,
            'won': won,
            'ml_prob': pick_ml_prob,
            'ml_decimal': pick_ml_decimal,
            'win_paid': win_paid if won else 0,
        })

    if not race_records:
        print("    No races with sufficient data")
        return

    rdf = pd.DataFrame(race_records)

    # Report by consensus level
    levels = [
        ('all_agree', 'All 3 agree' if expert_col else 'ML+AI agree'),
        ('two_agree', '2 of 3 agree'),
        ('no_agree', 'No agreement'),
    ]
    if expert_col is None:
        levels = [
            ('all_agree', 'ML + AI agree'),
            ('no_agree', 'ML ≠ AI'),
        ]

    print(f"\n    {'Level':<16s} {'Races':>6s} {'Won':>4s} {'Win%':>6s} "
          f"{'AvgML':>7s} {'$2 ROI':>7s}")
    print(f"    {'-'*16} {'-'*6} {'-'*4} {'-'*6} {'-'*7} {'-'*7}")

    for key, label in levels:
        subset = rdf[rdf['level'] == key]
        if subset.empty:
            continue

        n = len(subset)
        wins = subset['won'].sum()
        win_pct = wins / n
        avg_ml = subset['ml_prob'].mean()

        # ROI: $2 win bet on consensus pick every race
        total_bet = n * 2
        total_return = subset[subset['won'] == 1]['win_paid'].sum()
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0

        avg_odds = f"{1/avg_ml - 1:.1f}/1" if avg_ml > 0 else "n/a"

        print(f"    {label:<16s} {n:>6d} {wins:>4d} {win_pct:>5.0%} "
              f"{avg_odds:>7s} {roi:>+6.1%}")

    # Bonus: consensus picks at longer odds (3/1+)
    if expert_col:
        consensus = rdf[rdf['level'] == 'all_agree']
    else:
        consensus = rdf[rdf['level'] == 'all_agree']

    if not consensus.empty:
        overlay = consensus[consensus['ml_decimal'] >= 3.0]
        if not overlay.empty:
            n = len(overlay)
            wins = overlay['won'].sum()
            total_bet = n * 2
            total_return = overlay[overlay['won'] == 1]['win_paid'].sum()
            roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
            print(f"\n    Consensus at 3/1+: {n} races, {wins} won ({wins/n:.0%}), "
                  f"ROI {roi:+.1%}")

        short = consensus[consensus['ml_decimal'] < 3.0]
        if not short.empty:
            n = len(short)
            wins = short['won'].sum()
            total_bet = n * 2
            total_return = short[short['won'] == 1]['win_paid'].sum()
            roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
            print(f"    Consensus at <3/1: {n} races, {wins} won ({wins/n:.0%}), "
                  f"ROI {roi:+.1%}")


# =============================================================================
# 9. BACKTEST
# =============================================================================

def run_backtest(std_dev=5.0, degrees_of_freedom=1.0, ml_regularization=0.25,
                 optimize=False, num_sims=50000):
    """Full backtest across all dates with results."""

    picks = load_all_picks()
    if picks.empty:
        print("No AI picks found.")
        return None

    entries = load_entries()
    if entries.empty:
        print("No results found.")
        return None

    # Inner join = scratches removed
    df = merge_picks_and_results(picks, entries)
    if df.empty:
        print("No matching picks + results.")
        print(f"  Pick dates:  {sorted(picks['date'].unique())}")
        print(f"  Entry dates: {sorted(entries['date'].unique())}")
        return None

    gcols = [c for c in GROUP_COLS if c in df.columns]
    n_races = df.groupby(gcols).ngroups
    print(f"Merged: {len(df)} horses across {n_races} races, "
          f"{df['date'].nunique()} dates")

    # Normalize ML probs
    df = normalize_ml_probs(df)

    # Impute missing AI estimates
    print("Imputing missing estimates...")
    df = impute_missing(df, std_dev, degrees_of_freedom)

    for provider in AI_PROVIDERS:
        bc = f'{provider}_beyer'
        pc = f'{provider}_win_prob'
        b_n = df[bc].notna().sum() if bc in df.columns else 0
        p_n = df[pc].notna().sum() if pc in df.columns else 0
        if b_n or p_n:
            print(f"  {provider:>8s}: {b_n:4d} beyers, {p_n:4d} probs")
    if 'ml_win_prob_norm' in df.columns:
        print(f"  {'ml':>8s}: {df['ml_win_prob_norm'].notna().sum():4d} probs")

    # Weights: load from fitted_params.json if available, else use SOURCE_ROSTER defaults
    fitted_path = DATA_PATH / 'fitted_params.json'
    if fitted_path.exists() and not optimize:
        with open(fitted_path) as f:
            fitted = json.load(f)
        weights = fitted.get('weights', {s['name']: s['weight'] for s in SOURCE_ROSTER})
        std_dev = fitted.get('std_dev', std_dev)
        degrees_of_freedom = fitted.get('degrees_of_freedom', degrees_of_freedom)
        ml_regularization = fitted.get('ml_regularization', ml_regularization)
        print(f"Loaded fitted_params.json (std={std_dev:.2f}, df={degrees_of_freedom:.2f}, "
              f"ml_reg={ml_regularization:.2f})")
        print(f"  Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
    else:
        weights = {s['name']: s['weight'] for s in SOURCE_ROSTER}

    # Build initial composite
    df = build_composite(df, std_dev, degrees_of_freedom, num_sims,
                         weights=weights, ml_regularization=ml_regularization)

    if optimize:
        print("\n--- Optimizing params (std, df, ml_reg) ---")
        std_dev, degrees_of_freedom, ml_regularization = optimize_params(
            df, initial_std=std_dev, initial_df=degrees_of_freedom,
            initial_ml_reg=ml_regularization,
            num_sims=num_sims // 5
        )
        # Rebuild with new params
        df = build_composite(df, std_dev, degrees_of_freedom, num_sims // 5,
                             ml_regularization=ml_regularization)

        print("\n--- Optimizing source weights ---")
        weights = optimize_weights(df, std_dev, degrees_of_freedom,
                                   num_sims=num_sims // 5)

    # Final composite with best params
    print(f"\nFinal composite (std={std_dev:.2f}, df={degrees_of_freedom:.2f}, "
          f"ml_reg={ml_regularization:.2f})...")
    df = build_composite(df, std_dev, degrees_of_freedom, num_sims, weights=weights,
                         ml_regularization=ml_regularization)

    # Evaluate
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    metrics = evaluate(df)
    print(f"  Races:              {metrics['n_races']}")
    print(f"  Mean Spearman ρ:    {metrics.get('spearman_mean', np.nan):.4f}")
    print(f"  Median Spearman ρ:  {metrics.get('spearman_median', np.nan):.4f}")
    if 'log_loss' in metrics:
        print(f"  Log loss:           {metrics['log_loss']:.4f}")
    if 'top_pick_win_rate' in metrics:
        print(f"  Top pick win rate:  {metrics['top_pick_win_rate']:.1%}")

    if 'calibration' in metrics:
        print("\n  Calibration:")
        print(metrics['calibration'].to_string())

    # Per-source evaluation
    print("\n  Per-source Spearman:")
    for src in [s['name'] for s in SOURCE_ROSTER]:
        col = f'{src}_src'
        if col in df.columns and df[col].notna().any():
            m = evaluate(df, prob_col=col)
            print(f"    {src:>15s}: ρ={m.get('spearman_mean', np.nan):.4f}  "
                  f"(n={m['n_races']})")
    # Also show ML baseline
    if 'ml_win_prob_norm' in df.columns and df['ml_win_prob_norm'].notna().any():
        m = evaluate(df, prob_col='ml_win_prob_norm')
        print(f"    {'ml':>15s}: ρ={m.get('spearman_mean', np.nan):.4f}  "
              f"(n={m['n_races']})")

    # ── Model comparison: ML, ML+experts, AI, AI+experts ──
    print("\nBuilding expert-adjusted variants...")

    # ML + experts
    if 'ml_win_prob_norm' in df.columns:
        result = apply_historical_experts(df, 'ml_win_prob_norm', 'ml_plus_experts')
        if isinstance(result, tuple):
            df, n_exp, n_adj = result
            print(f"  ML + experts: {n_exp} expert files, {n_adj} horse adjustments")
        else:
            df = result

    # AI + experts
    if 'ensemble_prob' in df.columns:
        result = apply_historical_experts(df, 'ensemble_prob', 'ai_plus_experts')
        if isinstance(result, tuple):
            df, n_exp, n_adj = result
            print(f"  AI + experts: {n_exp} expert files, {n_adj} horse adjustments")
        else:
            df = result

    print_comparison_report(df)

    # AI-vs-ML disagreement analysis
    _print_disagreement_report(df)

    # Consensus analysis
    _print_consensus_report(df)

    # Export fitted equation
    export_equation(std_dev, degrees_of_freedom, weights,
                    ml_regularization=ml_regularization)

    return df, metrics, {'std_dev': std_dev, 'degrees_of_freedom': degrees_of_freedom,
                         'ml_regularization': ml_regularization, 'weights': weights}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest AI handicapping")
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--std', type=float, default=5.0)
    parser.add_argument('--df', type=float, default=1.0)
    parser.add_argument('--ml-reg', type=float, default=0.25,
                        help='ML regularization weight on beyer sim (0-1)')
    parser.add_argument('--sims', type=int, default=50000)
    args = parser.parse_args()

    result = run_backtest(
        std_dev=args.std, degrees_of_freedom=args.df,
        ml_regularization=args.ml_reg,
        optimize=args.optimize, num_sims=args.sims
    )

    if result is not None:
        df, metrics, params = result
        out = PICKS_DIR / 'backtest_results.csv'
        df.to_csv(out, index=False, encoding='utf-8')
        print(f"\nSaved {len(df)} rows to {out}")