"""
backtest_kfold.py

Simulate predict_v2's bet logic on the out-of-sample CV predictions from
regression_analysis.py. Tells you, for each signal and an ensemble of all
three: what ROI did you achieve, and how much of the 17% takeout did you
cut into?

A random bettor loses 17% per $1 in a 17%-takeout pool. If your system
achieves -10% ROI, you've cut into 7 percentage points of the takeout.
Positive ROI means you've overcome the house edge entirely.

The four "systems" evaluated:

  1. LOG:     pick the top cv_prob horse per race
  2. RDG:     pick the top ridge_value horse per race
  3. RAW:     pick the top raw_ridge_value horse per race
  4. ENS:     simple average of per-model ranks; pick horse with lowest
              mean rank (most agreement across models). --meta-model
              flag is a placeholder for future stacked blender.

Each "pick" is placed with predict_v2's bet-type rules:
  ML ≤ 9/2:  W   (cost $2, payout = win_paid)
  ML ≤ 9/1:  WP  (cost $4, payout = win_paid + place_paid)
  else:      WPS (cost $6, payout = win_paid + place_paid + show_paid)

Also evaluates filtered subsets:
  - All races (top pick always, even when marginal)
  - BET-only: only bet when predict_v2's trigger would fire
  - AGREE-only: only bet when ≥2 models see the horse as positive value

Reads from:
    {DATA_PATH}/regression_predictions.csv

Usage:
    python backtest_kfold.py
    python backtest_kfold.py --takeout 0.17
    python backtest_kfold.py --meta-model  # placeholder, not yet implemented
"""

import os, argparse, warnings, json
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])


# ── Constants mirroring predict_v2 ──────────────────────────────────
# These match predict_v2.py so the backtest simulates the real production
# bet logic. If you change predict_v2's defaults, update these too.

ML_WIN_ONLY_MAX = 4.5   # ML ≤ 4.5 (9/2) → W only
ML_WP_MAX       = 9.0   # ML ≤ 9.0 (9/1) → WP, else WPS

WIN_TAKEOUT = 0.17

# Logistic (ev_per_2 on a $2 win bet)
LOGISTIC_MIN = 0.00
LOGISTIC_EDGE_THRESHOLD = 0.05

# Log-ridge (normalized to RIDGE_RACE_TOTAL per race)
RIDGE_RACE_TOTAL = 18.3
RIDGE_MIN_MULTIPLIER = 0.70
RIDGE_EDGE_MULTIPLIER = 1.05
RIDGE_FLOOR = 0.5

# Raw-ridge (calibrated to training CV μ, σ — not per-race normalized)
RAW_EDGE_K = 1.0
RAW_MIN_K  = -0.5


ENTRIES_DIR = DATA_PATH / 'entries'
META_DIR = DATA_PATH / 'meta'


# ── predict_v3 imports ──────────────────────────────────────────────
# We import predict_v3 for shared constants like ML_MAX_FOR_BET.
# The entropy-based race-quality filter is computed locally in this
# file from a dynamic q67 threshold over the loaded data; we also
# WRITE that threshold (alongside thresholds for the other entropy
# axes) to a JSON file that predict_v3 reads at race time. This way
# the live gate uses the same numerical thresholds the backtest
# validates against.
#
# Note: v3 imports v2, so this transitively imports v2 too. Both modules
# are clean (no top-level side effects beyond load_dotenv), so this is
# safe at module load time.
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    import predict_v3 as p3
    HAS_V3 = True
except ImportError as e:
    print(f"WARNING: couldn't import predict_v3 ({e}); v3 backtest will be skipped")
    HAS_V3 = False


def _parse_payout(val):
    """Parse payout string like '$8.40' to float. Non-finishers → 0.
    Matches regression_analysis._parse_payout behavior so merges are consistent."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if s == '' or s.lower() in ('nan', 'none', '-'):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


# Cache of loaded meta CSVs by (track, date) so we don't re-read for every
# race in the exotic backtest. Cleared per backtest run via _META_CACHE.clear()
# at the start of run_backtest if you want a fresh load.
_META_CACHE = {}


def _meta_path_candidates(track, date):
    """
    Return the list of meta-file paths to try, in order, for a given
    track/date. Backtest_kfold normalizes track names internally (e.g.,
    'churchill-downs' → 'churchill', 'gulfstream-park' → 'gulfstream')
    via _merge_from_entries' alias table, but on-disk meta files are
    saved by download_results.py with whatever --track string was
    passed — usually the hyphenated form scraped from horseracingnation.

    So we try the normalized form first (most likely match for files
    saved with the same convention this code uses elsewhere), then fall
    back to known on-disk variants. Self-correcting whichever way the
    files are named.
    """
    # Reverse of the _TRACK_ALIASES table inside _merge_from_entries.
    # Kept here as a small explicit table since the lookup is the hot
    # path. If you add a new track alias, add the on-disk variant here.
    REVERSE_ALIASES = {
        'churchill': ['churchill-downs', 'churchill downs'],
        'gulfstream': ['gulfstream-park', 'gulfstream park'],
    }
    candidates = [META_DIR / f'{track}_{date}.csv']
    for variant in REVERSE_ALIASES.get(track, []):
        candidates.append(META_DIR / f'{variant}_{date}.csv')
    return candidates


def load_race_exotics(track, date, race):
    """
    Load exotic-bet payout info for a single (track, date, race) from the
    meta CSV. Returns dict with exacta_finish, exacta_payout, trifecta_finish,
    trifecta_payout, or None if the meta file is missing or the race isn't
    in it.

    exacta_finish: tuple of ints like (3, 2) — the actual 1st-2nd order
    trifecta_finish: tuple of ints like (3, 2, 5)
    payouts: float, $0 if no payout (e.g. dead heat or missing data)

    The meta CSV has columns exacta_finish ("3-2"), exacta_$2_payout ("$53.26"),
    trifecta_finish, trifecta_$2_payout among many others. We only care
    about those four for the exotics backtest.
    """
    cache_key = (track, date)
    if cache_key not in _META_CACHE:
        loaded = None
        for path in _meta_path_candidates(track, date):
            if path.exists():
                try:
                    loaded = pd.read_csv(path)
                    break
                except Exception as e:
                    print(f"  WARN: failed to load {path}: {e}")
                    # Continue to next candidate rather than caching None
                    # — a corrupt file shouldn't block a good backup variant
        _META_CACHE[cache_key] = loaded

    meta = _META_CACHE[cache_key]
    if meta is None:
        return None

    row = meta[meta['race'] == race]
    if row.empty:
        return None
    r = row.iloc[0]

    def _parse_finish(val):
        """Parse e.g. '3-2-5' to (3, 2, 5). Returns None on failure."""
        if pd.isna(val) or str(val).strip() == '':
            return None
        try:
            return tuple(int(x) for x in str(val).strip().split('-'))
        except (ValueError, TypeError):
            return None

    return {
        'exacta_finish': _parse_finish(r.get('exacta_finish')),
        'exacta_payout': _parse_payout(r.get('exacta_$2_payout')),
        'trifecta_finish': _parse_finish(r.get('trifecta_finish')),
        'trifecta_payout': _parse_payout(r.get('trifecta_$2_payout')),
    }


def _parse_ml_odds(odds_str):
    """Convert ML odds string (e.g. '5/2' or '3') to implied probability."""
    try:
        if pd.isna(odds_str) or str(odds_str).strip() == '':
            return np.nan
        s = str(odds_str).strip()
        if '/' in s:
            num, den = map(float, s.split('/'))
            return den / (num + den)
        val = float(s)
        return 1 / (val + 1) if val >= 1 else val
    except (ValueError, ZeroDivisionError):
        return np.nan


def _merge_from_entries(df, needed_cols):
    """
    Load entries CSVs and merge `needed_cols` into df on
    (date, track, race, horse_number).

    Called when regression_predictions.csv is missing payout or ml_decimal
    columns — rather than forcing a full re-run of regression_analysis.py,
    we re-derive those columns from the entries files. They're the ground
    truth for payouts and odds anyway; regression_analysis just happens to
    pass them through.

    Supports the same payout format as regression_analysis._parse_payout
    (string with $ sign, commas, etc.) and the same track/date normalizers.
    """
    if not ENTRIES_DIR.exists():
        print(f"  ENTRIES_DIR {ENTRIES_DIR} does not exist — cannot merge "
              f"missing columns {needed_cols}")
        return df

    # Same normalizers as regression_analysis (verbatim so merge keys match)
    _TRACK_ALIASES = {
        'gulfstream-park': 'gulfstream', 'gulfstream park': 'gulfstream',
        'churchill-downs': 'churchill', 'churchill downs': 'churchill',
    }

    def _normalize_track(t):
        if pd.isna(t):
            return t
        return _TRACK_ALIASES.get(str(t).strip().lower(), str(t).strip().lower())

    def _normalize_date(d):
        d = str(d).strip()
        if len(d) == 10 and d[4] == '-' and d[7] == '-':
            return d
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

    entry_dfs = []
    for f in ENTRIES_DIR.glob('*.csv'):
        try:
            entry_dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  skipping {f.name}: {e}")

    if not entry_dfs:
        print(f"  No readable entries CSVs in {ENTRIES_DIR}")
        return df

    entries = pd.concat(entry_dfs, ignore_index=True)

    if 'running' in entries.columns:
        entries = entries[entries['running'] == 1].copy()
    if 'date' in entries.columns:
        entries['date'] = entries['date'].apply(_normalize_date)
    if 'track' in entries.columns:
        entries['track'] = entries['track'].apply(_normalize_track)
    if 'horse' in entries.columns and 'horse_number' not in entries.columns:
        entries['horse_number'] = pd.to_numeric(entries['horse'], errors='coerce')

    # Derive the columns we need
    for col in ('win_paid', 'place_paid', 'show_paid'):
        num_col = f'{col}_num'
        if num_col in needed_cols:
            if col in entries.columns:
                entries[num_col] = entries[col].apply(_parse_payout)
            else:
                entries[num_col] = 0.0

    if 'ml_decimal' in needed_cols or 'ml_prob_norm' in needed_cols:
        if 'ml_odds' in entries.columns:
            entries['ml_prob'] = entries['ml_odds'].apply(_parse_ml_odds)
            # ml_decimal = 1 / ml_prob (from pari-mutuel definition)
            entries['ml_decimal'] = entries['ml_prob'].apply(
                lambda p: 1 / p if pd.notna(p) and p > 0 else np.nan
            )

    # `won` and `finish` from entries finish column. predict_v2 doesn't write
    # these (it runs pre-race), so walk_forward_summary needs them merged.
    if 'won' in needed_cols or 'finish' in needed_cols:
        if 'finish' in entries.columns:
            entries['finish'] = pd.to_numeric(entries['finish'], errors='coerce')
            if 'won' in needed_cols:
                entries['won'] = (entries['finish'] == 1).astype(int)

    # Merge — only the columns we need, on (date, track, race, horse_number)
    merge_cols = ['date', 'track', 'race', 'horse_number']
    bring = [c for c in needed_cols if c in entries.columns]
    if not bring:
        print(f"  None of the needed columns {needed_cols} found in entries")
        return df

    # Normalize df's merge keys too (in case df didn't come through the same pipeline)
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str).apply(_normalize_date)
    if 'track' in df.columns:
        df['track'] = df['track'].apply(_normalize_track)

    df['race'] = pd.to_numeric(df['race'], errors='coerce')
    df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce')
    entries['race'] = pd.to_numeric(entries['race'], errors='coerce')
    entries['horse_number'] = pd.to_numeric(entries['horse_number'], errors='coerce')

    before = len(df)
    df = df.merge(entries[merge_cols + bring], on=merge_cols, how='left')
    matched = df[bring[0]].notna().sum() if bring else 0
    print(f"  Merged {bring} from entries: "
          f"{matched}/{before} rows matched")

    return df


def load_cv_predictions(source=None):
    """Load out-of-sample predictions.

    Two supported sources:
      - regression_predictions.csv (default): CV predictions from
        regression_analysis.py. Columns: cv_prob, ridge_value,
        raw_ridge_value, won.
      - walk_forward_summary.csv: concatenated predict_v2 outputs from
        backtest_walk_forward.py. Columns: ensemble_prob, ev_per_2,
        ridge_value, raw_value. Needs `won` merged from entries.

    The walk-forward source is preferred because it reflects the real
    predict_v2 bet logic (LLM ensemble + correct takeout-adjusted ev_per_2)
    rather than just the logistic CV output. But it requires running
    backtest_walk_forward.py first.

    If the CSV is missing payout or odds columns, attempts to merge them
    from the entries CSVs. Only raises if essential model-output columns
    are missing — those can't be recovered.

    Args:
        source: 'walk_forward' or 'cv' (default auto-detects based on
                what files exist, preferring walk_forward).
    """
    wf_path = DATA_PATH / 'predictions' / 'walk_forward_summary.csv'
    cv_path = DATA_PATH / 'regression_predictions.csv'

    # Auto-detect if not specified: prefer walk_forward when available
    if source is None:
        if wf_path.exists():
            source = 'walk_forward'
        elif cv_path.exists():
            source = 'cv'
        else:
            raise FileNotFoundError(
                f"Neither {wf_path} nor {cv_path} exists. Run "
                f"backtest_walk_forward.py or regression_analysis.py first."
            )

    if source == 'walk_forward':
        path = wf_path
    elif source == 'cv':
        path = cv_path
    else:
        raise ValueError(f"source must be 'walk_forward' or 'cv', got {source}")

    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} horses from {path}")
    print(f"  Source: {source}")

    # Normalize column names so downstream code works on both sources.
    # walk_forward has {ensemble_prob, raw_value}; cv has {cv_prob, raw_ridge_value}.
    # Map both to a canonical {cv_prob, raw_ridge_value} internally.
    if source == 'walk_forward':
        if 'ensemble_prob' in df.columns:
            df['cv_prob'] = df['ensemble_prob']
        if 'raw_value' in df.columns and 'raw_ridge_value' not in df.columns:
            df['raw_ridge_value'] = df['raw_value']
        # predict_v2 already writes ev_per_2 using the correct formula —
        # keep it as-is, don't let compute_ev_per_2 overwrite it.
        # (compute_ev_per_2 checks if ev_per_2 is already present; we pass
        # `prefer_existing_ev=True` into run_backtest for walk_forward.)

    n_races = df.groupby(['date', 'track', 'race']).ngroups
    print(f"  Across {n_races} races, {df['date'].nunique()} dates, "
          f"{df['track'].nunique()} tracks")

    # Sanity-check what we have — cv_prob must be present after normalization
    if 'cv_prob' not in df.columns:
        raise RuntimeError(
            f"{path} missing cv_prob (or ensemble_prob). "
            f"Cannot proceed without a logistic probability signal."
        )

    # `won` is only in regression_predictions.csv (regression_analysis merged
    # it from entries). walk_forward_summary doesn't have it because
    # predict_v2 runs pre-race. Merge from entries if missing.
    if 'won' not in df.columns:
        print(f"  `won` column missing; merging from {ENTRIES_DIR}...")
        df = _merge_from_entries(df, ['won', 'finish'])
        if 'won' not in df.columns:
            raise RuntimeError(
                "Could not recover `won` from entries. Check that entries "
                "files have a `finish` column with race results."
            )
        df['won'] = df['won'].fillna(0).astype(int)

    has_ridge = 'ridge_value' in df.columns and df['ridge_value'].notna().any()
    has_raw = 'raw_ridge_value' in df.columns and df['raw_ridge_value'].notna().any()
    print(f"  Available signals: logistic✓, "
          f"log-ridge{'✓' if has_ridge else '✗'}, "
          f"raw-ridge{'✓' if has_raw else '✗'}")

    # Payout columns — required for ROI simulation.
    payout_cols = ['win_paid_num', 'place_paid_num', 'show_paid_num']
    missing_payouts = [c for c in payout_cols if c not in df.columns]
    if missing_payouts:
        print(f"\n  {path.name} missing {missing_payouts}; "
              f"merging from {ENTRIES_DIR}...")
        df = _merge_from_entries(df, missing_payouts)
        still_missing = [c for c in missing_payouts if c not in df.columns]
        if still_missing:
            raise RuntimeError(
                f"Could not recover {still_missing} — neither in "
                f"{path.name} nor derivable from entries CSVs. "
                f"Check that entries files exist at {ENTRIES_DIR} and contain "
                f"win_paid, place_paid, show_paid columns."
            )
        for c in payout_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

    # ML odds — needed for bet-type logic. Also recoverable from entries.
    if 'ml_decimal' not in df.columns:
        print(f"\n  {path.name} missing ml_decimal; "
              f"merging from {ENTRIES_DIR}...")
        df = _merge_from_entries(df, ['ml_decimal'])
        if 'ml_decimal' not in df.columns:
            print(f"  WARNING: no ml_decimal available — bet-type selection "
                  f"will default to WPS for all bets")

    # ml_prob_norm for diagnostics — predict_v2 writes it; regression_analysis
    # writes it under the same name. If missing, fall back to cv_prob.
    if 'ml_prob_norm' not in df.columns:
        print(f"  (ml_prob_norm missing, using cv_prob as ML-prob fallback for diagnostics)")
        df['ml_prob_norm'] = df['cv_prob']

    df = filter_broken_races(df)

    return df


def filter_broken_races(df):
    """
    Drop races that shouldn't contribute to backtest ROI.

    Three cases:

    (1) PENDING races: no finish data yet (today's card before results
        are posted). Silently excluded — they're not broken, they just
        haven't happened. The log line reports them separately from (2).

    (2) BROKEN races: finish data exists in entries but the winner isn't
        in the predictions dataframe. The AI pipeline missed the winning
        horse, so every horse in `df` for this race has won=0. Betting
        those horses is a guaranteed loss, poisoning ROI downward.

    (3) MULTI-winner races: two or more rows have won=1 for the same
        race. Duplicate finish=1 or merge-duplication bug. Ambiguous
        ground truth; dropped.

    The "pending" signal is: `finish` column exists in df but is all-NaN
    for that race. If finish isn't even present, we fall back to "no
    winner = broken" (conservative — can't tell pending apart).

    Dynamic: no hardcoded blocklist. Heal broken cards by re-running
    combine_ai_picks.py / the AI pipeline.
    """
    if 'won' not in df.columns:
        return df

    winners_per_race = df.groupby(['date', 'track', 'race'])['won'].sum()
    good_races = winners_per_race[winners_per_race == 1].index

    # Distinguish pending from broken using finish column
    if 'finish' in df.columns:
        finish_present_per_race = df.groupby(
            ['date', 'track', 'race'])['finish'].apply(
            lambda s: s.notna().any())
    else:
        # Without a finish column we can't tell pending from broken;
        # default to treating all as broken (conservative)
        finish_present_per_race = pd.Series(
            True, index=winners_per_race.index)

    zero_winner_races = winners_per_race[winners_per_race == 0].index
    pending_races = [r for r in zero_winner_races
                    if not finish_present_per_race.get(r, True)]
    broken_zero_races = [r for r in zero_winner_races
                        if finish_present_per_race.get(r, True)]
    multi_winner_races = winners_per_race[winners_per_race > 1].index.tolist()

    n_total_bad = len(pending_races) + len(broken_zero_races) + len(multi_winner_races)
    if n_total_bad == 0:
        return df

    n_before = len(df)
    race_key = df.set_index(['date', 'track', 'race']).index
    df = df[race_key.isin(good_races)].copy()
    n_after = len(df)

    # Report — pending is neutral, broken/multi is a warning
    msgs = []
    if pending_races:
        msgs.append(f"{len(pending_races)} pending (no results yet)")
    if broken_zero_races:
        msgs.append(f"{len(broken_zero_races)} broken (winner not in predictions)")
    if multi_winner_races:
        msgs.append(f"{len(multi_winner_races)} multi-winner (data error)")
    print(f"  Filtered {n_total_bad} races "
          f"({', '.join(msgs)}): {n_before} → {n_after} rows")

    if broken_zero_races and len(broken_zero_races) <= 15:
        # Brief enumeration so the user knows which cards still need fixing
        print(f"    Broken: {', '.join(str(r) for r in broken_zero_races[:15])}")

    return df


def compute_ev_per_2(df, takeout=WIN_TAKEOUT):
    """
    Reconstruct predict_v2's ev_per_2 per horse. Matches calculate_edges()
    in predict_v2:

      ROI = (model_prob / ml_prob) × (1 - takeout) - 1
      ev_per_2 = 2 × ROI

    IMPORTANT: The previous implementation used the wrong formula
    `2 × (cv_prob × ml_decimal - 1)` which OMITS the (1 - takeout) factor.
    That version overestimated EV by ~20% and triggered far more bets on
    longshots (where `ml_decimal` is large and the takeout miss compounds)
    than predict_v2 would actually make.

    If the dataframe already has ev_per_2 (walk_forward source, where
    predict_v2 wrote it with the correct formula), leaves it alone and
    skips recomputation.

    For the cv source, uses cv_prob as model_prob and ml_prob_norm or the
    inverse of ml_decimal as ml_prob. Technically cv_prob is the logistic
    out-of-sample prediction, NOT the LLM ensemble_prob that predict_v2
    uses live — so this ev_per_2 tests "the logistic model's edge" in
    isolation, not "predict_v2's live edge signal." Walk-forward source
    is the honest test.
    """
    df = df.copy()

    # Walk-forward source: predict_v2 already wrote ev_per_2 with the
    # correct formula. Don't clobber it.
    if 'ev_per_2' in df.columns and df['ev_per_2'].notna().any():
        n_present = df['ev_per_2'].notna().sum()
        print(f"  ev_per_2 already present ({n_present}/{len(df)} rows); "
              f"using as-is from predict_v2 output")
        return df

    if 'ml_decimal' not in df.columns:
        print("  WARNING: no ml_decimal column — ev_per_2 cannot be computed")
        print("  LOG system will pick top cv_prob but won't apply EV thresholds")
        df['ev_per_2'] = np.nan
        return df

    # Derive ml_prob from ml_decimal: ml_prob = 1 / ml_decimal (pari-mutuel
    # convention, same as predict_v2).
    ml_prob = 1.0 / df['ml_decimal'].where(df['ml_decimal'] > 0)

    # Correct formula (matches predict_v2.calculate_edges)
    proj = df['cv_prob']
    ev_roi = (proj / ml_prob) * (1 - takeout) - 1
    df['ev_per_2'] = np.where(ml_prob > 0, 2 * ev_roi, np.nan)

    return df


def apply_ridge_race_norm(df, ridge_col, race_total, floor):
    """
    Normalize ridge predictions per race to sum to `race_total`, floored
    at `floor`. Mirrors predict_v2's normalization for log-ridge.

    Applied only to log-ridge — raw-ridge uses raw model outputs directly
    with card-wide calibration (see compute_raw_calibration below).
    """
    df = df.copy()
    out_col = f'{ridge_col}_norm'
    df[out_col] = np.nan

    for _, g in df.groupby(['date', 'track', 'race']):
        idx = g.index
        raw = df.loc[idx, ridge_col].values.astype(float)
        floored = np.where(np.isnan(raw), floor, np.maximum(raw, floor))
        total = floored.sum()
        if total > 0:
            df.loc[idx, out_col] = floored * (race_total / total)

    return df


def compute_raw_calibration(df, raw_col):
    """
    Card-wide μ and σ of the raw-ridge predictions. Mirrors predict_v2's
    apply_raw_value_model — these are the reference points for the raw
    BET trigger (μ + K·σ).

    In predict_v2 these come from the training JSON (cv_pred_mean,
    cv_pred_std). Here we're analyzing the CV predictions directly, so
    the empirical μ/σ of these predictions IS the training calibration.
    """
    return df[raw_col].mean(), df[raw_col].std()


def compute_bet_flags(df, raw_cv_mean, raw_cv_std):
    """
    Add BET / AGREE / floor-veto flags per horse, matching predict_v2's
    generate_bets logic exactly. Each row gets:
      - log_bet, log_pos, log_qual
      - rdg_bet, rdg_pos, rdg_qual  (from ridge_value_norm, per-race BE)
      - raw_bet, raw_pos, raw_qual  (from raw_ridge_value, per-row calib)
      - n_models_positive
      - is_agree  (≥2 of 3 positive)
      - any_bet   (≥1 model at edge AND no floor veto)

    The raw_cv_mean and raw_cv_std arguments are POOLED fallbacks, used
    only when per-row calibration columns (raw_cv_mean/raw_cv_std) aren't
    present in df. Walk-forward predictions have per-card calibration
    stamped on each row — we honor those to match exactly what was in
    effect when predict_v2 made the prediction at training time. Pooling
    across multiple holdouts (each with its own JSON μ/σ) would break
    the `value is K·σ above training mean` semantics that drive the
    raw-ridge trigger.
    """
    df = df.copy()
    takeout_gross = 1.0 / max(1e-6, 1.0 - WIN_TAKEOUT)

    has_ridge = 'ridge_value_norm' in df.columns
    has_raw = 'raw_ridge_value' in df.columns

    # Per-race break-evens for log-ridge (matches predict_v2)
    def _per_race_ridge_be(race_df):
        n = len(race_df)
        if n == 0:
            return 0.0
        raw_fair = RIDGE_RACE_TOTAL / n
        return raw_fair * takeout_gross

    # Per-row vs pooled raw calibration
    # If predict_v2 stamped raw_cv_mean and raw_cv_std on each row, use
    # those (walk-forward path). Otherwise, use the single pooled values
    # passed in (CV path, where all predictions share one calibration).
    use_per_row_calib = (
        has_raw
        and 'raw_cv_mean' in df.columns
        and 'raw_cv_std' in df.columns
        and df['raw_cv_mean'].notna().any()
        and df['raw_cv_std'].notna().any()
    )

    if has_raw and use_per_row_calib:
        # Each row has its own μ, σ — fill any gaps with the pooled fallback
        raw_mean_series = df['raw_cv_mean'].fillna(raw_cv_mean)
        raw_std_series = df['raw_cv_std'].fillna(raw_cv_std)
        raw_be = raw_mean_series
        raw_edge = raw_mean_series + RAW_EDGE_K * raw_std_series
        raw_min = raw_mean_series + RAW_MIN_K * raw_std_series
        n_cards_with_calib = df[['date', 'track', 'raw_cv_mean', 'raw_cv_std']] \
            .drop_duplicates().shape[0]
        print(f"  Raw-ridge: using per-row calibration from "
              f"{n_cards_with_calib} distinct card-level μ/σ stamps")
    elif has_raw:
        # Pooled fallback — broadcast the scalars to series for uniform code below
        raw_be = pd.Series(raw_cv_mean, index=df.index)
        raw_edge = pd.Series(raw_cv_mean + RAW_EDGE_K * raw_cv_std, index=df.index)
        raw_min = pd.Series(raw_cv_mean + RAW_MIN_K * raw_cv_std, index=df.index)
        print(f"  Raw-ridge: using POOLED calibration "
              f"(μ=${raw_cv_mean:.3f}, σ=${raw_cv_std:.3f}). "
              f"If backtesting walk-forward output, consider regenerating "
              f"predictions with the updated predict_v2 that stamps per-row "
              f"calibration.")
    else:
        raw_be = raw_edge = raw_min = None

    # Initialize flag columns
    for flag in ['log_bet', 'log_pos', 'log_qual', 'rdg_bet', 'rdg_pos',
                 'rdg_qual', 'raw_bet', 'raw_pos', 'raw_qual']:
        df[flag] = False

    for (_, _, _), race_df in df.groupby(['date', 'track', 'race']):
        idx = race_df.index

        # Logistic
        if 'ev_per_2' in df.columns:
            ev = df.loc[idx, 'ev_per_2']
            df.loc[idx, 'log_qual'] = ev.fillna(-np.inf) >= LOGISTIC_MIN
            df.loc[idx, 'log_pos'] = ev.fillna(-np.inf) > 0
            df.loc[idx, 'log_bet'] = ev.fillna(-np.inf) >= LOGISTIC_EDGE_THRESHOLD

        # Log-ridge
        if has_ridge:
            ridge_be = _per_race_ridge_be(race_df)
            ridge_min_v = ridge_be * RIDGE_MIN_MULTIPLIER
            ridge_edge_v = ridge_be * RIDGE_EDGE_MULTIPLIER
            rv = df.loc[idx, 'ridge_value_norm']
            df.loc[idx, 'rdg_qual'] = rv.fillna(-np.inf) >= ridge_min_v
            df.loc[idx, 'rdg_pos']  = rv.fillna(-np.inf) >= ridge_be
            df.loc[idx, 'rdg_bet']  = rv.fillna(-np.inf) >= ridge_edge_v

        # Raw-ridge — use per-row thresholds from raw_be/raw_edge/raw_min Series
        if has_raw:
            rw = df.loc[idx, 'raw_ridge_value'].fillna(-np.inf)
            df.loc[idx, 'raw_qual'] = rw >= raw_min.loc[idx]
            df.loc[idx, 'raw_pos']  = rw >= raw_be.loc[idx]
            df.loc[idx, 'raw_bet']  = rw >= raw_edge.loc[idx]

    # Derived flags
    df['n_models_positive'] = (
        df['log_pos'].astype(int) + df['rdg_pos'].astype(int) +
        df['raw_pos'].astype(int)
    )
    df['is_agree'] = df['n_models_positive'] >= 2

    # BET = ≥1 edge trigger AND no active model vetoes on floor.
    # Mirrors predict_v2 — inactive models (e.g. no raw) don't veto.
    has_any_edge = df['log_bet'] | df['rdg_bet'] | df['raw_bet']
    # A model vetoes if it's active AND below its qualification floor.
    # For LOG, "active" means ev_per_2 is non-null.
    log_veto = df.get('ev_per_2', pd.Series(np.nan, index=df.index)).notna() & ~df['log_qual']
    rdg_veto = (df['ridge_value_norm'].notna() & ~df['rdg_qual']) if has_ridge else False
    raw_veto = (df['raw_ridge_value'].notna() & ~df['raw_qual']) if has_raw else False

    df['any_bet'] = has_any_edge & ~log_veto & ~rdg_veto & ~raw_veto

    return df


def compute_race_entropy_raw(probs):
    """
    Raw Shannon entropy of a race's probability distribution, in nats.

    H = -Σ p_i log(p_i)

    Bounded by [0, log(N)]: 0 = one horse holds all probability,
    log(N) = uniform distribution. NOT field-size invariant — a
    larger field has a higher ceiling. This is the right thing for
    diagnostics where you want to see the absolute amount of
    "uncertainty mass" in the race independent of how many horses
    there are.

    Inputs:
        probs: 1-D array-like of probabilities for one race.
               NaNs are dropped. Negatives are floored to 0.
               If the input doesn't sum to 1 (scratches, rounding,
               ridge values, etc.) it gets re-normalized to a
               probability distribution before the entropy calc.

    Returns:
        Float in [0, log(N)]. NaN if N < 2 or all values are
        zero/NaN (degenerate race).
    """
    arr = np.asarray(probs, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = np.clip(arr, 0.0, None)
    if len(arr) < 2:
        return np.nan
    total = arr.sum()
    if total <= 0:
        return np.nan
    p = arr / total
    p = p[p > 0]
    if len(p) < 2:
        return np.nan
    return -np.sum(p * np.log(p))


def compute_race_entropy(probs):
    """
    Normalized Shannon entropy: H / log(N).

    Bounded in [0, 1] by construction. Same input handling as
    compute_race_entropy_raw, just divides by log(N) at the end.

    Higher value = flatter race (more uncertain). Lower value =
    peaked race (model has a clear pick).

    Note: this is invariant to N for *uniform* distributions but
    drifts mildly with N for non-uniform ones. A "one strong
    contender plus weak filler" race scores higher entropy when
    there's more filler, even though the contender's relative
    dominance is the same. See compute_race_entropy_raw for the
    field-size-dependent version used in some diagnostics.
    """
    arr = np.asarray(probs, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = np.clip(arr, 0.0, None)
    if len(arr) < 2:
        return np.nan
    total = arr.sum()
    if total <= 0:
        return np.nan
    p = arr / total
    p = p[p > 0]
    if len(p) < 2:
        return np.nan
    h = -np.sum(p * np.log(p))
    return h / np.log(len(p))


# Backward-compat alias — older code paths may still call this.
# New code should use compute_race_entropy directly.
compute_ridge_entropy = compute_race_entropy


# v3 entropy threshold — populated dynamically by compute_v3_filters
# from the loaded data's 67th-percentile entropy (so the highest third
# of races by entropy passes the gate). Module-level so the print
# statements in run_backtest can reference whatever was computed.
# None means "not yet computed" (e.g., compute_v3_filters hasn't run
# or HAS_V3 is False).
# Entropy thresholds populated by compute_v3_filters. Kept as module
# globals so reporting code further down can read them. Q60/Q40 are
# the gate values used by the live two-rule policy; Q67 is preserved
# for back-compat with the reporting snapshot that grew up around it.
V3_ENTROPY_Q67 = None
V3_ENTROPY_Q60 = None
V3_ENTROPY_Q40 = None


def compute_v3_filters(df):
    """
    Add v3-style trigger flags + a forced_bet_type column to df.

    The v3 trigger logic (mirrors predict_v3.generate_bets_v3):
        Trigger condition: log_bet OR rdg_bet OR raw_bet (any model fires)
        Two rules, OR'd:
            Chalk rule:    race_entropy_norm ≥ q60  AND  ml_decimal ≤ ML_MAX_FOR_BET
            Longshot rule: race_entropy_norm ≤ q40  AND  ml_decimal ≥ ML_MIN_LONGSHOT
        Bet type: LOG-fired → W only; everything else → predict_v2's
                  standard W/WP/WPS-by-ml. Longshot-rule bets land in
                  WPS naturally since ml ≥ 12 > ML_WP_MAX (=9).

    The two-rule structure targets opposite corners of the entropy×ml
    bucket grid:
        (peakest, longshot) — peaked-market longshots that any model
            still likes. Hypothesis: when the public agrees on a
            chalk and our model still flags an outsider, the outsider
            may be genuinely overpriced. Currently weak in-sample
            (R² ≈ 0 on raw target, recent results favorite-heavy)
            but the corner is being tested prospectively.
        (flattest, chalk) — flat-market favorites where prices spread
            out enough to leave headroom for an edge to overcome
            17% takeout. The empirically-strongest cell so far.

    Mid-ml zone (8 < ml < 12) is excluded from both rules. The 3×5
    diagnostic showed mid-ml as the worst ROI bucket regardless of
    trigger or entropy band. No rule covers it.

    The gate uses NORMALIZED CV entropy (entropy of ensemble cv_prob
    divided by log(N)). Chosen over ML entropy because it rewards
    races where our own model has spread its probability mass —
    "races where our pick has competition" is a useful signal for
    value betting. ML entropy and raw CV entropy are still computed
    and stored for the bucket diagnostic.

    Thresholds (q60, q40) are computed dynamically from the loaded
    sample. The fraction of races passing each gate is stable by
    construction (40%) when more data is added, even as absolute
    threshold values drift.

    RAW model is now a third trigger source (was excluded from v3
    pre-this-change). Its empirical edge is weak in current sample
    but it's the model that targets payout magnitude, so including
    it on the longshot rule is the most natural fit. RAW also
    contributes to the chalk rule though it rarely fires there.

    Adds columns:
        race_entropy_ml          — H/log(N) of 1/ml_decimal (diagnostic)
        race_entropy_norm        — H/log(N) of cv_prob (used by gates)
        race_entropy_raw         — raw H of cv_prob (diagnostic)
        v3_log_fires             — log triggered AND passed either rule
        v3_rdg_fires             — rdg triggered AND passed either rule
        v3_raw_fires             — raw triggered AND passed either rule
        v3_chalk_rule_fires      — chalk rule passed
        v3_longshot_rule_fires   — longshot rule passed
        v3_bet                   — either rule fires
        forced_bet_type          — 'W' for log-fired, None otherwise

    Side effect 1: writes Q60/Q40/Q67 cv-norm entropy thresholds to
    module-level V3_ENTROPY_Q60/Q40/Q67.
    Side effect 2: writes a JSON file at DATA_PATH/predictions/
    v3_thresholds.json with q33/q40/q60/q67 for all three entropy
    axes, consumed by predict_v3 at race time.
    """
    global V3_ENTROPY_Q67, V3_ENTROPY_Q60, V3_ENTROPY_Q40

    if not HAS_V3:
        df['v3_bet'] = False
        return df

    df = df.copy()

    # Compute race entropy three ways. The GATE uses ML entropy
    # (rationale below); cv_prob entropy in normalized + raw form is
    # kept around for the bucket diagnostic.
    #
    # Why ML for the gate: the takeout-headroom argument is about
    # MARKET concentration, not model confidence. When the morning-
    # line implies a near-certain favorite, prices are short and
    # takeout eats most of any edge. When the morning line spreads
    # out, prices are longer and a value edge has more room to
    # survive takeout. That's a property of the market's probability
    # distribution, not ours. Gating on cv_prob entropy was selecting
    # races where OUR model says "I don't know" — philosophically
    # weird as a bet-selection criterion.
    #
    # Tradeoff: ml_decimal is morning line, set days in advance and
    # not updated for live odds movement. So this is "the market's
    # initial read on race shape" — which is fine for a stable
    # sample-period gate but won't react to last-minute action.
    #
    # ML probabilities have overround (track's margin baked in), so
    # they don't sum to 1 across a race. compute_race_entropy and
    # compute_race_entropy_raw re-normalize defensively before the
    # entropy calc, so this is correct math; just be aware that the
    # "probability" being analyzed is overround-stripped morning line.

    # ml-implied prob: 1/ml_decimal. NaN where ml_decimal is missing
    # or non-positive (defensive — shouldn't happen in clean data).
    if 'ml_decimal' in df.columns:
        df['_ml_prob'] = np.where(
            df['ml_decimal'] > 0,
            1.0 / df['ml_decimal'],
            np.nan,
        )
        race_ent_ml = (df.groupby(['date', 'track', 'race'])['_ml_prob']
                         .apply(compute_race_entropy))
        race_ent_ml.name = 'race_entropy_ml'
        df = df.merge(
            race_ent_ml.reset_index(),
            on=['date', 'track', 'race'], how='left',
            suffixes=('', '_drop'),
        )
        if 'race_entropy_ml_drop' in df.columns:
            df = df.drop(columns=['race_entropy_ml_drop'])
        df = df.drop(columns=['_ml_prob'])
    else:
        df['race_entropy_ml'] = np.nan

    # cv_prob (ensemble_prob if source=walk_forward, logistic CV
    # otherwise — see load_cv_predictions). Used by the diagnostic
    # only; the gate uses ML entropy now.
    if 'cv_prob' in df.columns:
        race_ent_norm = (df.groupby(['date', 'track', 'race'])['cv_prob']
                           .apply(compute_race_entropy))
        race_ent_norm.name = 'race_entropy_norm'
        race_ent_raw = (df.groupby(['date', 'track', 'race'])['cv_prob']
                          .apply(compute_race_entropy_raw))
        race_ent_raw.name = 'race_entropy_raw'
        df = df.merge(
            race_ent_norm.reset_index(),
            on=['date', 'track', 'race'], how='left',
            suffixes=('', '_drop'),
        )
        df = df.merge(
            race_ent_raw.reset_index(),
            on=['date', 'track', 'race'], how='left',
            suffixes=('', '_drop'),
        )
        for c in ('race_entropy_norm_drop', 'race_entropy_raw_drop'):
            if c in df.columns:
                df = df.drop(columns=[c])
    else:
        df['race_entropy_norm'] = np.nan
        df['race_entropy_raw'] = np.nan

    # Compute the q60 (chalk-rule gate) and q40 (longshot-rule gate)
    # thresholds from the per-race entropy distribution of the GATE axis
    # (race_entropy_norm — must match predict_v3's GATE_ENTROPY_KEY).
    # Use one row per race so we don't weight by horse count.
    #
    # Two rules now:
    #   Chalk rule    — entropy ≥ q60 (top 40% flattest races) AND ml ≤ 8
    #   Longshot rule — entropy ≤ q40 (bottom 40% peakest races) AND ml ≥ 12
    # Old single-gate q67 is also computed for back-compat with reporting
    # code that references V3_ENTROPY_Q67.
    race_ent_series = (df.drop_duplicates(['date', 'track', 'race'])
                         ['race_entropy_norm'].dropna())
    if len(race_ent_series) > 0:
        V3_ENTROPY_Q67 = float(race_ent_series.quantile(2/3))
        V3_ENTROPY_Q60 = float(race_ent_series.quantile(0.60))
        V3_ENTROPY_Q40 = float(race_ent_series.quantile(0.40))
    else:
        V3_ENTROPY_Q67 = V3_ENTROPY_Q60 = V3_ENTROPY_Q40 = None

    # ALSO compute q33/q40/q60/q67 for all three axes and write to a
    # JSON file so predict_v3 can apply the same thresholds at race time.
    # The backtest is the source of truth for these values; predict_v3
    # reads what we wrote here. The file lives in DATA_PATH/predictions
    # so it's collocated with walk_forward_summary.csv etc.
    #
    # We write the JSON unconditionally — even on a degenerate sample
    # the file gets refreshed with whatever's available (or nothing).
    # If predict_v3 finds the file but a needed key is missing it
    # should fail loudly rather than silently fall back.
    _race_lookup = df.drop_duplicates(['date', 'track', 'race'])
    threshold_payload = {
        'schema_version': 2,  # v2: added q40/q60 for two-rule gate
        'computed_at': datetime.now().isoformat(timespec='seconds'),
        'n_races_in_sample': int(len(_race_lookup)),
        'n_dates_in_sample': int(_race_lookup['date'].nunique()),
        'n_tracks_in_sample': int(_race_lookup['track'].nunique()),
        'thresholds': {},
    }
    for ent_col in ('race_entropy_ml', 'race_entropy_norm',
                    'race_entropy_raw'):
        if ent_col not in df.columns:
            continue
        s = _race_lookup[ent_col].dropna()
        if len(s) == 0:
            continue
        threshold_payload['thresholds'][f'{ent_col}_q33'] = float(s.quantile(1/3))
        threshold_payload['thresholds'][f'{ent_col}_q40'] = float(s.quantile(0.40))
        threshold_payload['thresholds'][f'{ent_col}_q60'] = float(s.quantile(0.60))
        threshold_payload['thresholds'][f'{ent_col}_q67'] = float(s.quantile(2/3))

    threshold_path = DATA_PATH / 'predictions' / 'v3_thresholds.json'
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(threshold_path, 'w') as f:
            json.dump(threshold_payload, f, indent=2)
        print(f"  Wrote v3 thresholds to {threshold_path}")
    except Exception as e:
        # Don't crash the backtest if the write fails — just warn. The
        # backtest's in-memory V3_ENTROPY_Q67 is still usable for this
        # run; only predict_v3's next run is affected.
        print(f"  WARN: couldn't write v3 thresholds to {threshold_path}: {e}")

    # ── Two-rule gate ──
    # Each rule consists of an entropy band + an ml band, and any
    # of (LOG, RDG, RAW) firing satisfies the trigger requirement.
    # A horse fires v3 if it passes EITHER rule. Rules are disjoint
    # by ml band (chalk ≤ 8, longshot ≥ 12), so no double-counting.
    log_bet = df.get('log_bet', pd.Series(False, index=df.index)).fillna(False)
    rdg_bet = df.get('rdg_bet', pd.Series(False, index=df.index)).fillna(False)
    raw_bet = df.get('raw_bet', pd.Series(False, index=df.index)).fillna(False)
    any_trigger = log_bet | rdg_bet | raw_bet

    # Chalk rule entropy gate: race_entropy_norm ≥ q60 (top 40% flattest).
    # Defensive: if Q60 wasn't computed, nothing passes.
    if V3_ENTROPY_Q60 is None:
        chalk_entropy_ok = pd.Series(False, index=df.index)
    else:
        chalk_entropy_ok = (df['race_entropy_norm'].notna()
                            & (df['race_entropy_norm'] >= V3_ENTROPY_Q60))

    # Longshot rule entropy gate: race_entropy_norm ≤ q40 (bottom 40% peakest).
    if V3_ENTROPY_Q40 is None:
        longshot_entropy_ok = pd.Series(False, index=df.index)
    else:
        longshot_entropy_ok = (df['race_entropy_norm'].notna()
                               & (df['race_entropy_norm'] <= V3_ENTROPY_Q40))

    # ml bands. Chalk ≤ 8, longshot ≥ 12. Mid-band (8 < ml < 12) is
    # excluded from both rules — diagnostic showed mid-ml was the worst
    # ROI bucket regardless of trigger.
    ml_chalk_ok    = df['ml_decimal'].notna() & (df['ml_decimal'] <= p3.ML_MAX_FOR_BET)
    ml_longshot_ok = df['ml_decimal'].notna() & (df['ml_decimal'] >= p3.ML_MIN_LONGSHOT)

    # Build per-rule fires
    chalk_rule_fires    = any_trigger & ml_chalk_ok    & chalk_entropy_ok
    longshot_rule_fires = any_trigger & ml_longshot_ok & longshot_entropy_ok

    # Per-trigger fires (within either rule). Useful for the existing
    # diagnostic scaffolding that slices by which model triggered.
    log_fires = log_bet & (chalk_rule_fires | longshot_rule_fires)
    rdg_fires = rdg_bet & (chalk_rule_fires | longshot_rule_fires)
    raw_fires = raw_bet & (chalk_rule_fires | longshot_rule_fires)

    df['v3_log_fires']           = log_fires
    df['v3_rdg_fires']           = rdg_fires
    df['v3_raw_fires']           = raw_fires
    df['v3_chalk_rule_fires']    = chalk_rule_fires
    df['v3_longshot_rule_fires'] = longshot_rule_fires
    df['v3_bet']                 = chalk_rule_fires | longshot_rule_fires

    # Forced bet type: LOG-fired → W (matches predict_v3 behavior).
    # All other triggered horses use predict_v2's standard W/WP/WPS
    # by ml. With longshot rule's ml ≥ 12 and ML_WP_MAX = 9, those
    # bets land in WPS naturally.
    df['forced_bet_type'] = np.where(log_fires, 'W', None)

    return df


def pick_top_per_race(df, rank_col, ascending=False):
    """For each race, pick the horse with highest (or lowest) rank_col."""
    if ascending:
        idx = df.groupby(['date', 'track', 'race'])[rank_col].idxmin()
    else:
        idx = df.groupby(['date', 'track', 'race'])[rank_col].idxmax()
    idx = idx.dropna()
    return df.loc[idx].copy()


def simulate_bet(row, takeout=WIN_TAKEOUT):
    """
    Return (cost, payout) for betting this horse using predict_v2's
    ML-odds-based bet type.

    cost = $2 for W, $4 for WP, $6 for WPS.
    payout = sum of win_paid + place_paid + show_paid components that
             correspond to the bet type. Each component is $0 unless
             the horse finished in that position.

    If the row has a non-null `forced_bet_type` column, it overrides
    the ML-based rule. Used by the v3 backtest, where LOG-triggered
    bets are W-only regardless of ML.
    """
    forced = row.get('forced_bet_type', None)
    if isinstance(forced, str) and forced in ('W', 'WP', 'WPS'):
        bet_type = forced
    else:
        ml_dec = row.get('ml_decimal', np.nan)
        # Default to WPS if ml_decimal is missing (most conservative assumption
        # for bet-type; matches predict_v2 fallback)
        if pd.isna(ml_dec):
            bet_type = 'WPS'
        elif ml_dec <= ML_WIN_ONLY_MAX:
            bet_type = 'W'
        elif ml_dec <= ML_WP_MAX:
            bet_type = 'WP'
        else:
            bet_type = 'WPS'

    wp = row.get('win_paid_num', 0) or 0
    pp = row.get('place_paid_num', 0) or 0
    sp = row.get('show_paid_num', 0) or 0

    if bet_type == 'W':
        return 2, wp, bet_type
    elif bet_type == 'WP':
        return 4, wp + pp, bet_type
    else:  # WPS
        return 6, wp + pp + sp, bet_type


def evaluate_system(picks_df, label, takeout=WIN_TAKEOUT):
    """
    Simulate the bets and return ROI metrics.

    ROI = (total_payout - total_cost) / total_cost
    takeout_cut = -takeout - ROI   (how much better than random)

    A random bettor expects ROI = -takeout. If your ROI > -takeout, you're
    cutting into the takeout. If ROI > 0, you've beaten the house entirely.
    """
    if picks_df.empty:
        return {
            'label': label, 'n_bets': 0, 'total_cost': 0,
            'total_payout': 0, 'roi': np.nan,
            'takeout_cut_pct': np.nan, 'win_rate': np.nan,
            'bet_type_breakdown': {},
        }

    total_cost = 0.0
    total_payout = 0.0
    n_wins = 0
    bet_type_stats = {'W': [0, 0, 0], 'WP': [0, 0, 0], 'WPS': [0, 0, 0]}
    # Each list: [n_bets, total_cost, total_payout]

    for _, row in picks_df.iterrows():
        cost, payout, bet_type = simulate_bet(row, takeout=takeout)
        total_cost += cost
        total_payout += payout
        if row.get('won') == 1:
            n_wins += 1
        bt = bet_type_stats[bet_type]
        bt[0] += 1
        bt[1] += cost
        bt[2] += payout

    n_bets = len(picks_df)
    roi = (total_payout - total_cost) / total_cost if total_cost > 0 else np.nan
    # takeout_cut_pct: positive = better than random, negative = worse.
    # Random baseline is -takeout (in the limit); being at -0.10 with 0.17
    # takeout means we cut 0.07 into the house edge.
    takeout_cut = roi - (-takeout) if total_cost > 0 else np.nan
    win_rate = n_wins / n_bets if n_bets > 0 else np.nan

    # Per-bet-type ROI
    bt_breakdown = {}
    for bt, (n, c, p) in bet_type_stats.items():
        if n > 0:
            bt_roi = (p - c) / c
            bt_breakdown[bt] = {'n': n, 'roi': bt_roi,
                               'cost': c, 'payout': p}

    return {
        'label': label, 'n_bets': n_bets, 'total_cost': total_cost,
        'total_payout': total_payout, 'roi': roi,
        'takeout_cut_pct': takeout_cut, 'win_rate': win_rate,
        'bet_type_breakdown': bt_breakdown,
    }


def compute_ensemble_rank(df):
    """
    Simple-average-rank ensemble:
      - For each race, rank horses by cv_prob, ridge_value_norm, raw_ridge_value
        (1 = best, N = worst)
      - Ensemble score = mean of the three ranks
      - Pick the horse with lowest ensemble rank per race

    If a signal is missing for a horse, that signal's rank is filled with
    the mean rank (= no information). This keeps missing-signal horses at
    a neutral position rather than dropping or overweighting them.
    """
    df = df.copy()
    rank_cols = []

    for col, out in [('cv_prob', 'log_rank'),
                     ('ridge_value_norm', 'rdg_rank'),
                     ('raw_ridge_value', 'raw_rank')]:
        if col in df.columns:
            # rank ascending=False: highest value gets rank 1
            df[out] = df.groupby(['date', 'track', 'race'])[col].rank(
                ascending=False, method='min', na_option='keep'
            )
            rank_cols.append(out)

    if not rank_cols:
        raise RuntimeError("No signals available for ensemble")

    # Fill missing with per-race mean rank (neutral)
    for col in rank_cols:
        df[col] = df.groupby(['date', 'track', 'race'])[col].transform(
            lambda s: s.fillna(s.mean())
        )

    df['ensemble_rank'] = df[rank_cols].mean(axis=1)
    return df, rank_cols


def print_roi_report(metrics_list, takeout=WIN_TAKEOUT, section_label=None):
    """Print side-by-side ROI comparison table.

    section_label: short string identifying which set of systems is
                   being reported (e.g. "predict_v2 by-trigger",
                   "predict_v3 entropy-filtered"). Appears in the
                   bet-type breakdown header so multiple calls in the
                   same run don't all show an identical label.
    """
    header_suffix = f" — {section_label}" if section_label else ""
    print(f"\n{'='*78}")
    print(f"BACKTEST RESULTS{header_suffix} — "
          f"Takeout baseline: random bettor ROI = {-takeout:+.1%}")
    print(f"{'='*78}")
    print(f"  {'System':<22s} {'n':>5s} {'cost':>7s} {'payout':>8s} "
          f"{'ROI':>8s} {'vs random':>10s} {'win%':>6s}")
    print(f"  {'-'*22} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

    for m in metrics_list:
        if m['n_bets'] == 0:
            print(f"  {m['label']:<22s} {'—':>5s} — no bets")
            continue
        roi_s = f"{m['roi']:+.1%}"
        cut_s = f"{m['takeout_cut_pct']:+.1%}"
        wr_s = f"{m['win_rate']:.1%}"
        print(f"  {m['label']:<22s} {m['n_bets']:>5d} "
              f"${m['total_cost']:>6.0f} ${m['total_payout']:>7.0f} "
              f"{roi_s:>8s} {cut_s:>10s} {wr_s:>6s}")

    # Bet-type decomposition for each system
    print(f"\n{'─'*78}")
    print(f"BET-TYPE BREAKDOWN{header_suffix} "
          f"(is any single bet type carrying the ROI?)")
    print(f"{'─'*78}")
    print(f"  {'System':<22s} {'W n/ROI':>14s} {'WP n/ROI':>14s} {'WPS n/ROI':>14s}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")
    for m in metrics_list:
        if m['n_bets'] == 0:
            continue
        parts = [f"  {m['label']:<22s}"]
        for bt in ['W', 'WP', 'WPS']:
            bd = m['bet_type_breakdown'].get(bt)
            if bd:
                parts.append(f"{bd['n']:>4d}/{bd['roi']:>+6.1%}    ")
            else:
                parts.append(f"{'—':>14s}")
        print(''.join(parts))


# ── ML-decimal bucket diagnostic ───────────────────────────────────────────
# Standard bucketing that aligns with predict_v2's bet types:
#   ≤3.5   (under 5/2):    tightest market, deep chalk
#   3.5-4.5 (5/2 to 9/2):  lukewarm favorites, still W-only in predict_v2
#   4.5-9.0 (9/2 to 9/1):  mid-range, WP territory
#   >9.0    (over 9/1):    longshots, WPS territory
#
# The diagnostic question is: do triggered bets cluster in one bucket?
# If RAW triggers are 80% in the ≤3.5 bucket, the "favorite tax" hypothesis
# is confirmed — we're betting horses the market has correctly priced tight.
ML_BUCKETS = [
    ('≤3.5', 0.0, 3.5),
    ('3.5-4.5', 3.5, 4.5),
    ('4.5-9.0', 4.5, 9.0),
    ('>9.0', 9.0, float('inf')),
]


def evaluate_by_ml_bucket(picks_df, label, takeout=WIN_TAKEOUT):
    """
    Segment bets by ml_decimal bucket and report ROI per bucket.

    Returns a list of (bucket_label, metrics) tuples in bucket order. Useful
    for diagnosing "are all my losses coming from favorites?" — if a trigger
    has -40% ROI overall but all its bets are in the ≤3.5 bucket, the
    problem is "bets favorites," not "bets bad horses in general."
    """
    results = []
    if picks_df.empty or 'ml_decimal' not in picks_df.columns:
        return results

    for bucket_label, lo, hi in ML_BUCKETS:
        bucket_picks = picks_df[
            (picks_df['ml_decimal'] > lo) &
            (picks_df['ml_decimal'] <= hi)
        ]
        m = evaluate_system(bucket_picks, f"{label} [{bucket_label}]",
                           takeout=takeout)
        results.append((bucket_label, m))

    return results


def print_bucket_diagnostic(label, picks_df, takeout=WIN_TAKEOUT):
    """
    Print a ML-decimal bucket breakdown for one system's bets.

    Shows n and ROI per bucket so you can see which odds range is driving
    the overall result. "↑ favorite tax" marker appears when favorites
    (≤4.5 ml_decimal combined) carry the losses.
    """
    buckets = evaluate_by_ml_bucket(picks_df, label, takeout=takeout)
    if not buckets:
        return

    print(f"\n  {label}:")
    print(f"    {'bucket':<10s} {'n':>4s} {'cost':>6s} {'payout':>7s} "
          f"{'ROI':>8s} {'vs rand':>8s}")
    for bucket_label, m in buckets:
        if m['n_bets'] == 0:
            print(f"    {bucket_label:<10s} {'—':>4s}")
            continue
        print(f"    {bucket_label:<10s} {m['n_bets']:>4d} "
              f"${m['total_cost']:>5.0f} ${m['total_payout']:>6.0f} "
              f"{m['roi']:>+7.1%} {m['takeout_cut_pct']:>+7.1%}")

    # Interpretation flag: if the ≤4.5 bucket combined has >60% of bets
    # AND worse-than-overall ROI, flag it as a likely favorite-tax issue
    fav_n = sum(m['n_bets'] for (bl, m) in buckets if bl in ('≤3.5', '3.5-4.5'))
    total_n = sum(m['n_bets'] for (_, m) in buckets)
    if total_n > 0 and fav_n / total_n > 0.60:
        fav_cost = sum(m['total_cost'] for (bl, m) in buckets
                      if bl in ('≤3.5', '3.5-4.5'))
        fav_pay = sum(m['total_payout'] for (bl, m) in buckets
                     if bl in ('≤3.5', '3.5-4.5'))
        fav_roi = (fav_pay - fav_cost) / fav_cost if fav_cost > 0 else 0
        print(f"    ↑ {fav_n}/{total_n} bets ({fav_n/total_n:.0%}) are on "
              f"favorites (ml≤4.5); favorite-only ROI = {fav_roi:+.1%}")


def _render_entropy_ml_table(slice_df, ent_col, ent_buckets,
                              ml_buckets, takeout):
    """
    Render a (entropy × ml) ROI table for the given slice. Both bucket
    lists are pre-computed by the caller, so the table size is
    determined by the caller — typically 3×3 (entropy thirds × ml
    thirds) or 3×5 (entropy thirds × ml quintiles).

    Both bucket lists shared across slices means cells line up and
    can be compared row-for-row across slices (LOG vs RDG vs ALL).

    Pure rendering — no boundary computation, no I/O beyond print.
    Returns nothing.

    Column widths auto-adapt: narrower columns when there are more
    ml buckets so the table fits in a reasonable width.
    """
    # Auto-adapt column width: 3 ml buckets → 14 wide, more → narrower
    col_w = 14 if len(ml_buckets) <= 3 else 11
    ent_lbl_w = 24

    if slice_df.empty:
        print(f"  (empty slice — no bets)")
        return

    header = f"  {'entropy bucket':<{ent_lbl_w}s}"
    for ml_lbl, _, _ in ml_buckets:
        header += f" {ml_lbl:>{col_w}s}"
    header += f" {'all (row)':>{col_w}s}"
    print(header)
    print(f"  {'-'*ent_lbl_w}" + (f" {'-'*col_w}" * (len(ml_buckets) + 1)))

    for ent_lbl, ent_lo, ent_hi in ent_buckets:
        in_ent = slice_df[
            slice_df[ent_col].notna()
            & (slice_df[ent_col] >  ent_lo)
            & (slice_df[ent_col] <= ent_hi)
        ]
        line = f"  {ent_lbl:<{ent_lbl_w}s}"
        if in_ent.empty:
            line += (f" {'—':>{col_w}s}" * (len(ml_buckets) + 1))
            print(line)
            continue

        for ml_lbl, ml_lo, ml_hi in ml_buckets:
            cell = in_ent[
                (in_ent['ml_decimal'] >  ml_lo)
                & (in_ent['ml_decimal'] <= ml_hi)
            ]
            if cell.empty:
                line += f" {'—':>{col_w}s}"
            else:
                m = evaluate_system(cell, '_cell', takeout=takeout)
                cell_str = f"{m['n_bets']}/{m['roi']:+.0%}"
                line += f" {cell_str:>{col_w}s}"
        row_m = evaluate_system(in_ent, '_row', takeout=takeout)
        row_str = f"{row_m['n_bets']}/{row_m['roi']:+.0%}"
        line += f" {row_str:>{col_w}s}"
        print(line)

    # Column totals
    col_line = f"  {'all (col)':<{ent_lbl_w}s}"
    for ml_lbl, ml_lo, ml_hi in ml_buckets:
        cell = slice_df[
            (slice_df['ml_decimal'] >  ml_lo)
            & (slice_df['ml_decimal'] <= ml_hi)
        ]
        if cell.empty:
            col_line += f" {'—':>{col_w}s}"
        else:
            m = evaluate_system(cell, '_col', takeout=takeout)
            cell_str = f"{m['n_bets']}/{m['roi']:+.0%}"
            col_line += f" {cell_str:>{col_w}s}"
    gm = evaluate_system(slice_df, '_grand', takeout=takeout)
    grand_str = f"{gm['n_bets']}/{gm['roi']:+.0%}"
    col_line += f" {grand_str:>{col_w}s}"
    print(col_line)


def print_entropy_x_ml_bucket_diagnostic(picks_df, takeout=WIN_TAKEOUT):
    """
    3×3 bucket diagnostic: ROI by (entropy third × ml_decimal third),
    sliced by trigger type.

    Both axes use dynamic quantile thirds (q33, q67) computed from
    picks_df (the full bet pool), so each row and column gets ~1/3
    of the bets and no cell starves on sample size. Bucket labels
    include the boundary values so the table is readable without
    a separate key.

    For each entropy axis (ML / cv-normalized / cv-raw), four slices
    are printed using the SAME bucket boundaries:
      1. ALL BETS         (predict_v2 union, every horse that triggered)
      2. LOG TRIGGERED    (rows where log_bet=True, may also be rdg/raw)
      3. RDG TRIGGERED    (rows where rdg_bet=True, may also be log/raw)
      4. RAW TRIGGERED    (rows where raw_bet=True, may also be log/rdg)

    Slices 2-4 OVERLAP on horses where multiple triggers fired. That's
    intentional — it gives each trigger more cell-N to read than
    exclusive slicing would, and the question being answered ("does
    this trigger work in this corner of the table?") doesn't require
    the slices be disjoint. The cost is that aggregate ROIs across
    slices don't sum to the all-bets aggregate.

    Why three entropy axes: the v3 gate uses normalized cv entropy
    (model's own confidence). ML entropy (market structure) and raw
    cv entropy (field-size-dependent) are computed too, in case
    comparing them surfaces a regime where market and model
    disagree on race shape.

    Watching for:
      - Trigger × cell asymmetry: e.g. LOG positive in (flattest,
        low-ml) while RDG is positive in (peakest, high-ml) would
        suggest the triggers should be applied in different regimes.
      - Monotonic trends across rows/columns: trustworthier than
        single-cell numbers.

    Args:
        picks_df: DataFrame of bets to bucket. Must include
                  race_entropy_ml, race_entropy_norm, race_entropy_raw,
                  ml_decimal, log_bet, rdg_bet, plus the bet/payout
                  columns evaluate_system needs.
        takeout: passed to evaluate_system for ROI deltas.
    """
    if picks_df.empty:
        print("\n  (no bets to bucket)")
        return

    # ml-decimal thirds — computed from the bet rows themselves so
    # each column gets ~1/3 of bets. Drop NaN ml first.
    ml_series = picks_df['ml_decimal'].dropna()
    if len(ml_series) < 9:
        print(f"\n  (only {len(ml_series)} bets with ml_decimal — too few for 3×3)")
        return
    ml_q33 = float(ml_series.quantile(1/3))
    ml_q67 = float(ml_series.quantile(2/3))
    ml_buckets = [
        (f'low ml (≤{ml_q33:.2f})',           -np.inf, ml_q33),
        (f'mid ml ({ml_q33:.2f}-{ml_q67:.2f})', ml_q33, ml_q67),
        (f'high ml (>{ml_q67:.2f})',          ml_q67,  np.inf),
    ]

    entropy_axes = [
        ('race_entropy_ml',   'ML ENTROPY (1/ml_decimal — market)'),
        ('race_entropy_norm', 'NORMALIZED CV ENTROPY (cv_prob/log(N) — model)'),
        ('race_entropy_raw',  'RAW CV ENTROPY (cv_prob in nats — field-size-dependent)'),
    ]

    # Precompute trigger slices once — the entropy axes loop reuses them.
    # Use .get with a default Series so missing columns don't crash.
    log_mask = picks_df.get('log_bet', pd.Series(False, index=picks_df.index))
    rdg_mask = picks_df.get('rdg_bet', pd.Series(False, index=picks_df.index))
    raw_mask = picks_df.get('raw_bet', pd.Series(False, index=picks_df.index))
    slices = [
        ('ALL BETS',      picks_df),
        ('LOG TRIGGERED', picks_df[log_mask.fillna(False)]),
        ('RDG TRIGGERED', picks_df[rdg_mask.fillna(False)]),
        ('RAW TRIGGERED', picks_df[raw_mask.fillna(False)]),
    ]

    for ent_col, label in entropy_axes:
        if ent_col not in picks_df.columns:
            print(f"\n  ({ent_col} missing — skipping)")
            continue

        # Use one row per race so large-field races aren't overweighted
        race_ent = (picks_df.drop_duplicates(['date', 'track', 'race'])
                            [ent_col].dropna())
        if len(race_ent) < 9:
            print(f"\n  {label}: too few races ({len(race_ent)}) for 3×3 bucketing")
            continue

        ent_q33 = float(race_ent.quantile(1/3))
        ent_q67 = float(race_ent.quantile(2/3))
        ent_buckets = [
            (f'peakest (≤{ent_q33:.3f})',          -np.inf, ent_q33),
            (f'mid ({ent_q33:.3f}-{ent_q67:.3f})',  ent_q33, ent_q67),
            (f'flattest (>{ent_q67:.3f})',          ent_q67, np.inf),
        ]

        print(f"\n{'═'*78}")
        print(f"  {label}")
        print(f"{'═'*78}")

        for slice_name, slice_df in slices:
            print(f"\n  ── {slice_name} ──")
            _render_entropy_ml_table(slice_df, ent_col, ent_buckets,
                                      ml_buckets, takeout)

    print(f"\n  Read this table as: cells = n_bets/ROI. Each row and column")
    print(f"  in the ALL BETS slice holds ~1/3 of bets by construction; the")
    print(f"  trigger slices use the SAME bucket boundaries (so cells across")
    print(f"  slices line up) but their counts are smaller. LOG, RDG, and")
    print(f"  RAW slices OVERLAP on horses where multiple triggers fired.")
    print(f"  Look for monotonic trends across rows/columns rather than")
    print(f"  reading single cells. The three entropy axes share the same")
    print(f"  ml-thirds, so column totals match across axes — only row")
    print(f"  assignments differ.")


def print_entropy_x_ml_3x5_diagnostic(picks_df, takeout=WIN_TAKEOUT):
    """
    3×5 bucket diagnostic: entropy thirds × ml-decimal QUINTILES.

    Same shape as print_entropy_x_ml_bucket_diagnostic but with 5
    ml columns instead of 3, for finer-grained granularity on the
    odds axis. Entropy stays at thirds because (a) entropy effects
    tend to be lower-resolution than odds effects, (b) we don't
    need 25 cells × 3 slices × 3 axes = 225 cells of noise.

    Tradeoff: 5 columns means smaller cells (~1/5 of bets per
    column instead of 1/3). For a 275-bet sample this is ~55
    bets per ALL BETS column, ~45 per RDG column. Sliced into
    entropy thirds, individual cells run 10-20 bets — read
    trends, not cells.

    Pre-registered prior context: this diagnostic was built to
    examine a specific hypothesis — that "favorites in low-entropy
    races" and "longshots in high-entropy races" are bad bets, and
    the opposite combinations (longshots in low-entropy, favorites
    in high-entropy) might be value. That's a diagonal-corner
    pattern. The 5-bin version exposes the corners more cleanly
    than 3-bin since "favorite" and "longshot" are more naturally
    quintile-1 and quintile-5 than third-1 and third-3.

    Same trigger slicing as the 3×3 version (ALL/LOG/RDG/RAW, with
    LOG, RDG, and RAW overlapping on multi-trigger bets).

    Args:
        picks_df: same requirements as print_entropy_x_ml_bucket_diagnostic
        takeout: passed through to evaluate_system
    """
    if picks_df.empty:
        print("\n  (no bets to bucket)")
        return

    # ml-decimal quintiles
    ml_series = picks_df['ml_decimal'].dropna()
    if len(ml_series) < 15:  # need at least 15 to make 5 columns meaningful
        print(f"\n  (only {len(ml_series)} bets with ml_decimal — too few for 3×5)")
        return
    ml_q20 = float(ml_series.quantile(0.20))
    ml_q40 = float(ml_series.quantile(0.40))
    ml_q60 = float(ml_series.quantile(0.60))
    ml_q80 = float(ml_series.quantile(0.80))
    ml_buckets = [
        (f'q1 (≤{ml_q20:.2f})',           -np.inf, ml_q20),
        (f'q2 ({ml_q20:.2f}-{ml_q40:.2f})', ml_q20, ml_q40),
        (f'q3 ({ml_q40:.2f}-{ml_q60:.2f})', ml_q40, ml_q60),
        (f'q4 ({ml_q60:.2f}-{ml_q80:.2f})', ml_q60, ml_q80),
        (f'q5 (>{ml_q80:.2f})',           ml_q80,  np.inf),
    ]

    entropy_axes = [
        ('race_entropy_ml',   'ML ENTROPY (1/ml_decimal — market)'),
        ('race_entropy_norm', 'NORMALIZED CV ENTROPY (cv_prob/log(N) — model)'),
        ('race_entropy_raw',  'RAW CV ENTROPY (cv_prob in nats — field-size-dependent)'),
    ]

    log_mask = picks_df.get('log_bet', pd.Series(False, index=picks_df.index))
    rdg_mask = picks_df.get('rdg_bet', pd.Series(False, index=picks_df.index))
    raw_mask = picks_df.get('raw_bet', pd.Series(False, index=picks_df.index))
    slices = [
        ('ALL BETS',      picks_df),
        ('LOG TRIGGERED', picks_df[log_mask.fillna(False)]),
        ('RDG TRIGGERED', picks_df[rdg_mask.fillna(False)]),
        ('RAW TRIGGERED', picks_df[raw_mask.fillna(False)]),
    ]

    for ent_col, label in entropy_axes:
        if ent_col not in picks_df.columns:
            print(f"\n  ({ent_col} missing — skipping)")
            continue

        race_ent = (picks_df.drop_duplicates(['date', 'track', 'race'])
                            [ent_col].dropna())
        if len(race_ent) < 9:
            print(f"\n  {label}: too few races ({len(race_ent)}) for entropy thirds")
            continue

        ent_q33 = float(race_ent.quantile(1/3))
        ent_q67 = float(race_ent.quantile(2/3))
        ent_buckets = [
            (f'peakest (≤{ent_q33:.3f})',          -np.inf, ent_q33),
            (f'mid ({ent_q33:.3f}-{ent_q67:.3f})',  ent_q33, ent_q67),
            (f'flattest (>{ent_q67:.3f})',          ent_q67, np.inf),
        ]

        print(f"\n{'═'*78}")
        print(f"  {label}")
        print(f"{'═'*78}")

        for slice_name, slice_df in slices:
            print(f"\n  ── {slice_name} ──")
            _render_entropy_ml_table(slice_df, ent_col, ent_buckets,
                                      ml_buckets, takeout)

    print(f"\n  Read this 3×5 table as: cells = n_bets/ROI. Watch the")
    print(f"  diagonal corners — (peakest, q1=favorites) and (flattest, q5=")
    print(f"  longshots) should be bad bets per the prior hypothesis;")
    print(f"  (peakest, q5=longshots) and (flattest, q1=favorites) are the")
    print(f"  candidate value cells. Cell sizes are smaller than 3×3 so")
    print(f"  read row/column trends, not single cells.")





def _run_exacta_box_backtest(rdg_pool, scope_label, takeout=WIN_TAKEOUT):
    """
    Run an exacta-box backtest over a pool of RDG-firing horses.

    The pool is filtered upstream by the caller — this function just
    consumes whatever races/horses are passed in, groups them by
    race, and boxes the qualifying ones (per Blake's box rules:
    2-horse → $4 box, 3-horse → $12 box, 4+ skipped).

    Args:
        rdg_pool: DataFrame of RDG-firing horses. Must contain
                  date, track, race, horse_number columns. Caller
                  controls scope (RDG-only vs RDG-incl-LOG, etc).
        scope_label: human-readable label for this pool, printed
                     in the section header.
        takeout: passed to evaluate_system for the singles comparison.

    Returns: dict with keys
        scope_label, n_races_total, n_2horse, n_3horse, n_4plus,
        n_evaluable, n_misses_total, exb_df (DataFrame of box rows
        or None), aggregate {n_bets, hits, hit_rate, cost, payout, roi},
        per_size {2: {...}, 3: {...}}, singles_metric (dict from
        evaluate_system on same-race singles, or None).

    The dict is consumed by _print_exacta_scope_comparison for the
    side-by-side at the end.
    """
    print(f"\n{'-'*78}")
    print(f"  EXACTA BOX backtest — {scope_label}")
    print(f"{'-'*78}")

    if len(rdg_pool) == 0:
        print(f"  Empty pool — nothing to box.")
        return {
            'scope_label': scope_label,
            'n_races_total': 0, 'n_2horse': 0, 'n_3horse': 0, 'n_4plus': 0,
            'n_evaluable': 0, 'n_misses_total': 0, 'exb_df': None,
            'aggregate': None, 'per_size': {}, 'singles_metric': None,
        }

    races_with_rdg = rdg_pool.groupby(['date', 'track', 'race']).size()
    multi_race_keys = races_with_rdg[races_with_rdg >= 2].index.tolist()
    n_2horse_races = (races_with_rdg == 2).sum()
    n_3horse_races = (races_with_rdg == 3).sum()
    n_4plus = (races_with_rdg >= 4).sum()

    print(f"  Race counts: {n_2horse_races} 2-horse, "
          f"{n_3horse_races} 3-horse, {n_4plus} 4+ (skipped)")

    exb_rows = []
    misses_file = []
    misses_race = []
    misses_exacta = []
    for (date, track, race) in multi_race_keys:
        triggered = rdg_pool[
            (rdg_pool['date'] == date)
            & (rdg_pool['track'] == track)
            & (rdg_pool['race'] == race)
        ]
        horses = sorted(triggered['horse_number'].astype(int).tolist())
        n_h = len(horses)
        if n_h not in (2, 3):
            continue  # 4+ skipped per rule

        # File-vs-race-vs-exacta miss attribution (matches original).
        cache_key = (track, date)
        if cache_key not in _META_CACHE:
            loaded = None
            for path in _meta_path_candidates(track, date):
                if path.exists():
                    try:
                        loaded = pd.read_csv(path)
                        break
                    except Exception as e:
                        print(f"  WARN: failed to load {path}: {e}")
            _META_CACHE[cache_key] = loaded

        meta = _META_CACHE[cache_key]
        if meta is None:
            misses_file.append((date, track, race))
            continue

        row = meta[meta['race'] == race]
        if row.empty:
            misses_race.append((date, track, race))
            continue

        exotics = load_race_exotics(track, date, race)
        if exotics is None or exotics['exacta_finish'] is None:
            misses_exacta.append((date, track, race))
            continue

        cost = 4 if n_h == 2 else 12
        actual_top2 = set(exotics['exacta_finish'][:2])
        horse_set = set(horses)
        hit = actual_top2.issubset(horse_set)
        payout = exotics['exacta_payout'] if hit else 0.0

        exb_rows.append({
            'date': date, 'track': track, 'race': race,
            'n_horses': n_h, 'horses': horses,
            'cost': cost, 'payout': payout, 'hit': hit,
            'exacta_finish': exotics['exacta_finish'],
            'exacta_payout_on_hit': exotics['exacta_payout'],
        })

    n_misses_total = len(misses_file) + len(misses_race) + len(misses_exacta)
    if n_misses_total > 0:
        print(f"\n  Missing data breakdown ({n_misses_total} multi-trigger "
              f"races skipped):")
        if misses_file:
            cards = sorted(set((d, t) for d, t, _ in misses_file))
            print(f"    {len(misses_file)} race(s) blocked by "
                  f"{len(cards)} missing meta file(s):")
            for d, t in cards:
                n_blocked = sum(1 for fd, ft, _ in misses_file
                                if fd == d and ft == t)
                print(f"      {d} / {t}  ({n_blocked} race(s))")
        if misses_race:
            print(f"    {len(misses_race)} race(s) where meta exists "
                  f"but race row is missing:")
            for d, t, r in sorted(misses_race):
                print(f"      {d} / {t}  R{r}")
        if misses_exacta:
            print(f"    {len(misses_exacta)} race(s) where meta exists "
                  f"but no exacta_finish data:")
            for d, t, r in sorted(misses_exacta):
                print(f"      {d} / {t}  R{r}")

    result = {
        'scope_label': scope_label,
        'n_races_total': len(races_with_rdg),
        'n_2horse': int(n_2horse_races), 'n_3horse': int(n_3horse_races),
        'n_4plus': int(n_4plus),
        'n_evaluable': len(exb_rows),
        'n_misses_total': n_misses_total,
        'exb_df': None,
        'aggregate': None, 'per_size': {}, 'singles_metric': None,
    }

    if not exb_rows:
        print(f"\n  No evaluable multi-trigger races.")
        return result

    exb_df = pd.DataFrame(exb_rows)
    n = len(exb_df)
    hits = int(exb_df['hit'].sum())
    cost = float(exb_df['cost'].sum())
    payout = float(exb_df['payout'].sum())
    roi = (payout - cost) / cost if cost > 0 else 0
    hit_rate = hits / n
    result['exb_df'] = exb_df
    result['aggregate'] = {
        'n_bets': n, 'hits': hits, 'hit_rate': hit_rate,
        'cost': cost, 'payout': payout, 'roi': roi,
    }

    print(f"\n  Aggregate ({n} boxes, ${cost:.0f} risked):")
    print(f"    Hits: {hits} ({hit_rate:.1%})")
    print(f"    Payout: ${payout:.0f}  Profit: ${payout - cost:+.0f}  "
          f"ROI: {roi:+.1%}")

    for n_h, label in [(2, '2-horse'), (3, '3-horse')]:
        sub = exb_df[exb_df['n_horses'] == n_h]
        if sub.empty:
            continue
        n_s = len(sub)
        hits_s = int(sub['hit'].sum())
        cost_s = float(sub['cost'].sum())
        payout_s = float(sub['payout'].sum())
        roi_s = (payout_s - cost_s) / cost_s if cost_s > 0 else 0
        result['per_size'][n_h] = {
            'n_bets': n_s, 'hits': hits_s, 'hit_rate': hits_s / n_s,
            'cost': cost_s, 'payout': payout_s, 'roi': roi_s,
        }
        print(f"  {label} boxes: {n_s} bets, {hits_s} hits "
              f"({hits_s/n_s:.1%}), "
              f"cost ${cost_s:.0f}, payout ${payout_s:.0f}, "
              f"ROI {roi_s:+.1%}")

    print(f"\n  Hit details (for inspection):")
    print(f"    {'date':<11} {'track':<18} {'race':>4} "
          f"{'box':<14} {'finish':<8} {'paid':>8} {'cost':>5}")
    hits_df = exb_df[exb_df['hit']].sort_values('payout', ascending=False)
    for _, r in hits_df.iterrows():
        box_str = '/'.join(str(h) for h in r['horses'])
        fin_str = '-'.join(str(h) for h in r['exacta_finish'])
        print(f"    {r['date']:<11} {r['track']:<18} "
              f"{r['race']:>4} {box_str:<14} {fin_str:<8} "
              f"${r['payout']:>6.2f} ${r['cost']:>3}")

    # Singles comparison on the same race subset
    multi_keys_df = pd.DataFrame(multi_race_keys,
                                 columns=['date', 'track', 'race'])
    multi_singles = rdg_pool.merge(multi_keys_df,
                                   on=['date', 'track', 'race'])
    m_multi_singles = evaluate_system(
        multi_singles,
        "Singles in multi-trigger races (W/WP/WPS by ML)",
        takeout=takeout)
    result['singles_metric'] = m_multi_singles
    print(f"\n  Comparison — same races, singles strategy:")
    print(f"    n={m_multi_singles['n_bets']}, "
          f"cost ${m_multi_singles['total_cost']:.0f}, "
          f"payout ${m_multi_singles['total_payout']:.0f}, "
          f"ROI {m_multi_singles['roi']:+.1%}")
    print(f"    (this is what we currently bet on these races)")

    return result


def _print_exacta_scope_comparison(result_a, result_b):
    """
    Side-by-side comparison of two exacta-box scopes.

    Designed to answer: "if I extend the exacta box to LOG-triggered
    races in predict/daily-summary, would the backtest get worse?"

    Prints the two aggregates, the delta in each metric, and a
    breakdown of the marginal races (those in B but not in A).
    """
    print(f"\n{'='*78}")
    print(f"EXACTA BOX SCOPE COMPARISON")
    print(f"  A = {result_a['scope_label']}")
    print(f"  B = {result_b['scope_label']}")
    print(f"{'='*78}")

    agg_a = result_a.get('aggregate')
    agg_b = result_b.get('aggregate')
    if not agg_a or not agg_b:
        print(f"  One or both pools had no evaluable boxes — comparison "
              f"not meaningful.")
        if agg_a:
            print(f"  A aggregate: {agg_a}")
        if agg_b:
            print(f"  B aggregate: {agg_b}")
        return

    print(f"  {'metric':<14} {'A':>12} {'B':>12} {'Δ (B-A)':>12}")
    print(f"  {'-'*14} {'-'*12:>12} {'-'*12:>12} {'-'*12:>12}")
    print(f"  {'n_boxes':<14} {agg_a['n_bets']:>12d} {agg_b['n_bets']:>12d} "
          f"{agg_b['n_bets'] - agg_a['n_bets']:>+12d}")
    print(f"  {'hits':<14} {agg_a['hits']:>12d} {agg_b['hits']:>12d} "
          f"{agg_b['hits'] - agg_a['hits']:>+12d}")
    print(f"  {'hit_rate':<14} {agg_a['hit_rate']:>11.1%} "
          f"{agg_b['hit_rate']:>11.1%} "
          f"{agg_b['hit_rate'] - agg_a['hit_rate']:>+11.1%}")
    print(f"  {'cost':<14} {('$%.0f' % agg_a['cost']):>12s} "
          f"{('$%.0f' % agg_b['cost']):>12s} "
          f"{('$%+.0f' % (agg_b['cost'] - agg_a['cost'])):>12s}")
    print(f"  {'payout':<14} {('$%.0f' % agg_a['payout']):>12s} "
          f"{('$%.0f' % agg_b['payout']):>12s} "
          f"{('$%+.0f' % (agg_b['payout'] - agg_a['payout'])):>12s}")
    print(f"  {'roi':<14} {agg_a['roi']:>+11.1%} {agg_b['roi']:>+11.1%} "
          f"{agg_b['roi'] - agg_a['roi']:>+11.1%}")

    # Marginal-races breakdown: races in B but not in A.
    # These are the LOG-triggered races we'd add if we extend predict/
    # daily-summary. Their isolated ROI is the cleanest answer to
    # "is it safe to add them?"
    if (result_a.get('exb_df') is not None
            and result_b.get('exb_df') is not None):
        a_keys = set(zip(result_a['exb_df']['date'],
                         result_a['exb_df']['track'],
                         result_a['exb_df']['race']))
        b_df = result_b['exb_df']
        b_keys = set(zip(b_df['date'], b_df['track'], b_df['race']))
        marginal_keys = b_keys - a_keys
        if marginal_keys:
            marg = b_df[b_df.apply(
                lambda r: (r['date'], r['track'], r['race']) in marginal_keys,
                axis=1
            )]
            n_m = len(marg)
            hits_m = int(marg['hit'].sum())
            cost_m = float(marg['cost'].sum())
            pay_m = float(marg['payout'].sum())
            roi_m = (pay_m - cost_m) / cost_m if cost_m > 0 else 0
            print(f"\n  Marginal races (in B but not A — the LOG-triggered "
                  f"adds):")
            print(f"    {n_m} boxes, {hits_m} hits ({hits_m/n_m:.1%}), "
                  f"${cost_m:.0f} cost, ${pay_m:.0f} payout, "
                  f"ROI {roi_m:+.1%}")
            print(f"    → if this ROI is similar to or better than A's, "
                  f"safe to extend.")
            print(f"    → if much worse, LOG-triggered races shouldn't "
                  f"get exacta boxes.")
        else:
            print(f"\n  No marginal races — scopes A and B have identical "
                  f"evaluable race sets.")


def run_backtest(takeout=WIN_TAKEOUT, use_meta_model=False, source=None):
    """Main backtest runner."""
    if use_meta_model:
        print("WARNING: --meta-model not yet implemented, falling back to "
              "simple-average-rank ensemble")

    df = load_cv_predictions(source=source)

    # --- Reconstruct the signals predict_v2 uses ---
    # Compute ev_per_2 from cv_prob + ml_decimal, using the correct
    # takeout-adjusted formula. If ev_per_2 was already written by
    # predict_v2 (walk_forward source), it's left untouched.
    df = compute_ev_per_2(df, takeout=takeout)

    # Normalize log-ridge per race (matches predict_v2's apply_ridge_value_model)
    if 'ridge_value' in df.columns:
        df = apply_ridge_race_norm(df, 'ridge_value',
                                  race_total=RIDGE_RACE_TOTAL,
                                  floor=RIDGE_FLOOR)

    # Raw-ridge calibration. Two modes:
    #   per-row (walk-forward): each row has raw_cv_mean/std stamped by
    #     predict_v2, reflecting the JSON active when that card was scored
    #   pooled (cv source, or old walk-forward files without stamps):
    #     one μ/σ across all predictions. Less accurate but a reasonable
    #     fallback.
    if 'raw_ridge_value' in df.columns:
        raw_mean, raw_std = compute_raw_calibration(df, 'raw_ridge_value')
        has_per_row = ('raw_cv_mean' in df.columns and
                      'raw_cv_std' in df.columns and
                      df['raw_cv_mean'].notna().any())
        if has_per_row:
            # Distinct stamped values tell us how many different calibrations
            # the backtest is respecting
            unique_stamps = df[['raw_cv_mean', 'raw_cv_std']].drop_duplicates()
            print(f"\nRaw-ridge calibration: {len(unique_stamps)} distinct per-card stamps "
                  f"(predict_v2 wrote them at prediction time)")
            print(f"  Pooled fallback: μ={raw_mean:.3f}, σ={raw_std:.3f} "
                  f"(used only if per-row stamp is null)")
        else:
            print(f"\nRaw-ridge calibration (POOLED from loaded CV preds): "
                  f"μ={raw_mean:.3f}, σ={raw_std:.3f}")
            print(f"  Bet threshold (μ+{RAW_EDGE_K}σ) = "
                  f"{raw_mean + RAW_EDGE_K*raw_std:.3f}")
    else:
        raw_mean, raw_std = np.nan, np.nan

    # Apply BET/AGREE/veto flags (predict_v2's trigger logic)
    df = compute_bet_flags(df, raw_mean, raw_std)

    # --- Build ensemble rank ---
    df, rank_cols = compute_ensemble_rank(df)
    print(f"\nEnsemble uses {len(rank_cols)} signals: {rank_cols}")

    # all_metrics retained only for the interpretation summary's
    # legacy structure; we no longer populate it with top-pick-every-race
    # systems since those weren't useful for evaluating predict_v2.
    all_metrics = []

    # --- Filtered subsets: bet EVERY horse that predict_v2 would bet ---
    # The pick-top-per-race formulation above is useful for comparing raw
    # signal quality, but it's NOT what predict_v2 does at bet time.
    # predict_v2's generate_bets() considers every horse in every race and
    # bets any horse that:
    #   - clears at least one model's BET edge threshold
    #   - is not floor-vetoed by any active model
    #
    # So "BET only" here means "every horse predict_v2 would bet on this
    # card," not "the top-ranked horse per race, filtered to triggers."
    # This matches the actual production bet rate (~4 bets per card).
    print(f"\n{'='*78}")
    print(f"PREDICT_V2-STYLE BETS (every horse that clears trigger logic)")
    print(f"  These are the actual bets predict_v2 would recommend. Each")
    print(f"  horse scored independently — a race can produce 0, 1, or many bets.")
    print(f"{'='*78}")

    # All qualifying horses, regardless of whether they're anyone's top pick.
    all_bets = df[df['any_bet']].copy()
    agree_bets = df[df['any_bet'] & df['is_agree']].copy()

    # Further slice by which model(s) triggered — mirrors predict_v2's
    # triggered_by column. Lets us compare ROI across trigger types.
    #   - LOG-triggered: log_bet fired (maybe also others)
    #   - RDG-triggered: rdg_bet fired
    #   - RAW-triggered: raw_bet fired
    # A single bet can appear in multiple categories if multiple models
    # triggered. The "LOG+RDG" intersection is particularly useful —
    # those are bets where two model types agree at the BET level.
    filter_metrics = []

    # All bets (predict_v2 would place every one of these)
    m_all = evaluate_system(all_bets, "All predict_v2 bets", takeout=takeout)
    filter_metrics.append(m_all)

    # AGREE subset: ≥2 of 3 models positive (not just trigger-level)
    m_agree = evaluate_system(agree_bets, "AGREE only (≥2 positive)",
                             takeout=takeout)
    filter_metrics.append(m_agree)

    # By-trigger slices — which model's edge logic is actually profitable?
    # Overlap is fine; these are descriptive cuts of the same pool.
    if 'log_bet' in df.columns:
        m_log = evaluate_system(df[df['any_bet'] & df['log_bet']],
                               "  └ LOG triggered", takeout=takeout)
        filter_metrics.append(m_log)
    if 'rdg_bet' in df.columns:
        m_rdg = evaluate_system(df[df['any_bet'] & df['rdg_bet']],
                               "  └ RDG triggered", takeout=takeout)
        filter_metrics.append(m_rdg)
    if 'raw_bet' in df.columns:
        m_raw = evaluate_system(df[df['any_bet'] & df['raw_bet']],
                               "  └ RAW triggered", takeout=takeout)
        filter_metrics.append(m_raw)

    # Multi-model-agreement-at-BET-level (strict): horses where 2+ models
    # all exceed their EDGE threshold, not just pos-level break-even. This
    # is a tighter subset than AGREE (which uses pos-level).
    n_bets_triggered = (
        df.get('log_bet', False).astype(int) +
        df.get('rdg_bet', False).astype(int) +
        df.get('raw_bet', False).astype(int)
    )
    df['_multi_trigger'] = n_bets_triggered >= 2
    m_multi = evaluate_system(df[df['any_bet'] & df['_multi_trigger']],
                             "Multi-trigger (≥2 at edge)", takeout=takeout)
    filter_metrics.append(m_multi)

    print_roi_report(filter_metrics, takeout=takeout,
                     section_label="predict_v2 by-trigger")

    # --- predict_v3 (entropy-filtered) ---
    # IMPORTANT: the v3 filter is in-sample. Even with a dynamic
    # entropy threshold (which avoids the "tuned on this data" problem
    # of fixed gini cutoffs), the ROI numbers below are upper bounds
    # on prospective performance because every other piece of the
    # pipeline (model fits, calibration, bet flags) was fit on this
    # same K-fold data. Use this output for:
    #   (a) sanity-checking that the v3 implementation behaves sensibly
    #   (b) measuring the v3 vs v2 delta on the same data (the delta
    #       is a less-biased measure of what the entropy gate adds)
    if HAS_V3:
        df = compute_v3_filters(df)
        v3_metrics = []

        v3_all = evaluate_system(df[df['v3_bet']],
                                 "All predict_v3 bets",
                                 takeout=takeout)
        v3_metrics.append(v3_all)

        # Slice by rule (chalk vs longshot). Rules are disjoint by ml
        # band so no double-counting between these two.
        v3_chalk = evaluate_system(
            df[df['v3_chalk_rule_fires']],
            "  └ Chalk rule (entropy ≥ q60, ml ≤ ML_MAX_FOR_BET)",
            takeout=takeout)
        v3_metrics.append(v3_chalk)

        v3_longshot = evaluate_system(
            df[df['v3_longshot_rule_fires']],
            "  └ Longshot rule (entropy ≤ q40, ml ≥ ML_MIN_LONGSHOT)",
            takeout=takeout)
        v3_metrics.append(v3_longshot)

        # Within-rule trigger slices. These overlap on multi-trigger
        # horses (e.g. RDG+RAW both firing on same horse). Use them
        # to see which model is doing the work in each rule.
        v3_chalk_log = evaluate_system(
            df[df['v3_chalk_rule_fires'] & df['v3_log_fires']],
            "      ├ chalk × LOG (any model on chalk side, but LOG also fired)",
            takeout=takeout)
        v3_metrics.append(v3_chalk_log)

        v3_chalk_rdg = evaluate_system(
            df[df['v3_chalk_rule_fires'] & df['v3_rdg_fires']],
            "      ├ chalk × RDG", takeout=takeout)
        v3_metrics.append(v3_chalk_rdg)

        v3_chalk_raw = evaluate_system(
            df[df['v3_chalk_rule_fires'] & df['v3_raw_fires']],
            "      └ chalk × RAW", takeout=takeout)
        v3_metrics.append(v3_chalk_raw)

        # LOG fires almost never at ml ≥ 12 (logistic edge concentrates
        # on chalk), but include for completeness.
        v3_long_log = evaluate_system(
            df[df['v3_longshot_rule_fires'] & df['v3_log_fires']],
            "      ├ longshot × LOG (rare — LOG seldom fires on longshots)",
            takeout=takeout)
        v3_metrics.append(v3_long_log)

        v3_long_rdg = evaluate_system(
            df[df['v3_longshot_rule_fires'] & df['v3_rdg_fires']],
            "      ├ longshot × RDG", takeout=takeout)
        v3_metrics.append(v3_long_rdg)

        v3_long_raw = evaluate_system(
            df[df['v3_longshot_rule_fires'] & df['v3_raw_fires']],
            "      └ longshot × RAW", takeout=takeout)
        v3_metrics.append(v3_long_raw)

        thr60_str = ('not computed' if V3_ENTROPY_Q60 is None
                     else f"{V3_ENTROPY_Q60:.3f}")
        thr40_str = ('not computed' if V3_ENTROPY_Q40 is None
                     else f"{V3_ENTROPY_Q40:.3f}")
        print(f"\n{'='*78}")
        print(f"PREDICT_V3 (two-rule entropy×ml gate) — IN-SAMPLE")
        print(f"  Entropy from ensemble cv_prob (normalized H/log(N)).")
        print(f"  Chalk-rule threshold    (q60): {thr60_str}")
        print(f"  Longshot-rule threshold (q40): {thr40_str}")
        print(f"  Trigger:    log_bet OR rdg_bet OR raw_bet (any model fires)")
        print(f"  Chalk rule: race_entropy_norm ≥ q60 AND "
              f"ml_decimal ≤ {p3.ML_MAX_FOR_BET}  →  W/WP/WPS by ml")
        print(f"  Longshot:   race_entropy_norm ≤ q40 AND "
              f"ml_decimal ≥ {p3.ML_MIN_LONGSHOT} →  W/WP/WPS by ml")
        print(f"  LOG-fired horses: forced W (overrides ml-based bet type)")
        print(f"")
        print(f"  Two corners, opposite bets:")
        print(f"    Chalk corner    — flat-market favorites: prices spread")
        print(f"      enough across field for an edge to overcome takeout.")
        print(f"      Empirically the strongest cell in current sample.")
        print(f"    Longshot corner — peaked-market longshots: when public")
        print(f"      consensus loads onto a chalk and any model still flags")
        print(f"      an outsider, the outsider may be overpriced. Weak in")
        print(f"      current sample (favorite-heavy run); tested prospectively.")
        print(f"  Mid-ml zone (8 < ml < 12) excluded — worst bucket regardless")
        print(f"  of trigger or entropy band per the 3×5 diagnostic.")
        print(f"")
        print(f"  ⚠ Thresholds are 60th/40th percentiles of THIS data's race")
        print(f"    cv-entropy distribution. ~40% of races pass each gate by")
        print(f"    construction; absolute thresholds drift modestly as data")
        print(f"    is added. Track prospectively on new cards.")
        print(f"{'='*78}")
        print_roi_report(v3_metrics, takeout=takeout,
                         section_label="predict_v3 entropy-filtered")

        # Race-entropy distribution snapshot — useful to see whether
        # the gates are doing real work or rejecting nearly nothing.
        race_ent_series = (df.drop_duplicates(['date', 'track', 'race'])
                             ['race_entropy_norm'])
        race_ent_clean = race_ent_series.dropna()
        if (len(race_ent_clean) > 0
                and V3_ENTROPY_Q60 is not None
                and V3_ENTROPY_Q40 is not None):
            n_pass_chalk    = (race_ent_clean >= V3_ENTROPY_Q60).sum()
            n_pass_longshot = (race_ent_clean <= V3_ENTROPY_Q40).sum()
            n_total = len(race_ent_clean)
            print(f"\n  Race cv-norm entropy distribution: "
                  f"min={race_ent_clean.min():.3f}, "
                  f"median={race_ent_clean.median():.3f}, "
                  f"q40={V3_ENTROPY_Q40:.3f}, "
                  f"q60={V3_ENTROPY_Q60:.3f}, "
                  f"max={race_ent_clean.max():.3f}")
            print(f"    Races passing chalk gate    (≥ q60): "
                  f"{n_pass_chalk}/{n_total} ({n_pass_chalk/n_total:.0%})")
            print(f"    Races passing longshot gate (≤ q40): "
                  f"{n_pass_longshot}/{n_total} ({n_pass_longshot/n_total:.0%})")

        # --- v3 RDG-only "one per race" variants ---
        # The current RDG filter fires on every horse above ridge break-even
        # in low-entropy races. Often that's multiple horses — ridge has
        # a ranked opinion but we treat all "above threshold" horses the
        # same. Question: would taking only the top ridge-scoring horse per
        # race improve ROI?
        #
        # We test two tiebreakers:
        #   (a) Highest ridge_value_norm — ridge's strongest opinion
        #   (b) Highest ensemble_prob — ridge agrees AND the ensemble likes
        # And add a diagnostic showing how often multi-trigger races even
        # happen, plus comparing top-vs-rest hit rates within those.
        # Restrict to chalk-rule RDG races (entropy ≥ q60, ml ≤ 8).
        # The longshot rule's RDG fires haven't been validated for
        # exacta boxing — boxing $4+ on horses paying ml ≥ 12 each
        # would chew through bankroll fast and the original exacta
        # finding was on chalk-side races.
        rdg_only_pool = df[df['v3_rdg_fires']
                           & ~df['v3_log_fires']
                           & df['v3_chalk_rule_fires']].copy()

        # Baseline metric for the whole pool — the one-per-race variants
        # below get compared against this. (Used to be defined further up
        # in the v3_metrics block but moved here so it stays adjacent to
        # rdg_only_pool's definition; if you change the pool filter, the
        # baseline updates with it.)
        v3_rdg_only = evaluate_system(
            rdg_only_pool,
            "  └ v3 RDG only (chalk-rule, no LOG; W/WP/WPS by ML)",
            takeout=takeout)

        if len(rdg_only_pool) > 0:
            # How concentrated is the multi-horse situation?
            races_with_rdg = rdg_only_pool.groupby(
                ['date', 'track', 'race']).size()
            n_rdg_races = len(races_with_rdg)
            n_multi = (races_with_rdg > 1).sum()
            n_single = (races_with_rdg == 1).sum()

            print(f"\n  RDG-only multi-horse diagnostic:")
            print(f"    Total RDG-only races: {n_rdg_races}")
            print(f"    Races with 1 trigger:  {n_single} "
                  f"({n_single/n_rdg_races:.0%})")
            print(f"    Races with 2+ triggers: {n_multi} "
                  f"({n_multi/n_rdg_races:.0%})")
            if n_multi > 0:
                avg_extras = (
                    rdg_only_pool[rdg_only_pool.set_index(
                        ['date', 'track', 'race']).index.isin(
                        races_with_rdg[races_with_rdg > 1].index)]
                    .groupby(['date', 'track', 'race']).size().mean()
                )
                print(f"    Avg horses in multi-trigger races: {avg_extras:.1f}")

            # Variant 1: top by ridge_value_norm per race
            if 'ridge_value_norm' in rdg_only_pool.columns:
                top_idx = rdg_only_pool.groupby(
                    ['date', 'track', 'race']
                )['ridge_value_norm'].idxmax()
                rdg_top_ridge = rdg_only_pool.loc[top_idx]
                m_top_ridge = evaluate_system(
                    rdg_top_ridge,
                    "  └ RDG only, top by ridge_value (1/race)",
                    takeout=takeout)

            # Variant 2: top by ensemble_prob per race
            # Pick whichever prob column exists — match what compute_bet_flags
            # / load_cv_predictions normalized to.
            prob_col = None
            for candidate in ('ensemble_prob', 'cv_prob'):
                if (candidate in rdg_only_pool.columns
                        and rdg_only_pool[candidate].notna().any()):
                    prob_col = candidate
                    break
            if prob_col is not None:
                top_idx = rdg_only_pool.groupby(
                    ['date', 'track', 'race']
                )[prob_col].idxmax()
                rdg_top_prob = rdg_only_pool.loc[top_idx]
                m_top_prob = evaluate_system(
                    rdg_top_prob,
                    f"  └ RDG only, top by {prob_col} (1/race)",
                    takeout=takeout)

            # Variant 3: the ADDITIONAL horses (rank 2+ by ridge_value_norm
            # within RDG-only races). If these match the top-ranked horse's
            # hit rate, the multi-horse behavior isn't the problem and
            # one-per-race won't help. If they're meaningfully worse, the
            # one-per-race rule is doing real work.
            if 'ridge_value_norm' in rdg_only_pool.columns and n_multi > 0:
                # Rank within (date, track, race) by ridge_value_norm desc
                rdg_only_pool['_rdg_rank'] = rdg_only_pool.groupby(
                    ['date', 'track', 'race']
                )['ridge_value_norm'].rank(ascending=False, method='first')
                rdg_extras = rdg_only_pool[rdg_only_pool['_rdg_rank'] > 1]
                m_extras = evaluate_system(
                    rdg_extras,
                    "  └ RDG only, EXTRAS (rank 2+)",
                    takeout=takeout)

            print(f"\n{'-'*78}")
            print(f"  RDG one-per-race variants (and the extras for context):")
            print(f"{'-'*78}")
            extras_metrics = [v3_rdg_only]  # for reference
            if 'ridge_value_norm' in rdg_only_pool.columns:
                extras_metrics.append(m_top_ridge)
            if prob_col is not None:
                extras_metrics.append(m_top_prob)
            if ('ridge_value_norm' in rdg_only_pool.columns
                    and n_multi > 0):
                extras_metrics.append(m_extras)
            print_roi_report(extras_metrics, takeout=takeout,
                             section_label="v3 RDG one-per-race variants")

            print(f"\n  How to read this:")
            print(f"    - If 'top by ridge' beats 'RDG only' meaningfully,")
            print(f"      the multi-horse behavior is leaking ROI.")
            print(f"    - If 'EXTRAS' has similar ROI to 'top by ridge',")
            print(f"      ridge has no useful intra-race ranking — additional")
            print(f"      horses cash about as often as the top one.")
            print(f"    - If 'EXTRAS' is much worse, ridge's ranking IS")
            print(f"      meaningful and one-per-race captures most of the")
            print(f"      signal at lower cost.")
            print(f"    Sample sizes are small — read trends, not single cells.")

            # --- v3 EXACTA BOX BACKTEST on multi-trigger RDG races ---
            # Hypothesis: if multiple ridge horses fire in the same race
            # AND the extras hit at similar rates to the top, exacta boxes
            # over the firing horses might capture the multi-horse signal
            # at exotic-pool prices.
            #
            # Box rules (per Blake):
            #   2 horses → $4 exacta box (2 combinations × $2)
            #     hits if both fired horses finish 1-2 in any order
            #   3 horses → $12 exacta box (6 combinations × $2)
            #     hits if any 2 of the 3 fired horses finish 1-2
            #   4+ → skip (a 4-horse box is $24, sample too thin to justify)
            #
            # On hit, the box pays exacta_$2_payout (one winning $2 ticket).
            #
            # IMPORTANT: in-sample, same caveat as everything else here.
            #
            # We require meta CSVs at DATA_PATH/meta/{track}_{date}.csv with
            # exacta_finish and exacta_$2_payout columns.
            #
            # SCOPE NOTE: this runs in two passes:
            #   (A) RDG-only races: races where RDG fires AND LOG does NOT.
            #       This is the historical scope — preserved as a baseline.
            #   (B) RDG races (incl. LOG): every race with 2+ RDG-firing
            #       horses, regardless of whether LOG also fires on any of
            #       them. Predict/daily-summary currently exclude LOG-
            #       triggered races from exacta consideration; this pass
            #       answers "would including them shift the backtest?"
            #       If (B) doesn't degrade vs (A), it's safe to add LOG
            #       races to predict/daily-summary.
            exb_results_A = _run_exacta_box_backtest(
                rdg_only_pool,
                "RDG-only races  (baseline: ~v3_log_fires)",
                takeout=takeout)

            # Build pool (B): same RDG-firing horses, but include races
            # where LOG also fires. Note we still only pool RDG-firing
            # horses — LOG-only horses (RDG didn't fire) aren't in the
            # box, since the box hypothesis is about ridge. Also still
            # restricted to chalk-rule races for the same reason as
            # pool A: longshot-rule exactas are untested.
            rdg_pool_all = df[df['v3_rdg_fires']
                              & df['v3_chalk_rule_fires']].copy()
            exb_results_B = _run_exacta_box_backtest(
                rdg_pool_all,
                "RDG races incl. LOG  (proposed: drop ~v3_log_fires)",
                takeout=takeout)

            # Side-by-side comparison so the impact is glanceable
            _print_exacta_scope_comparison(exb_results_A, exb_results_B)

    # --- ML-decimal bucket breakdown ---
    # The key diagnostic: are triggered bets clustering in a specific odds
    # range? If RAW is -40% and all its bets are on ≤3.5 favorites, the
    # problem is "bets favorites," not a bad value signal.
    print(f"\n{'='*78}")
    print(f"ML-DECIMAL BUCKET BREAKDOWN (where do the bets/losses live?)")
    print(f"  ≤3.5 = deep chalk, 3.5-4.5 = lukewarm faves,")
    print(f"  4.5-9.0 = mid-range, >9.0 = longshots")
    print(f"{'='*78}")

    print_bucket_diagnostic("All predict_v2 bets", all_bets, takeout=takeout)
    print_bucket_diagnostic("AGREE only", agree_bets, takeout=takeout)
    if 'log_bet' in df.columns:
        print_bucket_diagnostic("LOG triggered",
                               df[df['any_bet'] & df['log_bet']],
                               takeout=takeout)
    if 'rdg_bet' in df.columns:
        print_bucket_diagnostic("RDG triggered",
                               df[df['any_bet'] & df['rdg_bet']],
                               takeout=takeout)
    if 'raw_bet' in df.columns:
        print_bucket_diagnostic("RAW triggered",
                               df[df['any_bet'] & df['raw_bet']],
                               takeout=takeout)

    # --- 2D ENTROPY × ML_DECIMAL BUCKET DIAGNOSTIC ---
    # Look for non-monotonic patterns in ROI across the joint
    # distribution. The longshot-in-low-entropy hypothesis (peaked
    # races may overprice their favorite, leaving longshots
    # mispriced upward) is exactly the kind of pattern that 1D
    # bucketing hides — a high-ROI cell at (peakest entropy, >9.0
    # ml_decimal) doesn't show up in either marginal alone if the
    # rest of the column/row is bad.
    #
    # Runs on `all_bets` (every horse predict_v2 would bet) so the
    # signal isn't filtered down to just one trigger type. The 2D
    # table includes both NORMALIZED and RAW entropy axes since
    # the field-size effect that distinguishes them might itself
    # be informative.
    if HAS_V3 and not all_bets.empty:
        # Need the entropy columns — compute_v3_filters added them
        # to df above, but all_bets was a slice taken before that.
        # Re-merge all three entropy columns (ml/norm/raw) from df now.
        ent_cols = ['date', 'track', 'race',
                    'race_entropy_ml',
                    'race_entropy_norm', 'race_entropy_raw']
        ent_cols_present = [c for c in ent_cols if c in df.columns]
        # Only run the diagnostic if at least one entropy axis is
        # available (the ML axis is the most likely; the cv ones
        # require cv_prob).
        if any(c.startswith('race_entropy_') for c in ent_cols_present):
            all_bets_ent = all_bets.merge(
                df[ent_cols_present].drop_duplicates(['date', 'track', 'race']),
                on=['date', 'track', 'race'], how='left',
                suffixes=('', '_drop'),
            )
            for c in ('race_entropy_ml_drop',
                      'race_entropy_norm_drop',
                      'race_entropy_raw_drop'):
                if c in all_bets_ent.columns:
                    all_bets_ent = all_bets_ent.drop(columns=[c])

            print(f"\n{'='*78}")
            print(f"3×3 ENTROPY × ML_DECIMAL — predict_v2 ALL BETS")
            print(f"  Three entropy axes (ML / cv-normalized / cv-raw),")
            print(f"  each crossed with ml_decimal thirds.")
            print(f"{'='*78}")
            print_entropy_x_ml_bucket_diagnostic(all_bets_ent, takeout=takeout)

            # Same diagnostic at finer ml granularity — quintiles on
            # the odds axis. Cell sizes shrink (~1/5 vs ~1/3 of bets
            # per column) but the corners separate more cleanly, which
            # matters for the "favorites vs longshots" prior the
            # diagnostic was built to test.
            print(f"\n{'='*78}")
            print(f"3×5 ENTROPY × ML_DECIMAL — predict_v2 ALL BETS")
            print(f"  Same shape as above but ml-decimal QUINTILES")
            print(f"  (q1=tightest favorites, q5=biggest longshots).")
            print(f"{'='*78}")
            print_entropy_x_ml_3x5_diagnostic(all_bets_ent, takeout=takeout)

    # --- Summary interpretation ---
    print(f"\n{'='*78}")
    print(f"INTERPRETATION")
    print(f"{'='*78}")

    # predict_v2-style triggered bets: every horse that qualifies. This is
    # the actual production bet rate.
    trigger_systems = [m for m in filter_metrics
                      if m['n_bets'] >= 20 and not np.isnan(m.get('roi', np.nan))]
    if trigger_systems:
        # "All predict_v2 bets" is the umbrella; more interesting to find
        # the best slice (by-trigger or multi-trigger).
        best_filtered = max(trigger_systems, key=lambda m: m['roi'])
        print(f"  Best trigger-filtered: {best_filtered['label']}")
        print(f"    ROI = {best_filtered['roi']:+.1%} over {best_filtered['n_bets']} bets")
        print(f"    vs random = {best_filtered['takeout_cut_pct']:+.1%}")

    # Overall verdict
    combined = trigger_systems
    if combined:
        any_positive = any(m['roi'] > 0 for m in combined)
        any_beating_random = any(m['takeout_cut_pct'] > 0 for m in combined)
        print()
        if any_positive:
            print(f"  → At least one system has POSITIVE ROI in this sample.")
        elif any_beating_random:
            print(f"  → Still losing money, but at least one system beats random.")
            print(f"    The edge is real if this holds up on more dates.")
        else:
            print(f"  → No system beats random in this sample. No edge detected.")

    # Sample-size caveat — use trigger systems since all_metrics is
    # empty after the top-pick-every-race section was removed.
    total_bets = sum(m['n_bets'] for m in trigger_systems)
    if total_bets < 500:
        print(f"\n  ⚠ Sample size is small ({total_bets} total bets across "
              f"trigger systems).")
        print(f"    ROI has wide error bars — single-digit percentage swings "
              f"may be noise. Re-run on more dates to tighten estimates.")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest predict_v2 bet logic on out-of-sample predictions")
    parser.add_argument('--takeout', type=float, default=WIN_TAKEOUT,
                        help=f'Takeout rate (default {WIN_TAKEOUT})')
    parser.add_argument('--source', type=str, default=None,
                        choices=['walk_forward', 'cv', None],
                        help='Which predictions to backtest: walk_forward '
                             '(predict_v2 output from backtest_walk_forward.py — '
                             'the gold standard) or cv (logistic-only from '
                             'regression_analysis.py). Default auto-detects, '
                             'preferring walk_forward when available.')
    parser.add_argument('--meta-model', action='store_true',
                        help='Future: use a trained meta-model instead of rank-avg '
                             '(not implemented; falls back to rank-avg)')
    args = parser.parse_args()

    run_backtest(takeout=args.takeout, use_meta_model=args.meta_model,
                source=args.source)


if __name__ == '__main__':
    main()