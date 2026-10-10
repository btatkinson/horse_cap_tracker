"""
regression_analysis.py

Find optimal signal weights by regressing on actual race outcomes.
Ignores betting/odds entirely — just asks: what combination of signals
best predicts the winner?

Signals:
  - ML probability (morning line)
  - Claude direct win probability
  - Gemini direct win probability
  - Claude beyer (sim → prob)
  - Gemini beyer (sim → prob)
  - Human expert consensus (pool-weighted)

Two models are fit every run:
  1. Logistic regression on win/no-win — for pick-the-winner confidence
  2. Ridge regression on value target = 4×win_paid + 2×place_paid + 1×show_paid
     — for Pick 5 leg construction, where you want horses with positive
     expected dollar return across top-3 slots

Usage:
    python regression_analysis.py
    python regression_analysis.py --trees
    python regression_analysis.py --cv 5
"""

import os, json, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
ENTRIES_DIR = DATA_PATH / 'entries'
PICKS_DIR = DATA_PATH / 'ai_picks'

_TRACK_ALIASES = {
    'gulfstream-park': 'gulfstream', 'gulfstream park': 'gulfstream',
    'churchill-downs': 'churchill', 'churchill downs': 'churchill',
}

def _normalize_track(t):
    if pd.isna(t): return t
    t = str(t).strip().lower()
    return _TRACK_ALIASES.get(t, t)


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
                if yr < 100: yr += 2000
                return f"{yr}-{int(m):02d}-{int(day):02d}"
            except ValueError:
                pass
    return d


def _parse_ml_odds(odds_str):
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


def _parse_payout(val):
    """Parse payout string like '$8.40' or '$12.20' to float. Non-finishers → 0."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if s == '' or s.lower() in ('nan', 'none', '-'):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_data():
    """Load and merge AI picks, entries results, and expert picks."""

    # ── AI picks ──
    if not PICKS_DIR.exists():
        print("No ai_picks directory")
        return None

    ai_dfs = []
    for date_dir in sorted(PICKS_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        for track_dir in date_dir.iterdir():
            if not track_dir.is_dir():
                continue
            track = track_dir.name

            # Load claude, gemini, and openai
            frames = {}
            for provider in ['claude', 'gemini', 'openai']:
                csv = track_dir / f'{provider}.csv'
                if csv.exists():
                    pdf = pd.read_csv(csv)
                    # Rename columns
                    col_map = {}
                    for c in pdf.columns:
                        cl = c.lower().strip()
                        if 'beyer' in cl:
                            col_map[c] = f'{provider}_beyer'
                        elif 'closing' in cl:
                            col_map[c] = f'{provider}_closing'
                        elif 'win_prob' in cl or 'prob' in cl:
                            col_map[c] = f'{provider}_prob'
                    pdf = pdf.rename(columns=col_map)
                    pdf['race'] = pd.to_numeric(pdf['race'], errors='coerce')
                    pdf['horse_number'] = pd.to_numeric(pdf['horse_number'], errors='coerce')
                    frames[provider] = pdf

            if not frames:
                continue

            names = list(frames.keys())
            merged = frames[names[0]]
            for n in names[1:]:
                on = ['race', 'horse_number']
                if 'track' in merged.columns and 'track' in frames[n].columns:
                    on.append('track')
                merged = merged.merge(frames[n], on=on, how='outer', suffixes=('', f'_{n}'))

            merged['date'] = date_str
            merged['track'] = _normalize_track(track)
            ai_dfs.append(merged)

    if not ai_dfs:
        print("No AI pick data found")
        return None

    picks = pd.concat(ai_dfs, ignore_index=True)

    # ── Entries (results) ──
    if not ENTRIES_DIR.exists():
        print("No entries directory")
        return None

    entry_dfs = []
    for f in ENTRIES_DIR.glob('*.csv'):
        try:
            entry_dfs.append(pd.read_csv(f))
        except Exception:
            continue

    if not entry_dfs:
        return None

    entries = pd.concat(entry_dfs, ignore_index=True)
    if 'running' in entries.columns:
        entries = entries[entries['running'] == 1].copy()
    if 'date' in entries.columns:
        entries['date'] = entries['date'].apply(_normalize_date)
    if 'track' in entries.columns:
        entries['track'] = entries['track'].apply(_normalize_track)
    if 'finish' in entries.columns:
        entries['finish'] = pd.to_numeric(entries['finish'], errors='coerce')
        entries['won'] = (entries['finish'] == 1).astype(int)
    if 'horse' in entries.columns and 'horse_number' not in entries.columns:
        entries['horse_number'] = pd.to_numeric(entries['horse'], errors='coerce')
    if 'ml_odds' in entries.columns:
        entries['ml_prob'] = entries['ml_odds'].apply(_parse_ml_odds)

    # Parse payout columns → value target
    # value = 4 * win_paid + 2 * place_paid + 1 * show_paid
    # Non-finishers/non-hitters have 0 payout and contribute 0 to target.
    for col in ('win_paid', 'place_paid', 'show_paid'):
        if col in entries.columns:
            entries[f'{col}_num'] = entries[col].apply(_parse_payout)
        else:
            entries[f'{col}_num'] = 0.0

    entries['value_target'] = (
        4.0 * entries['win_paid_num']
        + 2.0 * entries['place_paid_num']
        + 1.0 * entries['show_paid_num']
    )

    # Normalize dates in picks
    picks['date'] = picks['date'].apply(_normalize_date)

    # Merge
    on = ['date', 'race', 'horse_number']
    if 'track' in picks.columns and 'track' in entries.columns:
        on.append('track')

    df = picks.merge(entries, on=on, how='inner')

    # ── Normalize ML within race ──
    df['ml_prob_norm'] = np.nan
    for _, race_df in df.groupby(['date', 'track', 'race']):
        idx = race_df.index
        raw = race_df['ml_prob'].values if 'ml_prob' in race_df.columns else None
        if raw is not None:
            valid = ~np.isnan(raw)
            if valid.any() and raw[valid].sum() > 0:
                normed = np.where(valid, raw / raw[valid].sum(), np.nan)
                # Handle NaN positions
                normed = np.where(valid, raw / np.nansum(raw), np.nan)
                df.loc[idx, 'ml_prob_norm'] = normed

    # ── Human expert pool ──
    df['human_pool'] = 0.0
    expert_dir = DATA_PATH / 'expert_picks'

    # Load expert ratings for weighting
    ratings_path = DATA_PATH / 'expert_ratings.json'
    ratings = {}
    if ratings_path.exists():
        with open(ratings_path) as f:
            ratings = json.load(f)

    # Build date-aware rating lookup: {expert: {date: rating_before}}
    # This prevents leakage — uses the rating GOING INTO each day
    def _get_expert_rating_for_date(expert_name, date_str):
        """Get the expert's rating as of the morning of date_str (no future info)."""
        info = ratings.get(expert_name)
        if not info:
            return info.get('initial_rating', 1500) if info else 1500

        initial = info.get('initial_rating', 1500)
        history = info.get('history', [])

        # Find the rating_before for this date, or the most recent rating_after
        # from a PRIOR date
        rating = initial
        for entry in sorted(history, key=lambda h: h['date']):
            if entry['date'] < date_str:
                rating = entry['rating_after']
            elif entry['date'] == date_str:
                rating = entry['rating_before']
                break
            else:
                break  # future date, stop

        return rating

    if expert_dir.exists():
        for date_dir in sorted(expert_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            date_str = date_dir.name
            for track_dir in date_dir.iterdir():
                if not track_dir.is_dir():
                    continue
                track = _normalize_track(track_dir.name)

                for csv_path in track_dir.glob('*.csv'):
                    expert_name = csv_path.stem
                    try:
                        edf = pd.read_csv(csv_path)
                    except Exception:
                        continue

                    if edf.empty:
                        continue

                    # Use date-aware rating (no leakage)
                    elo = _get_expert_rating_for_date(expert_name, date_str)
                    elo_mult = max(0.1, (elo - 1350) / 150)

                    for _, row in edf.iterrows():
                        race = int(row['race'])
                        hn = int(row['horse_number'])
                        delta = float(row['logit_delta'])

                        # Convert logit delta to pool dollars
                        # +0.70 best_bet → $2.00
                        # +0.50 strong → $1.43
                        # +0.30 mentioned → $0.86
                        # +0.15 contender → $0.43
                        # negative → subtract from pool
                        pool_amount = (delta / 0.7) * 2.0 * elo_mult

                        mask = (
                            (df['date'] == date_str) &
                            (df['track'] == track) &
                            (df['race'] == race) &
                            (df['horse_number'] == hn)
                        )
                        df.loc[mask, 'human_pool'] += pool_amount

    # Normalize human pool per race to probability
    df['human_prob'] = 0.0
    for _, race_df in df.groupby(['date', 'track', 'race']):
        idx = race_df.index
        pool = df.loc[idx, 'human_pool'].values
        # Shift so minimum is 0 (handle negatives)
        shifted = pool - pool.min() + 0.01  # small floor
        total = shifted.sum()
        if total > 0:
            df.loc[idx, 'human_prob'] = shifted / total

    return df


def evaluate_probs(df, prob_col, label=''):
    """Compute Spearman and log loss for a probability column."""
    rhos = []
    for _, race_df in df.groupby(['date', 'track', 'race']):
        valid = race_df.dropna(subset=[prob_col, 'finish'])
        if len(valid) >= 3:
            rho, _ = spearmanr(valid[prob_col], -valid['finish'])
            rhos.append(rho)

    mean_rho = np.mean(rhos) if rhos else np.nan

    valid = df.dropna(subset=[prob_col, 'won'])
    if len(valid) > 0:
        probs = valid[prob_col].clip(1e-6, 1 - 1e-6).values
        ll = log_loss(valid['won'].values, probs)
    else:
        ll = np.nan

    # Top pick win rate
    hits = []
    for _, race_df in df.groupby(['date', 'track', 'race']):
        valid = race_df.dropna(subset=[prob_col, 'won'])
        if len(valid) > 0:
            hits.append(valid.loc[valid[prob_col].idxmax(), 'won'])
    top_rate = np.mean(hits) if hits else np.nan

    return {'spearman': mean_rho, 'log_loss': ll, 'top_pick': top_rate,
            'n_races': len(rhos)}


def evaluate_value(df, score_col, target_col='value_target'):
    """
    Evaluate a dollar-prediction signal against the value target.

    Metrics:
      - rmse: root mean squared error (pooled across all horses)
      - top_pick_value: mean value_target for the horse the score ranks #1 in each race
      - top2_value: mean value_target summed over top-2 ranked per race
      - r2_overall: pooled R² against the value target
    """
    top_vals = []
    top2_vals = []

    for _, race_df in df.groupby(['date', 'track', 'race']):
        valid = race_df.dropna(subset=[score_col, target_col])
        if len(valid) < 2:
            continue

        # Top pick value — what did the horse the model liked most actually return?
        ordered = valid.sort_values(score_col, ascending=False)
        top_vals.append(ordered.iloc[0][target_col])
        if len(ordered) >= 2:
            top2_vals.append(ordered.iloc[:2][target_col].sum())

    mean_top = np.mean(top_vals) if top_vals else np.nan
    mean_top2 = np.mean(top2_vals) if top2_vals else np.nan

    # Pooled RMSE and R²
    valid_all = df.dropna(subset=[score_col, target_col])
    if len(valid_all) > 10:
        y = valid_all[target_col].values
        p = valid_all[score_col].values
        rmse = np.sqrt(np.mean((y - p) ** 2))
        ss_res = ((y - p) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        rmse = np.nan
        r2 = np.nan

    return {
        'rmse': rmse,
        'top_pick_value': mean_top,
        'top2_value': mean_top2,
        'r2': r2,
        'n_races': len(top_vals),
    }


def run_analysis(use_trees=False, cv_folds=5, reg_C=None):
    """Run regression analysis to find optimal signal weights."""

    df = load_data()
    if df is None or df.empty:
        return

    print(f"Loaded {len(df)} horses across {df.groupby(['date','track','race']).ngroups} races")

    # Sanity check value target
    if 'value_target' in df.columns:
        hitters = (df['value_target'] > 0).sum()
        print(f"  Value target: {hitters} horses with positive payout "
              f"(mean={df['value_target'].mean():.2f}, "
              f"max={df['value_target'].max():.2f})")

    # ── Define features ──
    # Convert beyers to within-race probabilities via softmax
    # Then transform all probabilities to logit space for regression
    for beyer_col in ['claude_beyer', 'gemini_beyer', 'openai_beyer']:
        prob_col = f'{beyer_col}_sim'
        df[prob_col] = np.nan
        if beyer_col in df.columns:
            for _, race_df in df.groupby(['date', 'track', 'race']):
                idx = race_df.index
                vals = df.loc[idx, beyer_col].values
                valid = ~np.isnan(vals)
                if valid.sum() >= 2:
                    # Softmax: higher beyer → higher prob
                    # Scale by ~0.1 to avoid extreme values
                    centered = vals[valid] - np.mean(vals[valid])
                    exp_vals = np.exp(centered * 0.1)
                    probs = exp_vals / exp_vals.sum()
                    full = np.full(len(vals), np.nan)
                    full[valid] = probs
                    df.loc[idx, prob_col] = full

    # Define probability features (all on same scale now)
    prob_features = [
        ('ml_prob_norm', 'ML'),
        ('claude_prob', 'Claude Prob'),
        ('gemini_prob', 'Gemini Prob'),
        ('openai_prob', 'OpenAI Prob'),
        ('claude_beyer_sim', 'Claude Beyer'),
        ('gemini_beyer_sim', 'Gemini Beyer'),
        ('openai_beyer_sim', 'OpenAI Beyer'),
        ('claude_closing', 'Claude Closing'),
        ('gemini_closing', 'Gemini Closing'),
        ('openai_closing', 'OpenAI Closing'),
        ('human_prob', 'Human Pool'),
    ]

    # Normalize all prob features per race (ensure they sum to ~1)
    for col, label in prob_features:
        if col not in df.columns or not df[col].notna().any():
            continue
        for _, race_df in df.groupby(['date', 'track', 'race']):
            idx = race_df.index
            vals = df.loc[idx, col].values
            valid = ~np.isnan(vals)
            if valid.any() and np.nansum(vals[valid]) > 0:
                df.loc[idx, col] = np.where(valid, vals / np.nansum(vals), np.nan)

    # Compute field size and interaction features
    df['inv_field_size'] = np.nan
    for _, race_df in df.groupby(['date', 'track', 'race']):
        idx = race_df.index
        n = len(race_df)
        df.loc[idx, 'inv_field_size'] = 1.0 / n

    # Interaction: ML_logit × 1/field_size
    # "Trust ML more in small fields" — if coefficient is positive,
    # the model amplifies ML's signal when the field is small
    clip_lo, clip_hi = 1e-4, 1 - 1e-4
    if 'ml_prob_norm' in df.columns:
        ml_logit = np.log(
            df['ml_prob_norm'].clip(clip_lo, clip_hi) /
            (1 - df['ml_prob_norm'].clip(clip_lo, clip_hi))
        )
        df['ml_x_fieldsize'] = ml_logit * df['inv_field_size']

    # Non-probability features (won't be logit-transformed)
    raw_features = [
        ('ml_x_fieldsize', 'ML×1/Field'),
    ]

    # Build feature lists
    feature_cols = []       # column names
    feature_labels = []     # display names
    feature_is_prob = []    # True = logit transform, False = use raw

    for col, label in prob_features:
        if col in df.columns and df[col].notna().any():
            feature_cols.append(col)
            feature_labels.append(label)
            feature_is_prob.append(True)

    for col, label in raw_features:
        if col in df.columns and df[col].notna().any():
            feature_cols.append(col)
            feature_labels.append(label)
            feature_is_prob.append(False)

    print(f"Features: {', '.join(feature_labels)}")
    print(f"  (probabilities in logit space, others raw)")

    # ── Individual signal evaluation ──
    print(f"\n{'='*65}")
    print(f"INDIVIDUAL SIGNAL PERFORMANCE")
    print(f"{'='*65}")
    print(f"  {'Signal':<16s} {'Spearman':>9s} {'LogLoss':>8s} {'TopPick':>8s} {'Top$':>7s}")
    print(f"  {'-'*16} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")

    has_value = 'value_target' in df.columns and df['value_target'].notna().any()
    for col, label in zip(feature_cols, feature_labels):
        m = evaluate_probs(df, col, label)
        rho_s = f"{m['spearman']:.4f}" if not np.isnan(m['spearman']) else "n/a"
        ll_s = f"{m['log_loss']:.4f}" if not np.isnan(m['log_loss']) else "n/a"
        tp_s = f"{m['top_pick']:.1%}" if not np.isnan(m['top_pick']) else "n/a"

        if has_value:
            v = evaluate_value(df, col)
            tv_s = f"{v['top_pick_value']:.2f}" if not np.isnan(v['top_pick_value']) else "n/a"
        else:
            tv_s = "n/a"

        print(f"  {label:<16s} {rho_s:>9s} {ll_s:>8s} {tp_s:>8s} {tv_s:>7s}")

    # ── Prepare regression data ──
    # Drop rows with any NaN in features or target
    valid_mask = df['won'].notna()
    for col in feature_cols:
        valid_mask = valid_mask & df[col].notna()

    reg_df = df[valid_mask].copy()
    print(f"\n{len(reg_df)} horses with complete data for regression")

    if len(reg_df) < 50:
        print("  Not enough data for regression")
        return

    # Transform features: probabilities → logit space, others stay raw
    clip_lo, clip_hi = 1e-4, 1 - 1e-4
    transformed_cols = []
    for col, is_prob in zip(feature_cols, feature_is_prob):
        if is_prob:
            t_col = f'{col}_logit'
            reg_df[t_col] = np.log(
                reg_df[col].clip(clip_lo, clip_hi) /
                (1 - reg_df[col].clip(clip_lo, clip_hi))
            )
        else:
            t_col = col  # use raw value
        transformed_cols.append(t_col)

    X = reg_df[transformed_cols].values       # logit-space for logistic regression
    X_value = reg_df[feature_cols].values      # raw probabilities for ridge (dollar target)
    y = reg_df['won'].values
    # log1p transform compresses the heavy tail of dollar payouts so ridge
    # doesn't optimize purely for "predict the mean." Most horses pay $0
    # (log1p(0)=0), a $10 payout → 2.4, a $50 payout → 3.9, a $150 → 5.0.
    # This makes the loss function care roughly equally about distinguishing
    # $0 from $5, $5 from $20, and $20 from $80, rather than letting a
    # single $150 longshot dominate the entire gradient.
    y_value = np.log1p(reg_df['value_target'].values) if has_value else None

    # Race groups for leave-one-group-out CV
    reg_df['race_id'] = reg_df['date'] + '_' + reg_df['track'] + '_' + reg_df['race'].astype(str)
    groups = reg_df['race_id'].values

    # Card groups for k-fold grouped CV (shared between both models)
    reg_df['card_id'] = reg_df['date'] + '_' + reg_df['track']
    card_ids = reg_df['card_id'].values
    unique_cards = np.unique(card_ids)
    n_cards = len(unique_cards)
    actual_folds = min(cv_folds, n_cards)

    # ── Logistic Regression ──
    # In logit space, coefficients directly tell you how much each signal's
    # log-odds contributes to the final log-odds of winning.
    # Coefficient of 1.0 = "trust this signal's logit fully"
    # Coefficient of 0.0 = "ignore this signal"
    print(f"\n{'='*65}")
    print(f"LOGISTIC REGRESSION (logit-space features → P(win))")
    print(f"{'='*65}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_scaled, y)

    print(f"\n  Coefficients (standardized):")
    coefs = list(zip(feature_labels, lr.coef_[0]))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    for label, coef in coefs:
        bar = '█' * int(abs(coef) * 20)
        sign = '+' if coef > 0 else '-'
        print(f"    {label:<16s} {coef:>+7.3f}  {sign}{bar}")

    # Raw (unscaled) coefficients — in logit space these are directly interpretable
    # Coef ≈ 1.0 means "trust this signal's logit fully"
    # Coef ≈ 0.0 means "ignore this signal"
    # Sum of coefs near 1.0 means signals are being averaged in logit space
    lr_raw = LogisticRegression(max_iter=1000, C=1.0)
    lr_raw.fit(X, y)
    print(f"\n  Logit-space coefficients (how much each signal's log-odds contributes):")
    total_coef = sum(abs(c) for c in lr_raw.coef_[0])
    for label, coef in zip(feature_labels, lr_raw.coef_[0]):
        pct = abs(coef) / total_coef * 100 if total_coef > 0 else 0
        print(f"    {label:<16s} {coef:>+7.4f}  ({pct:4.1f}%)")
    print(f"    {'intercept':<16s} {lr_raw.intercept_[0]:>+7.4f}")

    # In-sample predictions
    lr_probs = lr_raw.predict_proba(X)[:, 1]
    reg_df['lr_prob'] = lr_probs

    # Normalize per race
    for _, race_df in reg_df.groupby('race_id'):
        idx = race_df.index
        total = reg_df.loc[idx, 'lr_prob'].sum()
        if total > 0:
            reg_df.loc[idx, 'lr_prob'] /= total

    m = evaluate_probs(reg_df, 'lr_prob', 'LogReg')
    print(f"\n  In-sample: ρ={m['spearman']:.4f}  LL={m['log_loss']:.4f}  "
          f"TopPick={m['top_pick']:.1%}")

    # ── Tune C via CV ──
    if reg_C is None:
        print(f"\n  Tuning regularization (C) via {actual_folds}-fold CV...")
        c_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        best_c = 1.0
        best_rho = -999
        gkf_tune = GroupKFold(n_splits=actual_folds)

        for c_val in c_values:
            cv_probs_tune = np.zeros(len(reg_df))
            for train_idx, test_idx in gkf_tune.split(X, y, card_ids):
                lr_tune = LogisticRegression(max_iter=1000, C=c_val)
                lr_tune.fit(X[train_idx], y[train_idx])
                cv_probs_tune[test_idx] = lr_tune.predict_proba(X[test_idx])[:, 1]

            # Normalize per race and evaluate
            reg_df['_tune_prob'] = cv_probs_tune
            for _, race_df in reg_df.groupby('race_id'):
                idx = race_df.index
                total = reg_df.loc[idx, '_tune_prob'].sum()
                if total > 0:
                    reg_df.loc[idx, '_tune_prob'] /= total

            m_tune = evaluate_probs(reg_df, '_tune_prob', f'C={c_val}')
            rho = m_tune['spearman'] if not np.isnan(m_tune['spearman']) else -999
            marker = ' ← best' if rho > best_rho else ''
            print(f"    C={c_val:<5g}  ρ={rho:.4f}  TopPick={m_tune['top_pick']:.1%}{marker}")
            if rho > best_rho:
                best_rho = rho
                best_c = c_val

        reg_df.drop(columns=['_tune_prob'], inplace=True)
        reg_C = best_c
        print(f"  Best C={reg_C}")

    print(f"\n  {actual_folds}-Fold Grouped CV ({n_cards} cards, C={reg_C}):")

    gkf = GroupKFold(n_splits=actual_folds)
    cv_probs = np.zeros(len(reg_df))
    fold_models = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, card_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        lr_fold = LogisticRegression(max_iter=1000, C=reg_C)
        lr_fold.fit(X_train, y_train)
        fold_models.append(lr_fold)

        cv_probs[test_idx] = lr_fold.predict_proba(X_test)[:, 1]

        test_cards = np.unique(card_ids[test_idx])
        test_wins = y[test_idx].sum()
        print(f"    Fold {fold_i+1}: train={len(train_idx)}, test={len(test_idx)} "
              f"({len(test_cards)} cards, {test_wins} winners)")

    reg_df['cv_prob'] = cv_probs

    # Normalize per race
    for _, race_df in reg_df.groupby('race_id'):
        idx = race_df.index
        total = reg_df.loc[idx, 'cv_prob'].sum()
        if total > 0:
            reg_df.loc[idx, 'cv_prob'] /= total

    m_cv = evaluate_probs(reg_df, 'cv_prob', 'CV Ensemble')
    print(f"\n  CV Ensemble: ρ={m_cv['spearman']:.4f}  LL={m_cv['log_loss']:.4f}  "
          f"TopPick={m_cv['top_pick']:.1%}")

    # Compare to ML
    m_ml = evaluate_probs(reg_df, 'ml_prob_norm', 'ML')
    print(f"  ML baseline: ρ={m_ml['spearman']:.4f}  LL={m_ml['log_loss']:.4f}  "
          f"TopPick={m_ml['top_pick']:.1%}")
    edge = m_cv['spearman'] - m_ml['spearman']
    print(f"  Edge over ML: {edge:+.4f} Spearman")

    # ── Average fold coefficients for the production model ──
    avg_coefs = np.mean([m.coef_[0] for m in fold_models], axis=0)
    avg_intercept = np.mean([m.intercept_[0] for m in fold_models])

    print(f"\n  Averaged model coefficients ({actual_folds} folds):")
    for label, coef in zip(feature_labels, avg_coefs):
        print(f"    {label:<16s} {coef:>+8.4f}")
    print(f"    {'intercept':<16s} {avg_intercept:>+8.4f}")

    # Save the averaged model for use in predict_races
    model_params = {
        'feature_cols': feature_cols,
        'feature_labels': feature_labels,
        'feature_is_prob': feature_is_prob,
        'input_space': 'mixed',  # prob features logit-transformed, others raw
        'regularization_C': reg_C,
        'coefficients': avg_coefs.tolist(),
        'intercept': float(avg_intercept),
        'n_folds': actual_folds,
        'n_cards': n_cards,
        'n_horses': len(reg_df),
        'cv_spearman': float(m_cv['spearman']),
        'cv_log_loss': float(m_cv['log_loss']),
        'cv_top_pick': float(m_cv['top_pick']),
    }

    model_path = DATA_PATH / 'regression_model.json'
    with open(model_path, 'w') as f:
        json.dump(model_params, f, indent=2)
    print(f"\n  Model saved to {model_path}")

    # ── Ridge Regression on Value Target ─────────────────────────────
    # Target = 4 * win_paid + 2 * place_paid + 1 * show_paid (raw dollars)
    # Most horses contribute 0 (didn't hit the board). The few that do have
    # large values — hence ridge (L2) to keep coefficients stable against the
    # high-variance target. Same features, same folds as the logistic model,
    # so you can directly compare signal weights across the two objectives.
    if has_value:
        print(f"\n{'='*65}")
        print(f"RIDGE REGRESSION (signals → E[value])")
        print(f"  value = 4·win_paid + 2·place_paid + 1·show_paid")
        print(f"{'='*65}")

        # Tune alpha on the same grouped folds as the logistic model
        alpha_grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
        gkf_ridge = GroupKFold(n_splits=actual_folds)
        ridge_cv_preds = np.zeros(len(reg_df))
        ridge_fold_models = []

        # RidgeCV handles alpha selection internally using GCV, but we want
        # card-grouped CV to match the logistic model. So we do manual tuning.
        print(f"\n  Tuning alpha via {actual_folds}-fold grouped CV...")
        best_alpha = 1.0
        best_rmse = np.inf

        for alpha in alpha_grid:
            preds_tmp = np.zeros(len(reg_df))
            for train_idx, test_idx in gkf_ridge.split(X_value, y_value, card_ids):
                rr = Ridge(alpha=alpha)
                rr.fit(X_value[train_idx], y_value[train_idx])
                preds_tmp[test_idx] = rr.predict(X_value[test_idx])

            # Convert log1p predictions back to dollar space for evaluation
            reg_df['_ridge_tmp'] = np.expm1(preds_tmp)
            v = evaluate_value(reg_df, '_ridge_tmp')
            rmse = v['rmse'] if not np.isnan(v['rmse']) else np.inf

            marker = ' ← best' if rmse < best_rmse else ''
            print(f"    α={alpha:<6g}  RMSE={rmse:.4f}  "
                  f"R²={v['r2']:+.4f}  Top$={v['top_pick_value']:.2f}{marker}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha

        reg_df.drop(columns=['_ridge_tmp'], inplace=True)
        print(f"  Best α={best_alpha}")

        # Final CV pass with best alpha
        print(f"\n  {actual_folds}-Fold Grouped CV (α={best_alpha}):")
        for fold_i, (train_idx, test_idx) in enumerate(
                gkf_ridge.split(X_value, y_value, card_ids)):
            rr = Ridge(alpha=best_alpha)
            rr.fit(X_value[train_idx], y_value[train_idx])
            ridge_fold_models.append(rr)
            ridge_cv_preds[test_idx] = rr.predict(X_value[test_idx])

            test_cards = np.unique(card_ids[test_idx])
            # Show dollar-space means for readability
            mean_actual = np.expm1(y_value[test_idx]).mean()
            mean_pred = np.expm1(ridge_cv_preds[test_idx]).mean()
            print(f"    Fold {fold_i+1}: train={len(train_idx)}, test={len(test_idx)} "
                  f"({len(test_cards)} cards, "
                  f"actual μ=${mean_actual:.2f}, pred μ=${mean_pred:.2f})")

        # Convert log1p predictions back to dollar space for evaluation/output
        reg_df['ridge_value'] = np.expm1(ridge_cv_preds)

        v_cv = evaluate_value(reg_df, 'ridge_value')
        print(f"\n  Ridge CV: RMSE={v_cv['rmse']:.4f}  R²={v_cv['r2']:+.4f}")
        print(f"    Top-pick value: ${v_cv['top_pick_value']:.2f} per race")
        print(f"    Top-2 value:    ${v_cv['top2_value']:.2f} per race")

        # Baseline: rank by ML — what $ do you get picking the favorite?
        v_ml = evaluate_value(reg_df, 'ml_prob_norm')
        print(f"  ML baseline:")
        print(f"    Top-pick value: ${v_ml['top_pick_value']:.2f} per race")
        print(f"    Top-2 value:    ${v_ml['top2_value']:.2f} per race")
        edge_val = v_cv['top_pick_value'] - v_ml['top_pick_value']
        print(f"  Edge over ML (top pick): ${edge_val:+.2f} per race")

        # Average fold coefficients for production
        avg_ridge_coefs = np.mean([m.coef_ for m in ridge_fold_models], axis=0)
        avg_ridge_intercept = np.mean([m.intercept_ for m in ridge_fold_models])

        print(f"\n  Averaged ridge coefficients ({actual_folds} folds):")
        total_r = sum(abs(c) for c in avg_ridge_coefs)
        for label, coef in zip(feature_labels, avg_ridge_coefs):
            pct = abs(coef) / total_r * 100 if total_r > 0 else 0
            print(f"    {label:<16s} {coef:>+8.4f}  ({pct:4.1f}%)")
        print(f"    {'intercept':<16s} {avg_ridge_intercept:>+8.4f}")

        # Save ridge model alongside logistic
        ridge_params = {
            'target': 'log1p(4*win_paid + 2*place_paid + 1*show_paid)',
            'target_transform': 'log1p',  # predict_races must expm1() the output
            'feature_cols': feature_cols,
            'feature_labels': feature_labels,
            'feature_is_prob': feature_is_prob,
            'input_space': 'raw',
            'regularization_alpha': best_alpha,
            'coefficients': avg_ridge_coefs.tolist(),
            'intercept': float(avg_ridge_intercept),
            'n_folds': actual_folds,
            'n_cards': n_cards,
            'n_horses': len(reg_df),
            'cv_rmse': float(v_cv['rmse']) if not np.isnan(v_cv['rmse']) else None,
            'cv_r2': float(v_cv['r2']) if not np.isnan(v_cv['r2']) else None,
            'cv_top_pick_value': float(v_cv['top_pick_value']) if not np.isnan(v_cv['top_pick_value']) else None,
            'cv_top2_value': float(v_cv['top2_value']) if not np.isnan(v_cv['top2_value']) else None,
            'ml_baseline_top_pick_value': float(v_ml['top_pick_value']) if not np.isnan(v_ml['top_pick_value']) else None,
        }
        ridge_path = DATA_PATH / 'ridge_value_model.json'
        with open(ridge_path, 'w') as f:
            json.dump(ridge_params, f, indent=2)
        print(f"\n  Ridge model saved to {ridge_path}")

    # ── Per-race CV predictions output ──
    out_cols = ['date', 'track', 'race', 'horse_number', 'cv_prob',
                'ml_prob_norm', 'won', 'finish']
    if has_value:
        out_cols += ['ridge_value', 'value_target']
    out_cols = [c for c in out_cols if c in reg_df.columns]
    out_path = DATA_PATH / 'regression_predictions.csv'
    reg_df[out_cols].to_csv(out_path, index=False)
    print(f"  Predictions saved to {out_path}")

    # ── Tree-based models ──
    if use_trees:
        print(f"\n{'='*65}")
        print(f"TREE-BASED MODELS")
        print(f"{'='*65}")

        for name, model in [
            ('RandomForest', RandomForestClassifier(
                n_estimators=200, max_depth=4, min_samples_leaf=10,
                random_state=42)),
            ('GBM', GradientBoostingClassifier(
                n_estimators=100, max_depth=3, min_samples_leaf=10,
                learning_rate=0.1, random_state=42)),
        ]:
            model.fit(X, y)

            # Feature importance
            importances = model.feature_importances_
            print(f"\n  {name} Feature Importance:")
            imp_sorted = sorted(zip(feature_labels, importances),
                               key=lambda x: x[1], reverse=True)
            for label, imp in imp_sorted:
                bar = '█' * int(imp * 50)
                print(f"    {label:<16s} {imp:>6.3f}  {bar}")

            # In-sample
            tree_probs = model.predict_proba(X)[:, 1]
            reg_df[f'{name}_prob'] = tree_probs

            for _, race_df in reg_df.groupby('race_id'):
                idx = race_df.index
                total = reg_df.loc[idx, f'{name}_prob'].sum()
                if total > 0:
                    reg_df.loc[idx, f'{name}_prob'] /= total

            m = evaluate_probs(reg_df, f'{name}_prob', name)
            print(f"  In-sample: ρ={m['spearman']:.4f}  LL={m['log_loss']:.4f}  "
                  f"TopPick={m['top_pick']:.1%}")

            # Cross-validated (same GroupKFold)
            cv_probs_tree = np.zeros(len(reg_df))
            for fold_i, (train_idx, test_idx) in enumerate(
                    gkf.split(X, y, card_ids)):
                m_cv_model = model.__class__(**model.get_params())
                m_cv_model.fit(X[train_idx], y[train_idx])
                cv_probs_tree[test_idx] = m_cv_model.predict_proba(
                    X[test_idx])[:, 1]

            reg_df[f'{name}_cv_prob'] = cv_probs_tree
            for _, race_df in reg_df.groupby('race_id'):
                idx = race_df.index
                total = reg_df.loc[idx, f'{name}_cv_prob'].sum()
                if total > 0:
                    reg_df.loc[idx, f'{name}_cv_prob'] /= total

            m_cv = evaluate_probs(reg_df, f'{name}_cv_prob', f'{name} CV')
            print(f"  Cross-val:  ρ={m_cv['spearman']:.4f}  LL={m_cv['log_loss']:.4f}  "
                  f"TopPick={m_cv['top_pick']:.1%}")

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"SUMMARY: What matters most?")
    print(f"{'='*65}")
    print(f"  Logistic coefficients = marginal value of each signal for P(win).")
    print(f"  Ridge coefficients    = marginal value of each signal for E[$].")
    print(f"  Compare the two: signals that hold up in ridge but not logistic")
    print(f"  are identifying value (overlays), not just winners.")


def _date_cv_splits(df):
    """Leave-one-date-out cross validation splits."""
    dates = sorted(df['date'].unique())
    for test_date in dates:
        train_dates = [d for d in dates if d != test_date]
        yield train_dates, [test_date]


def main():
    parser = argparse.ArgumentParser(description='Regression analysis of handicapping signals')
    parser.add_argument('--trees', action='store_true', help='Include tree-based models')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of CV folds (grouped by date+track card)')
    parser.add_argument('--C', type=float, default=None, dest='reg_C',
                        help='Regularization strength for logistic (default: auto-tune)')
    args = parser.parse_args()

    run_analysis(use_trees=args.trees, cv_folds=args.folds, reg_C=args.reg_C)


if __name__ == '__main__':
    main()