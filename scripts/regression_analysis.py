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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
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

            # Load claude and gemini
            frames = {}
            for provider in ['claude', 'gemini']:
                csv = track_dir / f'{provider}.csv'
                if csv.exists():
                    pdf = pd.read_csv(csv)
                    # Rename columns
                    col_map = {}
                    for c in pdf.columns:
                        cl = c.lower().strip()
                        if 'beyer' in cl:
                            col_map[c] = f'{provider}_beyer'
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

                    # Expert's Elo-based multiplier (optional)
                    elo = ratings.get(expert_name, {}).get('rating', 1500)
                    elo_mult = max(0.5, (elo - 1400) / 200)  # 1500→0.5, 1600→1.0, 1700→1.5

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


def run_analysis(use_trees=False, cv_folds=5):
    """Run regression analysis to find optimal signal weights."""

    df = load_data()
    if df is None or df.empty:
        return

    print(f"Loaded {len(df)} horses across {df.groupby(['date','track','race']).ngroups} races")

    # ── Define features ──
    feature_cols = []
    feature_labels = []

    for col, label in [
        ('ml_prob_norm', 'ML'),
        ('claude_prob', 'Claude Prob'),
        ('gemini_prob', 'Gemini Prob'),
        ('claude_beyer', 'Claude Beyer'),
        ('gemini_beyer', 'Gemini Beyer'),
        ('human_prob', 'Human Pool'),
    ]:
        if col in df.columns and df[col].notna().any():
            feature_cols.append(col)
            feature_labels.append(label)

    print(f"Features: {', '.join(feature_labels)}")

    # ── Individual signal evaluation ──
    print(f"\n{'='*65}")
    print(f"INDIVIDUAL SIGNAL PERFORMANCE")
    print(f"{'='*65}")
    print(f"  {'Signal':<16s} {'Spearman':>9s} {'LogLoss':>8s} {'TopPick':>8s}")
    print(f"  {'-'*16} {'-'*9} {'-'*8} {'-'*8}")

    for col, label in zip(feature_cols, feature_labels):
        # Normalize per race for fair comparison
        norm_col = f'{col}_norm'
        df[norm_col] = np.nan
        for _, race_df in df.groupby(['date', 'track', 'race']):
            idx = race_df.index
            vals = df.loc[idx, col].values
            if col == 'claude_beyer' or col == 'gemini_beyer':
                # Higher beyer = better, convert to pseudo-prob
                valid = ~np.isnan(vals)
                if valid.any():
                    shifted = vals - np.nanmin(vals) + 1
                    df.loc[idx, norm_col] = np.where(valid, shifted / np.nansum(shifted), np.nan)
            else:
                valid = ~np.isnan(vals)
                if valid.any() and np.nansum(vals[valid]) > 0:
                    df.loc[idx, norm_col] = np.where(valid, vals / np.nansum(vals), np.nan)

        m = evaluate_probs(df, norm_col, label)
        rho_s = f"{m['spearman']:.4f}" if not np.isnan(m['spearman']) else "n/a"
        ll_s = f"{m['log_loss']:.4f}" if not np.isnan(m['log_loss']) else "n/a"
        tp_s = f"{m['top_pick']:.1%}" if not np.isnan(m['top_pick']) else "n/a"
        print(f"  {label:<16s} {rho_s:>9s} {ll_s:>8s} {tp_s:>8s}")

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

    X = reg_df[feature_cols].values
    y = reg_df['won'].values

    # Race groups for leave-one-group-out CV
    reg_df['race_id'] = reg_df['date'] + '_' + reg_df['track'] + '_' + reg_df['race'].astype(str)
    groups = reg_df['race_id'].values

    # ── Logistic Regression ──
    print(f"\n{'='*65}")
    print(f"LOGISTIC REGRESSION (features → P(win))")
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

    # Raw (unscaled) coefficients for interpretability
    lr_raw = LogisticRegression(max_iter=1000, C=1.0)
    lr_raw.fit(X, y)
    print(f"\n  Raw coefficients:")
    for label, coef in zip(feature_labels, lr_raw.coef_[0]):
        print(f"    {label:<16s} {coef:>+8.4f}")
    print(f"    {'intercept':<16s} {lr_raw.intercept_[0]:>+8.4f}")

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

    # ── K-Fold Grouped CV with Model Averaging ──
    from sklearn.model_selection import GroupKFold

    # Group by date+track (each card is a group)
    reg_df['card_id'] = reg_df['date'] + '_' + reg_df['track']
    card_ids = reg_df['card_id'].values
    unique_cards = np.unique(card_ids)
    n_cards = len(unique_cards)

    actual_folds = min(cv_folds, n_cards)
    print(f"\n  {actual_folds}-Fold Grouped CV ({n_cards} cards):")

    gkf = GroupKFold(n_splits=actual_folds)
    cv_probs = np.zeros(len(reg_df))
    fold_models = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, card_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        lr_fold = LogisticRegression(max_iter=1000, C=1.0)
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

    # ── Per-race CV predictions output ──
    out_cols = ['date', 'track', 'race', 'horse_number', 'cv_prob',
                'ml_prob_norm', 'won', 'finish']
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
    print(f"  The logistic regression coefficients show the marginal value")
    print(f"  of each signal after accounting for the others.")
    print(f"  A near-zero coefficient means that signal is redundant.")
    print(f"  A negative coefficient means it hurts after accounting for others.")


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
    args = parser.parse_args()

    run_analysis(use_trees=args.trees, cv_folds=args.folds)


if __name__ == '__main__':
    main()