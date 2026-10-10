"""
payout_model.py

Conditional payout model — Model B in the v5 EV-ratio pipeline.

Trained ONLY on horses that won, with target = log1p(win_paid). Predicts
"if this horse wins, what does it pay?" — used downstream by predict_v5
to compute the EV ratio  P(win) * predicted_payout / 2.

Why conditional (winners only) rather than marginal (all horses, losers=0)?
---------------------------------------------------------------------------
The marginal version is what raw_value_model.json already does. The
trouble with marginal is that it conflates two things: "how often does
this horse win" and "what does it pay when it wins." A horse with high
P(win) and a horse with high payout-when-it-wins both look attractive to
a marginal model, but for very different reasons. The whole point of v5
is to factor those apart so we can multiply them: P(win) × E[pay | win].

Training data sparsity
----------------------
At ~280 winners across the historical sample, this model fits ~10 features
on ~250 examples per CV fold. That's tight — ridge with conservative
regularization is appropriate (linear model, no interactions). Surface
n_winners loudly on every train so future-you notices when it's getting
sparse.

Selection bias note
-------------------
Coefficients on this model are interpreted differently than on the
logistic. ML probability, for example, will likely have a NEGATIVE
coefficient — high-ML horses are favorites, favorites pay less when
they win. That's the model behaving correctly for a conditional target,
not a sign error. Don't compare coefficients across models without
remembering the conditioning.

Output
------
predict_v5 reads payout_model.json with the same shape as
ridge_value_model.json so the loader code can be reused:
    {
      "target": "log1p(win_paid)",
      "target_transform": "log1p",
      "feature_cols": [...],
      "feature_is_prob": [...],
      "input_space": "raw",
      "regularization_alpha": ...,
      "coefficients": [...],
      "intercept": ...,
      "n_winners_train": ...,
      "cv_rmse_dollars": ...,        # in $ space, after expm1
      "cv_top_pick_actual_pay": ...,  # avg actual win_paid for top-ranked winners
      "calibrated_at": "..."
    }
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')


# Default location is set by the caller via DATA_PATH; the functions
# accept explicit paths to stay testable.
PAYOUT_MODEL_FILENAME = 'payout_model.json'

# Conservative alpha grid. The training set is small (~250 winners), so
# even modest alphas keep coefficients honest. We don't grid-search down
# to 0.001 like log-ridge does because regularization on a small dataset
# matters more than fitting the noise.
DEFAULT_ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def train_payout_model(reg_df, feature_cols, feature_is_prob,
                       card_ids, model_path,
                       alpha_grid=None, n_folds=5, verbose=True):
    """Fit the conditional payout model on winners and persist it.

    Args:
        reg_df: the same training DataFrame that regression_analysis.py
            uses for the other models. Must contain 'won' (0/1) and
            'win_paid_num' columns alongside the feature columns.
        feature_cols: list of column names. Same as logistic/ridge use.
        feature_is_prob: parallel list of bools. True = the column is a
            probability that should be logit-transformed before fitting.
            False = use raw value (e.g. ml_x_fieldsize). Mirrors
            regression_analysis.py's convention.
        card_ids: array aligned with reg_df rows, one card_id per row.
            Used to build GroupKFold on the WINNERS subset (we group on
            card so a fold doesn't have train and test from the same
            card — same leakage concern as the other models).
        model_path: where to write payout_model.json.
        alpha_grid: list of alphas to search. Default is conservative.
        n_folds: CV folds for alpha selection. Capped at the number of
            unique cards in the winners subset.
        verbose: print progress.

    Returns:
        dict — the persisted payload (also written to disk).

    Raises:
        RuntimeError if there are too few winners to train (< 50). At
        that point CV folds are degenerate and coefficient estimates
        are noise.
    """
    if alpha_grid is None:
        alpha_grid = list(DEFAULT_ALPHA_GRID)

    # Subset to winners — that's the whole conditional setup.
    winners_mask = (reg_df['won'] == 1) & reg_df['win_paid_num'].notna() \
                   & (reg_df['win_paid_num'] > 0)
    winners = reg_df[winners_mask].copy()
    winner_card_ids = card_ids[winners_mask.values]

    n_winners = len(winners)
    if verbose:
        print(f"\n{'='*65}")
        print(f"PAYOUT MODEL — conditional on winning")
        print(f"  target = log1p(win_paid)  |  trained on winners only")
        print(f"{'='*65}")
        print(f"\n  n winners in training set: {n_winners}")

    if n_winners < 50:
        raise RuntimeError(
            f"Only {n_winners} winners in training data — too few to fit "
            f"a payout model. Need 50+. Either backfill more cards or "
            f"skip Model B and fall back to the v4 corner rule."
        )
    if n_winners < 150 and verbose:
        print(f"  ⚠  Small training set. Coefficients may be noisy. "
              f"Consider alpha ≥ 3 if R² is unstable across folds.")

    # Build feature matrix. Probability columns get logit-transformed
    # (matches regression_analysis.py's preprocessing for the logistic);
    # raw columns pass through. We deliberately use the SAME transform
    # the logistic uses so coefficients across models compare cleanly.
    clip_lo, clip_hi = 1e-4, 1 - 1e-4
    X_cols = []
    transformed = pd.DataFrame(index=winners.index)
    for col, is_prob in zip(feature_cols, feature_is_prob):
        if is_prob:
            t_col = f'{col}_logit'
            transformed[t_col] = np.log(
                winners[col].clip(clip_lo, clip_hi) /
                (1 - winners[col].clip(clip_lo, clip_hi))
            )
        else:
            t_col = col
            transformed[t_col] = winners[col]
        X_cols.append(t_col)

    X = transformed[X_cols].values
    # log1p on win_paid: $2 → 1.10, $10 → 2.40, $50 → 3.93, $200 → 5.30.
    # Without it, one $200 longshot dominates the gradient.
    y = np.log1p(winners['win_paid_num'].values)

    # CV setup — group by card so train/test don't overlap on a card.
    n_cards_winners = len(np.unique(winner_card_ids))
    actual_folds = max(2, min(n_folds, n_cards_winners))
    if verbose and actual_folds < n_folds:
        print(f"  Reducing folds {n_folds} → {actual_folds} "
              f"(only {n_cards_winners} cards have a winner with all features)")

    # Tune alpha by RMSE in DOLLAR space (after expm1). RMSE in log space
    # would penalize errors on small-payout horses just as much as on
    # huge-payout horses, but in dollar space we care more about big
    # misses on big-payout horses — that's the bet thesis.
    if verbose:
        print(f"\n  Tuning alpha via {actual_folds}-fold grouped CV "
              f"(group=card, target=log1p($win_paid))...")

    gkf = GroupKFold(n_splits=actual_folds)
    best_alpha = alpha_grid[0]
    best_rmse_dollars = np.inf

    # Cache fold predictions for the best alpha so we don't refit later.
    best_fold_preds = None
    best_fold_models = None

    for alpha in alpha_grid:
        preds_log = np.zeros(len(winners))
        fold_models = []
        for train_idx, test_idx in gkf.split(X, y, winner_card_ids):
            rr = Ridge(alpha=alpha)
            rr.fit(X[train_idx], y[train_idx])
            preds_log[test_idx] = rr.predict(X[test_idx])
            fold_models.append(rr)

        preds_dollars = np.expm1(preds_log)
        actual_dollars = np.expm1(y)
        rmse_dollars = float(np.sqrt(np.mean((actual_dollars - preds_dollars) ** 2)))

        marker = ' ← best' if rmse_dollars < best_rmse_dollars else ''
        if verbose:
            print(f"    α={alpha:<6g}  RMSE=${rmse_dollars:6.2f}  "
                  f"mean_pred=${preds_dollars.mean():.2f}  "
                  f"mean_actual=${actual_dollars.mean():.2f}{marker}")
        if rmse_dollars < best_rmse_dollars:
            best_rmse_dollars = rmse_dollars
            best_alpha = alpha
            best_fold_preds = preds_log.copy()
            best_fold_models = fold_models

    if verbose:
        print(f"  Best α={best_alpha}  (CV RMSE=${best_rmse_dollars:.2f} in $ space)")

    # Average fold coefficients — same approach as the other models so
    # production uses the average rather than a refit on all data.
    # Refitting on all data would be slightly higher-variance and would
    # leak in subtle ways during walk-forward backtests.
    avg_coefs = np.mean([m.coef_ for m in best_fold_models], axis=0)
    avg_intercept = float(np.mean([m.intercept_ for m in best_fold_models]))

    if verbose:
        print(f"\n  Averaged ridge coefficients ({actual_folds} folds):")
        total = sum(abs(c) for c in avg_coefs)
        for col, coef in zip(X_cols, avg_coefs):
            pct = abs(coef) / total * 100 if total > 0 else 0
            sign = '+' if coef > 0 else '-'
            bar = '█' * int(abs(coef) * 5)
            print(f"    {col:<26s} {coef:>+8.4f}  ({pct:4.1f}%)  {sign}{bar}")
        print(f"    {'intercept':<26s} {avg_intercept:>+8.4f}")
        print(f"\n  Reminder: target is win_paid CONDITIONAL on winning.")
        print(f"  ML probability often has a NEGATIVE coefficient here —")
        print(f"  favorites pay less when they win. That's correct behavior.")

    # Quick sanity diagnostic — does ranking winners by predicted payout
    # actually rank them by actual payout? Spearman across the OOS preds.
    from scipy.stats import spearmanr
    rho, _ = spearmanr(np.expm1(best_fold_preds), np.expm1(y))
    if verbose:
        print(f"\n  Diagnostic: Spearman(predicted, actual) on winners = {rho:+.4f}")
        if rho < 0:
            print(f"  ⚠  Negative correlation. Model is anti-predictive. "
                  f"Investigate before using for live betting.")
        elif rho < 0.1:
            print(f"  ⚠  Weak correlation. Model has little signal beyond "
                  f"the mean. Live EV will be noisy.")

    payload = {
        'target': 'log1p(win_paid)',
        'target_transform': 'log1p',
        'conditional_on': 'won == 1',
        'feature_cols': feature_cols,
        'feature_is_prob': feature_is_prob,
        'input_space': 'mixed',  # prob features logit-transformed, others raw
        'regularization_alpha': float(best_alpha),
        'coefficients': avg_coefs.tolist(),
        'intercept': avg_intercept,
        'n_folds': actual_folds,
        'n_winners_train': int(n_winners),
        'cv_rmse_dollars': best_rmse_dollars,
        'cv_spearman': float(rho) if not np.isnan(rho) else None,
        'cv_mean_pred_dollars': float(np.expm1(best_fold_preds).mean()),
        'cv_mean_actual_dollars': float(np.expm1(y).mean()),
        'calibrated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'w') as f:
        json.dump(payload, f, indent=2)
    if verbose:
        print(f"\n  Payout model saved to {model_path}")

    return payload, best_fold_preds, winners_mask


def load_payout_model(model_path):
    """Load and validate a persisted payout model. Raises clearly if missing."""
    p = Path(model_path)
    if not p.exists():
        raise RuntimeError(
            f"payout_model.json not found at {p}. Train it by re-running "
            f"regression_analysis.py (which now also fits Model B), or "
            f"see payout_model.train_payout_model for direct usage."
        )
    payload = json.loads(p.read_text())
    required = {'feature_cols', 'feature_is_prob', 'coefficients',
                'intercept', 'target_transform'}
    missing = required - set(payload.keys())
    if missing:
        raise RuntimeError(
            f"payout_model.json is missing required keys: {missing}. "
            f"Re-train with the current regression_analysis.py."
        )
    if payload.get('target_transform') != 'log1p':
        raise RuntimeError(
            f"Unexpected target_transform={payload.get('target_transform')!r}. "
            f"Predict pipeline assumes log1p; expm1 conversion will be wrong "
            f"for any other transform."
        )
    return payload


# Column name mapping: regression-side names → predict_v2 df names.
# regression_analysis.py uses short names like 'claude_prob' but predict_v2's
# DataFrame (and the walk_forward_summary.csv that flows from it) uses
# longer names like 'claude_win_prob'. The mapping mirrors the one in
# predict_v2._apply_linear_model (lines ~1242-1256). Kept duplicated rather
# than imported because importing predict_v2 from here would create a
# circular import (predict_v2 → ridge_value_model → ... and predict_v5
# imports both predict_v2 and payout_model).
#
# Any name not in this map is used as-is (e.g. 'human_prob' is the same
# in both schemas, as are the *_beyer_sim columns).
_COL_MAP = {
    'ml_prob_norm':       'ml_win_prob_norm',
    'claude_prob':        'claude_win_prob',
    'gemini_prob':        'gemini_win_prob',
    'openai_prob':        'openai_win_prob',
    'claude_closing':     'claude_closing_prob',
    'gemini_closing':     'gemini_closing_prob',
    'openai_closing':     'openai_closing_prob',
}


def _resolve_col(model_col, df_columns):
    """Return the actual df column name for a model feature name.

    Tries the model name first (regression-side schema), then the mapped
    name (predict_v2 schema). Returns None if neither is present.
    """
    if model_col in df_columns:
        return model_col
    mapped = _COL_MAP.get(model_col)
    if mapped is not None and mapped in df_columns:
        return mapped
    return None


def predict_payout(df, model):
    """Apply the loaded payout model to a prediction DataFrame.

    Returns the original df with a new 'predicted_payout' column (in
    DOLLAR space — already expm1'd from log space). Horses missing any
    feature get NaN, which propagates downstream so they're excluded
    from the bet rule.

    Handles two df schemas:
      - regression-style (from regression_analysis): columns named
        'claude_prob', 'ml_prob_norm', etc. Already per-race normalized.
      - predict_v2-style (from predict_v2 pipeline or walk_forward_summary):
        columns named 'claude_win_prob', 'ml_win_prob_norm', etc. Prob
        features are NOT per-race normalized — predict_v2 normalizes them
        inside _apply_linear_model just before applying coefficients.
        We replicate that normalization here so the model sees the same
        feature distribution it was trained on.

    Args:
        df: prediction df. Must have all the model's feature columns
            under either their regression name or their predict_v2 name.
        model: dict from load_payout_model().
    """
    feature_cols = model['feature_cols']
    feature_is_prob = model['feature_is_prob']
    coefs = np.array(model['coefficients'])
    intercept = model['intercept']

    # Resolve each model feature to an actual df column.
    resolved = []
    unresolved = []
    for fc in feature_cols:
        actual = _resolve_col(fc, df.columns)
        if actual is None:
            unresolved.append(fc)
        resolved.append(actual)

    if unresolved:
        raise RuntimeError(
            f"Prediction df is missing payout-model feature columns: "
            f"{unresolved}. Payout model was trained with: {feature_cols}. "
            f"Tried both regression-style and predict_v2-style names. "
            f"Either retrain on a feature subset the df actually has, or "
            f"ensure predict_v2._ensure_regression_features ran on the df "
            f"(it builds *_beyer_sim, human_prob, ml_x_fieldsize)."
        )

    # Per-race normalize prob features that came from predict_v2 schema.
    # The regression-side schema already has them normalized (load_data
    # in regression_analysis.py does it); the predict_v2 schema does not.
    # Detect the schema by checking whether at least one prob feature
    # used the mapped name. Cheap heuristic but correct in practice
    # because the two schemas are distinct — you don't have one df with
    # 'claude_prob' AND 'gemini_win_prob'.
    needs_normalize = False
    for fc, actual in zip(feature_cols, resolved):
        if actual != fc and _COL_MAP.get(fc) == actual:
            needs_normalize = True
            break

    df = df.copy()
    if needs_normalize:
        # Determine the race-grouping columns. A live predict_v2 df has
        # one card so 'race' alone uniquely identifies a race. A
        # walk_forward_summary df has many cards — grouping by 'race'
        # alone would pool every card's Race 1 together, producing
        # ratios near 1/N_cards instead of 1/field_size and blowing
        # up the calibrated payouts (was producing $75+ predictions
        # vs the correct ~$15). Use whichever combo of (date, track,
        # race) the df actually has.
        group_cols = [c for c in ('date', 'track', 'race') if c in df.columns]
        if 'race' not in group_cols:
            raise RuntimeError(
                "Cannot per-race normalize: df has no 'race' column. "
                "predict_payout needs at least 'race', and ideally "
                "'date'+'track'+'race' for multi-card frames."
            )

        # Normalize each prob feature per race, matching predict_v2's
        # behavior in _apply_linear_model (lines ~1259-1271).
        # Cast to float64 first so the subsequent float assignment doesn't
        # trip pandas LossySetitemError on int-typed columns (defensive —
        # predict_v2 writes floats, but walk_forward_summary.csv parsing
        # could in theory produce int columns for clean integer probabilities).
        for fc, actual, is_prob in zip(feature_cols, resolved, feature_is_prob):
            if not is_prob:
                continue
            if df[actual].dtype != np.float64:
                df[actual] = df[actual].astype(float)
            for _, race_df in df.groupby(group_cols):
                idx = race_df.index
                vals = df.loc[idx, actual].values
                valid = ~np.isnan(vals)
                if valid.any() and np.nansum(vals[valid]) > 0:
                    df.loc[idx, actual] = np.where(
                        valid, vals / np.nansum(vals), np.nan
                    )

    # Build feature matrix. Probability features → logit space; raw stay raw.
    clip_lo, clip_hi = 1e-4, 1 - 1e-4
    X_rows = []
    for actual, is_prob in zip(resolved, feature_is_prob):
        if is_prob:
            v = df[actual].clip(clip_lo, clip_hi)
            X_rows.append(np.log(v / (1 - v)).values)
        else:
            X_rows.append(df[actual].values)
    X = np.column_stack(X_rows)

    # Linear prediction in log space, expm1 to dollar space.
    # NaN in any feature naturally propagates to NaN prediction, which
    # excludes the horse from downstream bet selection.
    pred_log = X @ coefs + intercept
    pred_dollars = np.expm1(pred_log)

    df['predicted_payout'] = pred_dollars
    return df