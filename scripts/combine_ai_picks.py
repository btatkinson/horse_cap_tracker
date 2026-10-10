"""
combine_ai_picks.py

Merge per-provider AI pick CSVs (claude.csv, gemini.csv, openai.csv)
into a single ai_combined.csv with cross-provider imputation for missing
values.

The merge does an outer join on (race, horse_number) so any horse that
appears in at least one provider's CSV gets a row. Missing columns are
filled via sklearn's IterativeImputer — e.g. gemini_beyer NaN is filled
by regressing on (claude_beyer, openai_beyer, *_win_prob, *_closing_prob)
where those are known. The imputed values are written to disk so predict_v2
and regression_analysis can read them directly — this makes data complete
AND auditable (you can inspect ai_combined.csv to see what got filled).

After imputation, horses that still have NaN in any critical column
(*_beyer or *_win_prob for any provider that's present for OTHER horses)
are dropped with a loud error.

Output:
    {DATA_PATH}/ai_picks/{date}/{track}/ai_combined.csv

Columns: race, horse_number, date, track,
         {provider}_beyer, {provider}_win_prob, {provider}_closing_prob
         (for each provider whose CSV existed)

Usage:
    python combine_ai_picks.py                     # process all dates/tracks
    python combine_ai_picks.py --date 2026-04-11
    python combine_ai_picks.py --date 2026-04-11 --track keeneland
    python combine_ai_picks.py --force             # overwrite existing ai_combined
"""

import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
PICKS_DIR = DATA_PATH / 'ai_picks'

PROVIDERS = ['claude', 'gemini', 'openai']


def _load_provider(csv_path: Path, provider: str):
    """Load a provider CSV and normalize column names.

    Input columns vary by provider (e.g. 'gemini_expected_beyer', 'beyer',
    'closing_odds', etc.); this normalizes to the canonical set
    {provider}_beyer, {provider}_win_prob, {provider}_closing_prob.
    Returns None if the file can't be loaded or is empty.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    ERROR loading {csv_path.name}: {e}")
        return None

    if len(df) == 0:
        return None

    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if 'beyer' in cl:
            col_map[c] = f'{provider}_beyer'
        elif 'closing' in cl:
            col_map[c] = f'{provider}_closing_prob'
        elif 'win_prob' in cl or cl == 'prob' or cl.endswith('_prob'):
            col_map[c] = f'{provider}_win_prob'
    df = df.rename(columns=col_map)

    if 'race' not in df.columns or 'horse_number' not in df.columns:
        print(f"    ERROR {csv_path.name}: missing race or horse_number")
        return None

    df['race'] = pd.to_numeric(df['race'], errors='coerce').astype('Int64')
    df['horse_number'] = pd.to_numeric(df['horse_number'],
                                        errors='coerce').astype('Int64')

    # Keep only canonical columns + keys
    keep = ['race', 'horse_number']
    for suffix in ('_beyer', '_win_prob', '_closing_prob'):
        col = f'{provider}{suffix}'
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            keep.append(col)

    return df[keep].drop_duplicates(['race', 'horse_number'])


def _impute(df, feature_cols):
    """
    Fill NaNs in feature_cols via sklearn IterativeImputer. Returns
    (imputed_df, report) where report is dict of {col: n_filled}.

    Skips imputation if sklearn unavailable, if fewer than 5 rows have
    any signal, or if only one feature column has data.
    """
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    except ImportError:
        print("    sklearn IterativeImputer unavailable; skipping imputation")
        return df, {}

    if len(feature_cols) < 2:
        return df, {}

    sub = df[feature_cols].copy()
    before_missing = {c: int(sub[c].isna().sum()) for c in feature_cols}
    if sum(before_missing.values()) == 0:
        return df, {}

    n_usable = sub.dropna(how='all').shape[0]
    if n_usable < 5:
        print(f"    only {n_usable} rows have any signal; skipping imputation")
        return df, {}

    try:
        imputer = IterativeImputer(max_iter=10, random_state=42)
        filled = imputer.fit_transform(sub.values)
    except Exception as e:
        print(f"    IterativeImputer failed: {type(e).__name__}: {e}")
        return df, {}

    filled_df = pd.DataFrame(filled, columns=feature_cols, index=sub.index)

    # Track which columns actually got filled (for reporting)
    report = {}
    for c in feature_cols:
        filled_count = before_missing[c] - int(filled_df[c].isna().sum())
        if filled_count > 0:
            report[c] = filled_count

    df = df.copy()
    for c in feature_cols:
        df[c] = filled_df[c].values

    return df, report



def load_card(date_str: str, track: str):
    """
    Load and outer-merge per-provider CSVs for one card. Returns the
    merged DataFrame with a 'card' column marking (date, track), or None
    if no providers could be loaded.
    """
    track_dir = PICKS_DIR / date_str / track
    if not track_dir.exists():
        return None

    frames = {}
    for p in PROVIDERS:
        df = _load_provider(track_dir / f'{p}.csv', p)
        if df is not None:
            frames[p] = df

    if not frames:
        return None

    names = list(frames.keys())
    merged = frames[names[0]]
    for n in names[1:]:
        merged = merged.merge(frames[n], on=['race', 'horse_number'],
                            how='outer')

    merged['date'] = date_str
    merged['track'] = track
    merged['_card'] = f'{date_str}/{track}'  # marker for per-card split later
    return merged


def combine_all(targets, out_targets, force=False, verbose=False):
    """
    Load ALL cards in `targets`, fit one global IterativeImputer across
    them, then write ai_combined.csv for each card in `out_targets`.

    Pooling the imputer fit across cards gives far better coefficients
    than per-card fitting (thousands of rows to learn claude↔gemini↔openai
    relationships vs 80 rows per card). The relationships we're learning —
    e.g. 'gemini_beyer tends to be 2 points lower than claude_beyer when
    openai agrees' — are stable across tracks/distances/days, so pooling
    is actually the right thing to do.

    Args:
        targets: list of (date, track) pairs to LOAD (for imputer fit context)
        out_targets: list of (date, track) pairs to WRITE output for.
                     Usually equal to targets, but can be a subset when the
                     user only wants to regenerate specific cards.
    """
    # --- Load all targets ---
    print(f"Loading {len(targets)} cards...")
    card_dfs = []
    skipped = []
    for date_str, track in targets:
        df = load_card(date_str, track)
        if df is None:
            skipped.append((date_str, track, 'no provider CSVs'))
            continue
        card_dfs.append(df)

    if not card_dfs:
        print("No cards loaded; nothing to do.")
        return 0, len(skipped)

    pooled = pd.concat(card_dfs, ignore_index=True)
    print(f"  Pooled {len(pooled)} rows across {len(card_dfs)} cards")

    # Collect feature columns present anywhere in the pool
    feature_cols = []
    for p in PROVIDERS:
        for suffix in ('_beyer', '_win_prob', '_closing_prob'):
            col = f'{p}{suffix}'
            if col in pooled.columns:
                feature_cols.append(col)

    # Identify pre-impute all-NaN rows (for later drop) — must happen BEFORE
    # imputation, because the imputer will fill all-NaN rows with column
    # means, making them look valid.
    core_cols = [c for c in feature_cols
                 if c.endswith('_beyer') or c.endswith('_win_prob')]
    if core_cols:
        pooled['_all_nan_pre'] = pooled[core_cols].isna().all(axis=1)
    else:
        pooled['_all_nan_pre'] = False

    # --- Fit one global imputer ---
    imputed, report = _impute(pooled, feature_cols)

    if report:
        filled_total = sum(report.values())
        print(f"  Global imputer filled {filled_total} values across all cards:")
        if verbose:
            for col, n in sorted(report.items()):
                print(f"    {col}: {n} filled")
        else:
            # one-liner summary per column when not verbose
            by_prov = {}
            for col, n in report.items():
                provider = col.split('_')[0]
                by_prov.setdefault(provider, 0)
                by_prov[provider] += n
            for prov, n in sorted(by_prov.items()):
                print(f"    {prov}: {n} filled")

    # --- Split back into per-card, validate, write ---
    out_set = set(out_targets)
    n_ok = 0
    n_fail = 0

    # Compute critical cols once (same for all cards)
    critical_cols = []
    for p in PROVIDERS:
        for suffix in ('_beyer', '_win_prob'):
            col = f'{p}{suffix}'
            if col in imputed.columns and imputed[col].notna().any():
                critical_cols.append(col)

    for card_id in sorted(imputed['_card'].unique()):
        date_str, track = card_id.split('/', 1)
        if (date_str, track) not in out_set:
            continue  # only loaded this card for imputer context

        track_dir = PICKS_DIR / date_str / track
        out_path = track_dir / 'ai_combined.csv'
        if out_path.exists() and not force:
            if verbose:
                print(f"  {card_id}: ai_combined.csv exists (skip; --force to overwrite)")
            n_ok += 1
            continue

        card_df = imputed[imputed['_card'] == card_id].copy()

        # Drop conditions — same two as before, now evaluated per-card
        # for the error messaging.
        all_nan_pre = card_df['_all_nan_pre']
        post_impute_nan = pd.Series([False] * len(card_df), index=card_df.index)
        if critical_cols:
            post_impute_nan = card_df[critical_cols].isna().any(axis=1)

        drop_mask = all_nan_pre | post_impute_nan
        n_dropped = int(drop_mask.sum())

        if n_dropped > 0:
            dropped = card_df[drop_mask]
            print(f"  {card_id}: DROPPING {n_dropped} horses:")
            for idx, row in dropped.iterrows():
                r, h = row['race'], row['horse_number']
                reasons = []
                if all_nan_pre.loc[idx]:
                    reasons.append("no signal from any provider")
                if post_impute_nan.loc[idx]:
                    nan_cols = [c for c in critical_cols if pd.isna(row[c])]
                    reasons.append(f"still-NaN post-impute: {', '.join(nan_cols)}")
                print(f"    R{r} #{h}: {'; '.join(reasons)}")
            card_df = card_df[~drop_mask]

        # Drop helper columns and write
        card_df = card_df.drop(columns=['_card', '_all_nan_pre'],
                              errors='ignore')
        col_order = ['date', 'track', 'race', 'horse_number']
        col_order += [c for c in card_df.columns if c not in col_order]
        card_df = card_df[col_order]
        card_df.sort_values(['race', 'horse_number']).to_csv(out_path, index=False)

        if verbose or n_dropped > 0:
            print(f"  {card_id}: {len(card_df)} rows written to ai_combined.csv")
        n_ok += 1

    return n_ok, n_fail + len(skipped)


def main():
    parser = argparse.ArgumentParser(
        description='Merge per-provider AI picks into ai_combined.csv with imputation')
    parser.add_argument('--date', type=str, default=None,
                        help='Process only this date (e.g. 2026-04-11)')
    parser.add_argument('--track', type=str, default=None,
                        help='Process only this track (requires --date)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing ai_combined.csv files')
    parser.add_argument('--verbose', action='store_true',
                        help='Per-column imputation detail + per-card status')
    parser.add_argument('--local-fit', action='store_true',
                        help='Fit the imputer only on the targeted cards '
                             '(not all historical cards). Faster but noisier '
                             '— use only when you know historical cards are '
                             "not relevant or when you're debugging.")
    args = parser.parse_args()

    if not PICKS_DIR.exists():
        print(f"No ai_picks directory at {PICKS_DIR}")
        return

    # Discover ALL (date, track) combos. We need the full set for the
    # imputer fit, even when the user only wants to write a subset.
    all_targets = []
    for date_dir in sorted(PICKS_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        for track_dir in sorted(date_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            all_targets.append((date_dir.name, track_dir.name))

    if not all_targets:
        print(f"No cards found under {PICKS_DIR}")
        return

    # The subset the user actually wants written
    out_targets = all_targets
    if args.date:
        out_targets = [(d, t) for (d, t) in out_targets if d == args.date]
    if args.track:
        out_targets = [(d, t) for (d, t) in out_targets if t == args.track]

    if not out_targets:
        print(f"No matching date/track combos to write")
        if args.date:
            print(f"  (filter: --date {args.date}"
                 f"{' --track ' + args.track if args.track else ''})")
        return

    fit_targets = out_targets if args.local_fit else all_targets

    if args.local_fit:
        print(f"Local fit: imputer will see only the {len(fit_targets)} target card(s)")
    elif len(out_targets) < len(all_targets):
        print(f"Writing {len(out_targets)} card(s); imputer fits on all "
              f"{len(all_targets)} for stability")

    n_ok, n_fail = combine_all(fit_targets, out_targets,
                               force=args.force, verbose=args.verbose)
    print(f"\n{n_ok} succeeded, {n_fail} failed")


if __name__ == '__main__':
    main()