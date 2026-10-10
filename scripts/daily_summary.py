"""
daily_summary.py

Grade a day's predictions and bets against actual results.
Shows how the model did, what bets hit/missed, and running totals.

Usage:
    python daily_summary.py --date 2026-04-08 --track keeneland
    python daily_summary.py --date 2026-04-08 --track keeneland --bankroll 500

Reads from:
    {DATA_PATH}/predictions/{date}/{track}/predictions.csv
    {DATA_PATH}/predictions/{date}/{track}/bet_log.csv
    {DATA_PATH}/entries/*.csv   (results)

Writes to:
    {DATA_PATH}/daily_summaries/{date}_{track}.txt
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])

_TRACK_ALIASES = {
    'gulfstream-park': 'gulfstream-park', 'gulfstream park': 'gulfstream-park', 'gulfstream': 'gulfstream-park',
    'churchill-downs': 'churchill-downs', 'churchill downs': 'churchill-downs', 'churchill': 'churchill-downs',
    'santa-anita': 'santa-anita', 'santa anita': 'santa-anita',
    'del-mar': 'del-mar', 'del mar': 'del-mar',
    'fair-grounds': 'fair-grounds', 'fair grounds': 'fair-grounds',
    'tampa-bay': 'tampa', 'tampa bay': 'tampa', 'tampa bay downs': 'tampa',
    'oaklawn-park': 'oaklawn', 'oaklawn park': 'oaklawn',
    'laurel-park': 'laurel', 'laurel park': 'laurel',
    'belmont-park': 'belmont', 'belmont park': 'belmont',
}

def _normalize_track(t):
    if pd.isna(t): return t
    t = str(t).strip().lower()
    return _TRACK_ALIASES.get(t, t)


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


# ============================================================================
# Exotics meta loading (for exacta box grading)
# ============================================================================
# Mirrors the structure used by backtest_kfold.py so behavior is consistent.
# Meta CSVs live at {DATA_PATH}/meta/{track}_{date}.csv with columns including
# exacta_finish ("3-2") and exacta_$2_payout ("$53.26"). Cached per (track,
# date) so repeated lookups within one day's summary don't re-read the file.

META_DIR = DATA_PATH / 'meta'
_META_CACHE = {}

# Reverse aliases: when the meta file may be saved under a different track
# slug than what daily_summary normalized to. Mirrors backtest_kfold.
_META_REVERSE_ALIASES = {
    'gulfstream-park': ['gulfstream', 'gulfstream park'],
    'gulfstream': ['gulfstream-park', 'gulfstream park'],
    'tampa': ['tampa-bay', 'tampa-bay-downs', 'tampa bay downs'],
    'oaklawn': ['oaklawn-park', 'oaklawn park'],
    'laurel': ['laurel-park', 'laurel park'],
    'belmont': ['belmont-park', 'belmont park'],
}


def _meta_path_candidates(track, date):
    paths = [META_DIR / f'{track}_{date}.csv']
    for variant in _META_REVERSE_ALIASES.get(track, []):
        paths.append(META_DIR / f'{variant}_{date}.csv')
    return paths


def _parse_exacta_payout(val):
    """Parse '$53.26' → 53.26. Empty/NA → 0.0."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if s == '' or s.lower() in ('nan', 'none', '-'):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_exacta_finish(val):
    """Parse '3-2' → (3, 2). Empty/NA → None."""
    if pd.isna(val) or str(val).strip() == '':
        return None
    try:
        return tuple(int(x) for x in str(val).strip().split('-'))
    except (ValueError, TypeError):
        return None


def load_race_exacta(track, date_str, race):
    """
    Look up exacta finish + payout for one race. Returns:
        {'exacta_finish': (h1, h2), 'exacta_payout': float}  on success
        {'_status': 'no_meta_file'} if no meta CSV anywhere
        {'_status': 'no_race_row'}  if meta exists but race missing
        {'_status': 'no_exacta'}    if race exists but no exacta data

    The caller can use _status to attribute "skipped" reasons in the
    summary, the same way backtest_kfold does.
    """
    cache_key = (track, date_str)
    if cache_key not in _META_CACHE:
        loaded = None
        for path in _meta_path_candidates(track, date_str):
            if path.exists():
                try:
                    loaded = pd.read_csv(path)
                    break
                except Exception:
                    continue  # try next alias
        _META_CACHE[cache_key] = loaded

    meta = _META_CACHE[cache_key]
    if meta is None:
        return {'_status': 'no_meta_file'}

    row = meta[meta['race'] == race]
    if row.empty:
        return {'_status': 'no_race_row'}

    r = row.iloc[0]
    finish = _parse_exacta_finish(r.get('exacta_finish'))
    if finish is None:
        return {'_status': 'no_exacta'}

    return {
        'exacta_finish': finish,
        'exacta_payout': _parse_exacta_payout(r.get('exacta_$2_payout')),
    }


def load_results(date_str, track):
    """Load results from entries for this date/track."""
    entries_dir = DATA_PATH / 'entries'
    if not entries_dir.exists():
        return None

    dfs = []
    for f in entries_dir.glob('*.csv'):
        try:
            df = pd.read_csv(f)
            if 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            if 'track' in df.columns:
                df['track'] = df['track'].apply(_normalize_track)
            mask = df['date'] == date_str
            if 'track' in df.columns:
                mask = mask & (df['track'] == track)
            subset = df[mask]
            if len(subset) > 0:
                dfs.append(subset)
        except Exception:
            continue

    if not dfs:
        return None

    result = pd.concat(dfs, ignore_index=True)
    if 'running' in result.columns:
        result = result[result['running'] == 1].copy()
    if 'horse' in result.columns and 'horse_number' not in result.columns:
        result['horse_number'] = pd.to_numeric(result['horse'], errors='coerce')
    if 'finish' in result.columns:
        result['finish'] = pd.to_numeric(result['finish'], errors='coerce')
        result['won'] = (result['finish'] == 1).astype(int)
    for col in ['win_paid', 'place_paid', 'show_paid']:
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col].astype(str).str.replace(r'[^0-9.]', '', regex=True),
                errors='coerce')
    if 'ml_odds' in result.columns:
        result['ml_prob'] = result['ml_odds'].apply(_parse_ml_odds)

    return result


def run_summary(date_str, track, bankroll=None):
    """Generate daily summary."""
    pred_path = DATA_PATH / 'predictions' / date_str / track / 'predictions.csv'
    bet_path = DATA_PATH / 'predictions' / date_str / track / 'bet_log.csv'

    if not pred_path.exists():
        print(f"No predictions found for {date_str} / {track}")
        print(f"  Run: python predict_races.py --date {date_str} --track {track}")
        return

    preds = pd.read_csv(pred_path)
    results = load_results(date_str, track)

    if results is None:
        print(f"No results found for {date_str} / {track}")
        return

    lines = []
    def out(s=''):
        lines.append(s)
        print(s)

    out(f"{'='*72}")
    out(f"DAILY SUMMARY: {date_str} / {track}")
    out(f"{'='*72}")

    # Merge predictions with results
    merge_cols = ['race', 'horse_number']
    result_cols = ['race', 'horse_number', 'finish', 'won',
                   'win_paid', 'place_paid', 'show_paid']
    result_cols = [c for c in result_cols if c in results.columns]
    if 'horse_name' in results.columns:
        result_cols.append('horse_name')

    merged = preds.merge(results[result_cols].drop_duplicates(),
                         on=merge_cols, how='left')

    n_races = merged['race'].nunique()
    n_horses = len(merged)
    n_with_results = merged['finish'].notna().sum()

    out(f"\n  {n_races} races, {n_horses} horses ({n_with_results} with results)")

    # ── Model accuracy ──────────────────────────────────────────────
    out(f"\n{'─'*40}")
    out(f"MODEL ACCURACY")
    out(f"{'─'*40}")

    # Spearman per race
    spearman_vals = []
    for race_num, race_df in merged.groupby('race'):
        valid = race_df.dropna(subset=['ensemble_prob', 'finish'])
        if len(valid) >= 3:
            rho, _ = spearmanr(valid['ensemble_prob'], -valid['finish'])
            spearman_vals.append(rho)

    if spearman_vals:
        out(f"  Mean Spearman ρ:    {np.mean(spearman_vals):.4f}")
        out(f"  Median Spearman ρ:  {np.median(spearman_vals):.4f}")

    # Top pick win rate
    top_picks = []
    for race_num, race_df in merged.groupby('race'):
        valid = race_df.dropna(subset=['ensemble_prob', 'finish'])
        if len(valid) > 0:
            top = valid.loc[valid['ensemble_prob'].idxmax()]
            top_picks.append(int(top.get('won', 0)))

    if top_picks:
        out(f"  Top pick win rate:  {np.mean(top_picks):.1%} ({sum(top_picks)}/{len(top_picks)})")

    # Top pick place rate (1st or 2nd)
    top_place = []
    for race_num, race_df in merged.groupby('race'):
        valid = race_df.dropna(subset=['ensemble_prob', 'finish'])
        if len(valid) > 0:
            top = valid.loc[valid['ensemble_prob'].idxmax()]
            top_place.append(1 if top['finish'] <= 2 else 0)

    if top_place:
        out(f"  Top pick ITM rate:  {np.mean(top_place):.1%} ({sum(top_place)}/{len(top_place)})")

    # Log loss
    valid_ll = merged.dropna(subset=['ensemble_prob', 'won'])
    if len(valid_ll) > 0:
        clip = 1e-6
        probs = valid_ll['ensemble_prob'].clip(clip, 1 - clip)
        ll = -np.mean(valid_ll['won'] * np.log(probs) +
                       (1 - valid_ll['won']) * np.log(1 - probs))
        out(f"  Log loss:           {ll:.4f}")

    # ── Per-source AI performance ───────────────────────────────────
    # Check for the 4 source columns (from build_ensemble's _src columns)
    # Fall back to checking beyer/win_prob columns directly
    ai_sources = []
    for provider in ['claude', 'gemini']:
        for signal in ['beyer', 'prob']:
            src_col = f'{provider}_{signal}_src'
            if src_col in merged.columns and merged[src_col].notna().any():
                ai_sources.append((f'{provider}_{signal}', src_col))
            else:
                # Try raw columns for Spearman on beyers directly
                if signal == 'beyer':
                    raw_col = f'{provider}_beyer'
                    if raw_col in merged.columns and merged[raw_col].notna().any():
                        ai_sources.append((f'{provider}_beyer', raw_col))
                elif signal == 'prob':
                    raw_col = f'{provider}_win_prob'
                    if raw_col in merged.columns and merged[raw_col].notna().any():
                        ai_sources.append((f'{provider}_prob', raw_col))

    # Also add ML baseline
    if 'ml_win_prob_norm' in merged.columns and merged['ml_win_prob_norm'].notna().any():
        ai_sources.append(('ml', 'ml_win_prob_norm'))

    if ai_sources:
        out(f"\n{'─'*40}")
        out(f"PER-SOURCE PERFORMANCE")
        out(f"{'─'*40}")
        out(f"  {'Source':<16s} {'Spearman':>9s} {'TopPick':>8s} {'LogLoss':>8s}")
        out(f"  {'-'*16} {'-'*9} {'-'*8} {'-'*8}")

        for src_name, src_col in ai_sources:
            # Spearman
            src_spearman = []
            for race_num, race_df in merged.groupby('race'):
                valid = race_df.dropna(subset=[src_col, 'finish'])
                if len(valid) >= 3:
                    rho, _ = spearmanr(valid[src_col], -valid['finish'])
                    src_spearman.append(rho)

            # Top pick win rate
            src_top_wins = []
            for race_num, race_df in merged.groupby('race'):
                valid = race_df.dropna(subset=[src_col, 'finish'])
                if len(valid) > 0:
                    top = valid.loc[valid[src_col].idxmax()]
                    src_top_wins.append(int(top.get('won', 0)))

            # Log loss
            src_ll_valid = merged.dropna(subset=[src_col, 'won'])
            src_ll = np.nan
            if len(src_ll_valid) > 0:
                sp = src_ll_valid[src_col].clip(1e-6, 1 - 1e-6)
                # Normalize per race for fair comparison
                src_ll_probs = sp.copy()
                for race_num, race_df in src_ll_valid.groupby('race'):
                    idx = race_df.index
                    total = src_ll_probs.loc[idx].sum()
                    if total > 0:
                        src_ll_probs.loc[idx] /= total
                src_ll = -np.mean(
                    src_ll_valid['won'] * np.log(src_ll_probs.clip(1e-6)) +
                    (1 - src_ll_valid['won']) * np.log((1 - src_ll_probs).clip(1e-6)))

            rho_str = f"{np.mean(src_spearman):.4f}" if src_spearman else "  n/a"
            top_str = (f"{np.mean(src_top_wins):.0%} ({sum(src_top_wins)}/{len(src_top_wins)})"
                       if src_top_wins else "  n/a")
            ll_str = f"{src_ll:.4f}" if not np.isnan(src_ll) else "  n/a"

            out(f"  {src_name:<16s} {rho_str:>9s} {top_str:>8s} {ll_str:>8s}")

    # ── Race by race ────────────────────────────────────────────────
    out(f"\n{'─'*40}")
    out(f"RACE BY RACE")
    out(f"{'─'*40}")

    has_names = 'horse_name' in merged.columns

    for race_num in sorted(merged['race'].unique()):
        race = merged[merged['race'] == race_num].copy()
        race_valid = race.dropna(subset=['finish'])
        if race_valid.empty:
            continue

        winner = race_valid[race_valid['finish'] == 1]
        if winner.empty:
            continue

        w = winner.iloc[0]
        w_hn = int(w['horse_number'])
        w_prob = w.get('ensemble_prob', np.nan)
        w_name = str(w.get('horse_name', '')).strip() if has_names else ''
        w_paid = w.get('win_paid', np.nan)

        # Our top pick
        top = race.loc[race['ensemble_prob'].idxmax()]
        top_hn = int(top['horse_number'])
        top_finish = int(top['finish']) if pd.notna(top.get('finish')) else '?'

        hit = '✓' if top_hn == w_hn else ''
        name_str = f" {w_name[:20]}" if w_name else ""

        out(f"\n  R{race_num}: Winner #{w_hn}{name_str}"
            f" (prob={w_prob:.1%}, paid ${w_paid:.2f})" if pd.notna(w_paid)
            else f"\n  R{race_num}: Winner #{w_hn}{name_str} (prob={w_prob:.1%})")

        if top_hn == w_hn:
            out(f"       Our pick: #{top_hn} ✓ WINNER")
        else:
            top_name = str(top.get('horse_name', '')).strip() if has_names else ''
            top_name_str = f" {top_name[:20]}" if top_name else ""
            out(f"       Our pick: #{top_hn}{top_name_str} → finished {top_finish}")

    # ── Bet results ─────────────────────────────────────────────────
    if bet_path.exists():
        bet_df = pd.read_csv(bet_path)

        out(f"\n{'─'*40}")
        out(f"BET RESULTS")
        out(f"{'─'*40}")

        # Merge bets with results
        straight_bets = bet_df[bet_df['bet_type'].isin(['W', 'WP', 'WPS'])].copy()
        exotic_bets = bet_df[~bet_df['bet_type'].isin(['W', 'WP', 'WPS'])].copy()

        total_wagered = 0
        total_returned = 0

        if not straight_bets.empty:
            out(f"\n  Straight bets:")
            for _, bet in straight_bets.iterrows():
                race_num = int(bet['race'])
                hn = int(bet['horse_number'])
                bt = bet['bet_type']

                race_results = results[
                    (results['race'] == race_num) &
                    (results['horse_number'] == hn)
                ]
                if race_results.empty:
                    out(f"    R{race_num} {bt:<4s} #{hn}: no result")
                    continue

                row = race_results.iloc[0]
                finish = int(row['finish']) if pd.notna(row.get('finish')) else '?'
                name = str(row.get('horse_name', '')).strip() if 'horse_name' in row else ''
                name_str = f" {name[:18]}" if name else ""

                # Calculate return
                win_return = 0
                place_return = 0
                show_return = 0

                if finish == 1 and pd.notna(row.get('win_paid')):
                    win_return = row['win_paid'] / 2  # win_paid is per $2
                if finish <= 2 and pd.notna(row.get('place_paid')) and bt in ('WP', 'WPS'):
                    place_return = row['place_paid'] / 2
                if finish <= 3 and pd.notna(row.get('show_paid')) and bt == 'WPS':
                    show_return = row['show_paid'] / 2

                # Cost per bet type
                if bt == 'W':
                    cost = 2
                    ret = win_return * 2 if finish == 1 else 0
                elif bt == 'WP':
                    cost = 4
                    ret = 0
                    if finish == 1:
                        ret = win_return * 2 + place_return * 2
                    elif finish == 2:
                        ret = place_return * 2
                elif bt == 'WPS':
                    cost = 6
                    ret = 0
                    if finish == 1:
                        ret = win_return * 2 + place_return * 2 + show_return * 2
                    elif finish == 2:
                        ret = place_return * 2 + show_return * 2
                    elif finish == 3:
                        ret = show_return * 2

                total_wagered += cost
                total_returned += ret

                pnl = ret - cost
                result_str = f"fin={finish}"
                if ret > 0:
                    result_str += f" → ${ret:.2f}"
                    marker = " ✓" if pnl > 0 else ""
                else:
                    marker = ""

                out(f"    R{race_num} {bt:<4s} #{hn:<3d}{name_str:<20s} "
                    f"${cost} → {result_str}{marker}")

        if not exotic_bets.empty:
            out(f"\n  Exotic bets:")
            for _, bet in exotic_bets.iterrows():
                out(f"    R{int(bet['race'])} {bet['bet_type']:<10s} #{bet['horse_number']}")

        # ── Exacta box grading ───────────────────────────────────────────
        # Derive exacta boxes from the bet_log using the same trigger
        # rules predict_v3 uses for its exacta box recommendations:
        #   * race must have 2 or 3 horses bet with triggered_by == 'RDG'
        #   * no LOG fire anywhere in the race (LOG races already get W
        #     coverage, and the v3 backtest pool was RDG_only)
        # Cost: $4 for 2-horse box, $12 for 3-horse box.
        # Hit: actual top-2 finishers must both be in our boxed horses.
        # Payout: read from {DATA_PATH}/meta/{track}_{date}.csv.
        if 'triggered_by' in bet_df.columns:
            exacta_boxes = []
            for race_num, race_df in bet_df.groupby('race'):
                # Any LOG fire disqualifies the entire race per v3 rules
                has_log = race_df['triggered_by'].fillna('').str.contains(
                    'LOG').any()
                if has_log:
                    continue
                rdg_only = race_df[race_df['triggered_by'] == 'RDG']
                n_h = len(rdg_only)
                if n_h not in (2, 3):
                    continue
                horses = sorted(int(h) for h in rdg_only['horse_number'])
                exacta_boxes.append({
                    'race': int(race_num),
                    'horses': horses,
                    'n_horses': n_h,
                    'cost': 4 if n_h == 2 else 12,
                })

            if exacta_boxes:
                out(f"\n  Exacta boxes (derived from RDG-only multi-trigger races):")
                box_wagered = 0.0
                box_returned = 0.0
                box_hits = 0

                # Track miss reasons so we know what data is needed
                misses_no_meta = 0
                misses_no_race = 0
                misses_no_exacta = 0

                for box in exacta_boxes:
                    race_num = box['race']
                    horses = box['horses']
                    cost = box['cost']
                    box_str = '/'.join(str(h) for h in horses)

                    exacta_info = load_race_exacta(track, date_str, race_num)
                    status = exacta_info.get('_status') if isinstance(
                        exacta_info, dict) and '_status' in exacta_info else None

                    if status == 'no_meta_file':
                        out(f"    R{race_num} {box_str:<10s} ${cost:>2}: "
                            f"no meta CSV (download_results / backfill needed)")
                        misses_no_meta += 1
                        continue
                    if status == 'no_race_row':
                        out(f"    R{race_num} {box_str:<10s} ${cost:>2}: "
                            f"meta exists but race row missing")
                        misses_no_race += 1
                        continue
                    if status == 'no_exacta':
                        out(f"    R{race_num} {box_str:<10s} ${cost:>2}: "
                            f"no exacta data on this race")
                        misses_no_exacta += 1
                        continue

                    actual_top2 = set(exacta_info['exacta_finish'][:2])
                    payout = exacta_info['exacta_payout']
                    hit = actual_top2.issubset(set(horses))
                    fin_str = '-'.join(
                        str(h) for h in exacta_info['exacta_finish'])

                    box_wagered += cost
                    if hit:
                        box_returned += payout
                        box_hits += 1
                        marker = " ✓"
                        result_str = f"hit {fin_str} → ${payout:.2f}"
                    else:
                        marker = ""
                        result_str = f"missed (actual {fin_str})"

                    out(f"    R{race_num} {box_str:<10s} ${cost:>2}: "
                        f"{result_str}{marker}")

                # Roll into the day's totals so P&L includes boxes
                total_wagered += box_wagered
                total_returned += box_returned

                box_pnl = box_returned - box_wagered
                box_roi = (box_pnl / box_wagered * 100
                           if box_wagered > 0 else 0)
                evaluable = sum(1 for b in exacta_boxes) - (
                    misses_no_meta + misses_no_race + misses_no_exacta)
                out(f"    ── Exacta subtotal: {box_hits}/{evaluable} hits "
                    f"on ${box_wagered:.0f} wagered, "
                    f"${box_returned:.2f} returned "
                    f"(P&L ${box_pnl:+.2f}, ROI {box_roi:+.1f}%)")

        # Summary
        pnl = total_returned - total_wagered
        roi = pnl / total_wagered * 100 if total_wagered > 0 else 0

        out(f"\n  {'─'*35}")
        out(f"  Wagered:   ${total_wagered:.2f}")
        out(f"  Returned:  ${total_returned:.2f}")
        out(f"  P&L:       ${pnl:+.2f}")
        out(f"  ROI:       {roi:+.1f}%")

        if bankroll is not None:
            new_bankroll = bankroll + pnl
            out(f"\n  Bankroll:  ${bankroll:.2f} → ${new_bankroll:.2f}")

    # ── Expert picks grading ────────────────────────────────────────
    expert_dir = DATA_PATH / 'expert_picks' / date_str / track
    if expert_dir.exists():
        expert_csvs = sorted(expert_dir.glob('*.csv'))
        if expert_csvs:
            out(f"\n{'─'*40}")
            out(f"EXPERT PICKS GRADING")
            out(f"{'─'*40}")
            out(f"  Simulated $2 bets using our rules:")
            out(f"  W for ML≤9/2, WP for 5-9, WPS for 10+")
            out(f"  Best bets get $2 W/WP/WPS, others $2 W only")
            out(f"")
            out(f"  {'Expert':<22s} {'Bets':>4s} {'Won':>4s} {'Wag':>7s} "
                f"{'Ret':>7s} {'P&L':>7s} {'ROI':>7s}")
            out(f"  {'-'*22} {'-'*4} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

            for csv_path in expert_csvs:
                expert_name = csv_path.stem
                edf = pd.read_csv(csv_path)

                if edf.empty:
                    continue

                exp_wagered = 0
                exp_returned = 0
                exp_bets = 0
                exp_wins = 0

                for _, pick in edf.iterrows():
                    race_num = int(pick['race'])
                    hn = int(pick['horse_number'])
                    delta = pick['logit_delta']
                    sentiment = pick.get('sentiment', '')

                    # Only bet positive picks
                    if delta <= 0:
                        continue

                    # Find this horse in results
                    horse_result = results[
                        (results['race'] == race_num) &
                        (results['horse_number'] == hn)
                    ]
                    if horse_result.empty:
                        continue

                    row = horse_result.iloc[0]
                    finish = int(row['finish']) if pd.notna(row.get('finish')) else 99

                    # Determine bet type from ML odds
                    ml_odds_str = row.get('ml_odds', '')
                    ml_dec = np.nan
                    try:
                        s = str(ml_odds_str).strip()
                        if '/' in s:
                            num, den = map(float, s.split('/'))
                            ml_dec = num / den
                        elif s:
                            ml_dec = float(s)
                    except (ValueError, ZeroDivisionError):
                        pass

                    # Scale base bet by confidence:
                    #   +0.70 best_bet  → $2.00 base
                    #   +0.50 strong    → $1.40
                    #   +0.30 mentioned → $0.86
                    #   +0.15 contender → $0.43
                    base_bet = 2.0 * (delta / 0.7)

                    # Bet type by ML odds — only best bets / strong get WP/WPS
                    is_top_pick = sentiment in ('best_bet', 'strong') or delta >= 0.5

                    if is_top_pick:
                        if pd.notna(ml_dec) and ml_dec <= 4.5:
                            bt = 'W'
                            cost = base_bet
                        elif pd.notna(ml_dec) and ml_dec <= 9:
                            bt = 'WP'
                            cost = base_bet * 2  # W + P
                        else:
                            bt = 'WPS'
                            cost = base_bet * 3  # W + P + S
                    else:
                        bt = 'W'
                        cost = base_bet

                    # Calculate return — scale payouts by bet size
                    # win_paid/place_paid/show_paid are per $2 bet
                    bet_multiplier = base_bet / 2.0
                    ret = 0
                    if finish == 1 and pd.notna(row.get('win_paid')):
                        ret += row['win_paid'] * bet_multiplier
                    if finish <= 2 and pd.notna(row.get('place_paid')) and bt in ('WP', 'WPS'):
                        ret += row['place_paid'] * bet_multiplier
                    if finish <= 3 and pd.notna(row.get('show_paid')) and bt == 'WPS':
                        ret += row['show_paid'] * bet_multiplier

                    exp_wagered += cost
                    exp_returned += ret
                    exp_bets += 1
                    if finish == 1:
                        exp_wins += 1

                if exp_bets == 0:
                    continue

                exp_pnl = exp_returned - exp_wagered
                exp_roi = exp_pnl / exp_wagered * 100 if exp_wagered > 0 else 0

                out(f"  {expert_name:<22s} {exp_bets:>4d} {exp_wins:>4d} "
                    f"${exp_wagered:>5.0f} ${exp_returned:>5.0f} "
                    f"${exp_pnl:>+6.0f} {exp_roi:>+6.1f}%")

    # ── Save summary ────────────────────────────────────────────────
    out_dir = DATA_PATH / 'daily_summaries'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}_{track}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  Summary saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Daily prediction summary vs results')
    parser.add_argument('--date', type=str, required=True, help='Date (e.g. 2026-04-08)')
    parser.add_argument('--track', type=str, required=True, help='Track (e.g. keeneland)')
    parser.add_argument('--bankroll', type=float, default=None,
                        help='Starting bankroll for P&L tracking')
    args = parser.parse_args()

    run_summary(args.date, args.track, bankroll=args.bankroll)


if __name__ == '__main__':
    main()