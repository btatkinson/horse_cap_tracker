"""
historical_summary.py

Aggregate all daily prediction results into a historical summary with
running bankroll tracking and performance charts.

Usage:
    python historical_summary.py --bankroll 500
    python historical_summary.py --bankroll 500 --track keeneland
    python historical_summary.py --bankroll 500 --chart

Reads from:
    {DATA_PATH}/predictions/{date}/{track}/predictions.csv
    {DATA_PATH}/predictions/{date}/{track}/bet_log.csv
    {DATA_PATH}/entries/*.csv   (results)

Writes to:
    {DATA_PATH}/historical_summary.txt
    {DATA_PATH}/historical_bankroll.png  (if --chart)
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
    'gulfstream-park': 'gulfstream', 'gulfstream park': 'gulfstream',
    'churchill-downs': 'churchill', 'churchill downs': 'churchill',
    'santa-anita': 'santa-anita', 'santa anita': 'santa-anita',
    'del-mar': 'del-mar', 'del mar': 'del-mar',
    'fair-grounds': 'fair-grounds', 'fair grounds': 'fair-grounds',
    'tampa-bay': 'tampa', 'tampa bay': 'tampa',
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


def load_all_results():
    """Load all entries results."""
    entries_dir = DATA_PATH / 'entries'
    if not entries_dir.exists():
        return None
    dfs = []
    for f in entries_dir.glob('*.csv'):
        try:
            df = pd.read_csv(f)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            if 'track' in df.columns:
                df['track'] = df['track'].apply(_normalize_track)
            dfs.append(df)
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
    return result


def find_all_prediction_days():
    """Find all date/track combos that have predictions."""
    pred_dir = DATA_PATH / 'predictions'
    if not pred_dir.exists():
        return []
    days = []
    for date_dir in sorted(pred_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for track_dir in sorted(date_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            pred_file = track_dir / 'predictions.csv'
            bet_file = track_dir / 'bet_log.csv'
            if pred_file.exists():
                days.append({
                    'date': date_dir.name,
                    'track': track_dir.name,
                    'pred_path': pred_file,
                    'bet_path': bet_file if bet_file.exists() else None,
                })
    return days


def grade_day(day, results):
    """Grade one day's predictions. Returns dict of metrics."""
    preds = pd.read_csv(day['pred_path'])
    date_str = day['date']
    track = day['track']

    # Filter results to this date/track
    if results is None:
        return None

    mask = results['date'] == date_str
    if 'track' in results.columns:
        mask = mask & (results['track'] == track)
    day_results = results[mask]
    if day_results.empty:
        return None

    # Merge
    result_cols = ['race', 'horse_number', 'finish', 'won',
                   'win_paid', 'place_paid', 'show_paid']
    result_cols = [c for c in result_cols if c in day_results.columns]
    merged = preds.merge(day_results[result_cols].drop_duplicates(),
                         on=['race', 'horse_number'], how='left')

    n_races = merged['race'].nunique()
    n_with_finish = merged['finish'].notna().sum()
    if n_with_finish == 0:
        return None

    # Spearman
    spearman_vals = []
    for race_num, race_df in merged.groupby('race'):
        valid = race_df.dropna(subset=['ensemble_prob', 'finish'])
        if len(valid) >= 3:
            rho, _ = spearmanr(valid['ensemble_prob'], -valid['finish'])
            spearman_vals.append(rho)

    # Top pick win rate
    top_picks = []
    for race_num, race_df in merged.groupby('race'):
        valid = race_df.dropna(subset=['ensemble_prob', 'finish'])
        if len(valid) > 0:
            top = valid.loc[valid['ensemble_prob'].idxmax()]
            top_picks.append(int(top.get('won', 0)))

    # Bet P&L
    wagered = 0
    returned = 0
    n_bets = 0
    n_wins = 0
    bet_details = []  # individual bet records for EV binning

    if day['bet_path']:
        bet_df = pd.read_csv(day['bet_path'])
        straight = bet_df[bet_df['bet_type'].isin(['W', 'WP', 'WPS'])]

        for _, bet in straight.iterrows():
            race_num = int(bet['race'])
            hn = int(bet['horse_number'])
            bt = bet['bet_type']

            horse_result = day_results[
                (day_results['race'] == race_num) &
                (day_results['horse_number'] == hn)
            ]
            if horse_result.empty:
                continue

            row = horse_result.iloc[0]
            finish = int(row['finish']) if pd.notna(row.get('finish')) else 99

            if bt == 'W':
                cost = 2
                ret = row.get('win_paid', 0) if finish == 1 and pd.notna(row.get('win_paid')) else 0
            elif bt == 'WP':
                cost = 4
                ret = 0
                if finish == 1 and pd.notna(row.get('win_paid')):
                    ret += row['win_paid']
                if finish <= 2 and pd.notna(row.get('place_paid')):
                    ret += row['place_paid']
            elif bt == 'WPS':
                cost = 6
                ret = 0
                if finish == 1 and pd.notna(row.get('win_paid')):
                    ret += row['win_paid']
                if finish <= 2 and pd.notna(row.get('place_paid')):
                    ret += row['place_paid']
                if finish <= 3 and pd.notna(row.get('show_paid')):
                    ret += row['show_paid']

            wagered += cost
            returned += ret
            n_bets += 1
            if finish == 1:
                n_wins += 1

            bet_details.append({
                'date': date_str,
                'track': track,
                'race': race_num,
                'horse_number': hn,
                'bet_type': bt,
                'ev_roi': bet.get('ev_roi', np.nan),
                'ev_per_2': bet.get('ev_per_2', np.nan),
                'ensemble_prob': bet.get('ensemble_prob', np.nan),
                'ml_prob': bet.get('ml_prob', np.nan),
                'cost': cost,
                'returned': ret,
                'pnl': ret - cost,
                'finish': finish,
                'won': 1 if finish == 1 else 0,
            })

    return {
        'date': date_str,
        'track': track,
        'n_races': n_races,
        'spearman': np.mean(spearman_vals) if spearman_vals else np.nan,
        'top_pick_wins': sum(top_picks),
        'top_pick_total': len(top_picks),
        'top_pick_rate': np.mean(top_picks) if top_picks else np.nan,
        'n_bets': n_bets,
        'n_wins': n_wins,
        'wagered': wagered,
        'returned': returned,
        'pnl': returned - wagered,
        'bet_details': bet_details,
    }


def run_summary(starting_bankroll=500, track_filter=None, do_chart=False):
    """Run historical summary across all prediction days."""
    days = find_all_prediction_days()
    if not days:
        print("No predictions found.")
        return

    if track_filter:
        days = [d for d in days if d['track'] == track_filter.lower()]
        if not days:
            print(f"No predictions found for track '{track_filter}'")
            return

    results = load_all_results()
    if results is None:
        print("No results found in entries/")
        return

    lines = []
    def out(s=''):
        lines.append(s)
        print(s)

    out(f"{'='*72}")
    out(f"HISTORICAL SUMMARY")
    out(f"{'='*72}")
    out(f"  Starting bankroll: ${starting_bankroll:.2f}")
    if track_filter:
        out(f"  Track filter: {track_filter}")
    out(f"  Prediction days found: {len(days)}")

    # Grade each day
    day_results = []
    for day in days:
        grade = grade_day(day, results)
        if grade:
            day_results.append(grade)

    if not day_results:
        out("\n  No graded days (results may not be available yet)")
        return

    df = pd.DataFrame(day_results)

    # ── Day-by-day table ────────────────────────────────────────────
    out(f"\n{'─'*72}")
    out(f"DAY BY DAY")
    out(f"{'─'*72}")

    bankroll = starting_bankroll
    bankroll_history = [{'date': 'start', 'bankroll': bankroll, 'label': 'Start'}]

    out(f"  {'Date':<12s} {'Track':<12s} {'Races':>5s} {'ρ':>7s} {'Top%':>6s} "
        f"{'Bets':>5s} {'Won':>4s} {'Wag':>7s} {'Ret':>7s} {'P&L':>7s} {'Bank':>8s}")
    out(f"  {'-'*12} {'-'*12} {'-'*5} {'-'*7} {'-'*6} "
        f"{'-'*5} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    for _, row in df.iterrows():
        bankroll += row['pnl']
        rho_s = f"{row['spearman']:.4f}" if not np.isnan(row['spearman']) else "n/a"
        top_s = f"{row['top_pick_rate']:.0%}" if not np.isnan(row['top_pick_rate']) else "n/a"

        out(f"  {row['date']:<12s} {row['track']:<12s} {row['n_races']:>5d} {rho_s:>7s} "
            f"{top_s:>6s} {row['n_bets']:>5d} {row['n_wins']:>4d} "
            f"${row['wagered']:>5.0f} ${row['returned']:>5.0f} "
            f"${row['pnl']:>+6.0f} ${bankroll:>7.0f}")

        bankroll_history.append({
            'date': row['date'],
            'bankroll': bankroll,
            'label': f"{row['date']} {row['track'][:3].upper()}"
        })

    # ── Cumulative stats ────────────────────────────────────────────
    out(f"\n{'─'*72}")
    out(f"CUMULATIVE")
    out(f"{'─'*72}")

    total_races = df['n_races'].sum()
    total_bets = df['n_bets'].sum()
    total_wins = df['n_wins'].sum()
    total_wagered = df['wagered'].sum()
    total_returned = df['returned'].sum()
    total_pnl = total_returned - total_wagered
    total_roi = total_pnl / total_wagered * 100 if total_wagered > 0 else 0

    mean_rho = df['spearman'].mean()
    median_rho = df['spearman'].median()
    total_top_wins = df['top_pick_wins'].sum()
    total_top_total = df['top_pick_total'].sum()
    top_rate = total_top_wins / total_top_total if total_top_total > 0 else 0

    out(f"  Days:               {len(df)}")
    out(f"  Races:              {total_races}")
    out(f"  Mean Spearman ρ:    {mean_rho:.4f}")
    out(f"  Median Spearman ρ:  {median_rho:.4f}")
    out(f"  Top pick win rate:  {top_rate:.1%} ({total_top_wins}/{total_top_total})")
    out(f"")
    out(f"  Total bets:         {total_bets}")
    out(f"  Win rate:           {total_wins/total_bets:.1%} ({total_wins}/{total_bets})" if total_bets > 0 else "")
    out(f"  Total wagered:      ${total_wagered:.2f}")
    out(f"  Total returned:     ${total_returned:.2f}")
    out(f"  P&L:                ${total_pnl:+.2f}")
    out(f"  ROI:                {total_roi:+.1f}%")
    out(f"")
    out(f"  Starting bankroll:  ${starting_bankroll:.2f}")
    out(f"  Current bankroll:   ${bankroll:.2f}")
    out(f"  Bankroll change:    ${bankroll - starting_bankroll:+.2f} "
        f"({(bankroll/starting_bankroll - 1)*100:+.1f}%)")

    # ── Per-track breakdown ─────────────────────────────────────────
    if df['track'].nunique() > 1:
        out(f"\n{'─'*72}")
        out(f"BY TRACK")
        out(f"{'─'*72}")
        out(f"  {'Track':<14s} {'Days':>5s} {'Races':>6s} {'ρ':>7s} {'Top%':>6s} "
            f"{'Wag':>7s} {'P&L':>7s} {'ROI':>7s}")
        out(f"  {'-'*14} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

        for track, tdf in df.groupby('track'):
            t_rho = tdf['spearman'].mean()
            t_top = tdf['top_pick_wins'].sum() / tdf['top_pick_total'].sum() if tdf['top_pick_total'].sum() > 0 else 0
            t_wag = tdf['wagered'].sum()
            t_pnl = tdf['pnl'].sum()
            t_roi = t_pnl / t_wag * 100 if t_wag > 0 else 0
            out(f"  {track:<14s} {len(tdf):>5d} {tdf['n_races'].sum():>6d} "
                f"{t_rho:>7.4f} {t_top:>5.0%} ${t_wag:>5.0f} ${t_pnl:>+6.0f} {t_roi:>+6.1f}%")

    # ── Results by predicted EV ─────────────────────────────────────
    all_bets = []
    for dr in day_results:
        all_bets.extend(dr.get('bet_details', []))

    if all_bets:
        bets_df = pd.DataFrame(all_bets)

        if 'ev_roi' in bets_df.columns and bets_df['ev_roi'].notna().any():
            out(f"\n{'─'*72}")
            out(f"RESULTS BY PREDICTED EV")
            out(f"{'─'*72}")

            # Bin by EV ROI
            bins = [0, 0.05, 0.10, 0.20, 0.50, float('inf')]
            labels = ['0-5%', '5-10%', '10-20%', '20-50%', '>50%']
            bets_df['ev_bin'] = pd.cut(bets_df['ev_roi'], bins=bins, labels=labels, right=False)

            out(f"  {'EV Bin':<10s} {'Bets':>5s} {'Won':>4s} {'Win%':>6s} "
                f"{'Wag':>7s} {'Ret':>7s} {'P&L':>7s} {'ROI':>7s} {'Avg EV':>7s}")
            out(f"  {'-'*10} {'-'*5} {'-'*4} {'-'*6} "
                f"{'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

            for label in labels:
                bin_df = bets_df[bets_df['ev_bin'] == label]
                if bin_df.empty:
                    continue
                b_n = len(bin_df)
                b_wins = bin_df['won'].sum()
                b_winpct = b_wins / b_n if b_n > 0 else 0
                b_wag = bin_df['cost'].sum()
                b_ret = bin_df['returned'].sum()
                b_pnl = b_ret - b_wag
                b_roi = b_pnl / b_wag * 100 if b_wag > 0 else 0
                b_avg_ev = bin_df['ev_roi'].mean()

                out(f"  {label:<10s} {b_n:>5d} {b_wins:>4d} {b_winpct:>5.0%} "
                    f"${b_wag:>5.0f} ${b_ret:>5.0f} ${b_pnl:>+6.0f} {b_roi:>+6.1f}% "
                    f"{b_avg_ev:>+6.1%}")

            # Also show by bet type
            out(f"\n  By bet type:")
            out(f"  {'Type':<6s} {'Bets':>5s} {'Won':>4s} {'Win%':>6s} "
                f"{'Wag':>7s} {'Ret':>7s} {'P&L':>7s} {'ROI':>7s}")
            out(f"  {'-'*6} {'-'*5} {'-'*4} {'-'*6} "
                f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}")

            for bt in ['W', 'WP', 'WPS']:
                bt_df = bets_df[bets_df['bet_type'] == bt]
                if bt_df.empty:
                    continue
                b_n = len(bt_df)
                b_wins = bt_df['won'].sum()
                b_winpct = b_wins / b_n
                b_wag = bt_df['cost'].sum()
                b_ret = bt_df['returned'].sum()
                b_pnl = b_ret - b_wag
                b_roi = b_pnl / b_wag * 100 if b_wag > 0 else 0

                out(f"  {bt:<6s} {b_n:>5d} {b_wins:>4d} {b_winpct:>5.0%} "
                    f"${b_wag:>5.0f} ${b_ret:>5.0f} ${b_pnl:>+6.0f} {b_roi:>+6.1f}%")

    # ── Save text ───────────────────────────────────────────────────
    out_path = DATA_PATH / 'historical_summary.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  Summary saved to {out_path}")

    # ── Chart ───────────────────────────────────────────────────────
    if do_chart and len(bankroll_history) > 1:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                            gridspec_kw={'height_ratios': [2, 1]})
            fig.suptitle('Horse Racing Handicapping Pipeline — Performance',
                         fontsize=14, fontweight='bold')

            # Top chart: bankroll over time
            dates = []
            banks = []
            for h in bankroll_history:
                if h['date'] == 'start':
                    continue
                dates.append(pd.to_datetime(h['date']))
                banks.append(h['bankroll'])

            ax1.plot(dates, banks, 'b-o', markersize=6, linewidth=2, label='Bankroll')
            ax1.axhline(y=starting_bankroll, color='gray', linestyle='--',
                       alpha=0.7, label=f'Starting (${starting_bankroll})')
            ax1.fill_between(dates, starting_bankroll, banks,
                            where=[b >= starting_bankroll for b in banks],
                            alpha=0.15, color='green')
            ax1.fill_between(dates, starting_bankroll, banks,
                            where=[b < starting_bankroll for b in banks],
                            alpha=0.15, color='red')
            ax1.set_ylabel('Bankroll ($)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

            # Bottom chart: daily P&L bars
            bar_dates = [pd.to_datetime(r['date']) for _, r in df.iterrows()]
            bar_pnl = df['pnl'].values
            colors = ['green' if p >= 0 else 'red' for p in bar_pnl]
            bar_labels = [f"{r['track'][:3].upper()}" for _, r in df.iterrows()]

            ax2.bar(bar_dates, bar_pnl, color=colors, alpha=0.7, width=0.6)
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.set_ylabel('Daily P&L ($)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

            # Add track labels to bars
            for d, p, lbl in zip(bar_dates, bar_pnl, bar_labels):
                ax2.annotate(lbl, (d, p), textcoords="offset points",
                           xytext=(0, 5 if p >= 0 else -12),
                           ha='center', fontsize=7, alpha=0.7)

            plt.tight_layout()
            chart_path = DATA_PATH / 'historical_bankroll.png'
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Chart saved to {chart_path}")

        except ImportError:
            print("  matplotlib not installed — skipping chart")


def main():
    parser = argparse.ArgumentParser(description='Historical performance summary')
    parser.add_argument('--bankroll', type=float, default=500,
                        help='Starting bankroll (default $500)')
    parser.add_argument('--track', type=str, default=None,
                        help='Filter to one track')
    parser.add_argument('--chart', action='store_true',
                        help='Generate bankroll chart (requires matplotlib)')
    args = parser.parse_args()

    run_summary(starting_bankroll=args.bankroll, track_filter=args.track,
                do_chart=args.chart)


if __name__ == '__main__':
    main()