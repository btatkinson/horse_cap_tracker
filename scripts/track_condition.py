"""
track_condition.py

Fetch and persist track conditions + weather metadata.

Usage:
    python track_condition.py --track keeneland --date 2026-04-08
    python track_condition.py --track gulfstream                   # defaults to today

Saves to: {DATA_PATH}/historical_meta.pkl
Structure:
    {
        '2026-04-05': {
            'gulfstream': {
                'dirt': 'Fast',
                'turf': 'Firm',
                'source': 'equibase',
                'temperature': '82°F',
                'wind': '10 mph',
                'precip_pct': 5,
                'forecast': 'Mostly sunny',
                'prompt': '...',
                'fetched_at': '2026-04-05T10:30:00',
            }
        }
    }

From other scripts:
    from track_condition import load_meta, get_track_condition
    meta = load_meta()
    info = meta.get('2026-04-05', {}).get('gulfstream')
"""

import argparse
import os
import pickle
import re
from datetime import date, datetime, time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = os.environ['DATA_PATH']
META_PATH = os.path.join(DATA_PATH, 'historical_meta.pkl')

# Race window (local time) — averaged for weather estimates.
# Most thoroughbred cards run roughly noon to 5 PM local.
RACE_WINDOW_START = time(12, 0)  # 12 PM
RACE_WINDOW_END   = time(17, 0)  # 5 PM

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("pip install requests beautifulsoup4")


# Track metadata: equibase code, lat/lon for NWS
TRACKS = {
    'keeneland':       {'code': 'KEE', 'lat': 38.0464, 'lon': -84.5908, 'state': 'KY'},
    'churchill-downs':  {'code': 'CD',  'lat': 38.2025, 'lon': -85.7705, 'state': 'KY'},
    'churchill':        {'code': 'CD',  'lat': 38.2025, 'lon': -85.7705, 'state': 'KY'},
    'saratoga':         {'code': 'SAR', 'lat': 43.0667, 'lon': -73.7750, 'state': 'NY'},
    'belmont':          {'code': 'BEL', 'lat': 40.7167, 'lon': -73.7167, 'state': 'NY'},
    'aqueduct':         {'code': 'AQU', 'lat': 40.6723, 'lon': -73.8353, 'state': 'NY'},
    'gulfstream':       {'code': 'GP',  'lat': 25.9268, 'lon': -80.1422, 'state': 'FL'},
    'gulfstream-park':  {'code': 'GP',  'lat': 25.9268, 'lon': -80.1422, 'state': 'FL'},
    'santa-anita':      {'code': 'SA',  'lat': 34.1397, 'lon': -118.0432, 'state': 'CA'},
    'del-mar':          {'code': 'DMR', 'lat': 32.9339, 'lon': -117.2572, 'state': 'CA'},
    'oaklawn':          {'code': 'OP',  'lat': 34.4910, 'lon': -93.0630, 'state': 'AR'},
    'pimlico':          {'code': 'PIM', 'lat': 39.3491, 'lon': -76.6778, 'state': 'MD'},
    'laurel':           {'code': 'LRL', 'lat': 39.0904, 'lon': -76.8384, 'state': 'MD'},
    'tampa':            {'code': 'TAM', 'lat': 28.0122, 'lon': -82.4250, 'state': 'FL'},
    'tampa-bay':        {'code': 'TAM', 'lat': 28.0122, 'lon': -82.4250, 'state': 'FL'},
    'fair-grounds':     {'code': 'FG',  'lat': 29.9856, 'lon': -90.0628, 'state': 'LA'},
    'turfway':          {'code': 'TP',  'lat': 39.0553, 'lon': -84.6611, 'state': 'KY'},
    'woodbine':         {'code': 'WO',  'lat': 43.7167, 'lon': -79.6000, 'state': 'ON'},
    'ellis':            {'code': 'ELP', 'lat': 37.7963, 'lon': -87.6006, 'state': 'KY'},
    'monmouth':         {'code': 'MTH', 'lat': 40.2251, 'lon': -74.0121, 'state': 'NJ'},
    'parx':             {'code': 'PRX', 'lat': 40.1770, 'lon': -75.0026, 'state': 'PA'},
}


# ─── Persistence ───────────────────────────────────────────────────────────────

def load_meta() -> dict:
    """Load the historical metadata dict from disk."""
    if os.path.exists(META_PATH):
        with open(META_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_meta(meta: dict):
    """Save the historical metadata dict to disk."""
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, 'wb') as f:
        pickle.dump(meta, f)


# ─── Equibase ──────────────────────────────────────────────────────────────────

def get_equibase_conditions(track: str) -> dict | None:
    """Fetch current track conditions from Equibase late-changes page (race day only)."""
    meta = TRACKS.get(track.lower())
    if not meta:
        return None

    code = meta['code']
    url = f"https://www.equibase.com/static/latechanges/html/latechanges{code}-USA.html"

    try:
        resp = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, 'html.parser')
        text = soup.get_text()

        result = {'source': 'equibase', 'dirt': None, 'turf': None}

        dirt_match = re.search(r'Dirt:\s*(\w+(?:\s+\w+)?)', text)
        turf_match = re.search(r'Turf:\s*(\w+(?:\s+\w+)?)', text)
        updated_match = re.search(r'Last Updated:\s*(.+?)(?:\n|$)', text)

        if dirt_match:
            result['dirt'] = dirt_match.group(1).strip()
        if turf_match:
            result['turf'] = turf_match.group(1).strip()
        if updated_match:
            result['updated'] = updated_match.group(1).strip()

        if not result['dirt'] and not result['turf']:
            return None

        return result

    except Exception as e:
        print(f"  Equibase fetch failed: {e}")
        return None


# ─── NWS Weather ───────────────────────────────────────────────────────────────

def _nws_points(lat: float, lon: float, headers: dict) -> dict | None:
    """Hit the /points endpoint to get forecast URLs for a lat/lon."""
    resp = requests.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=10, headers=headers)
    if resp.status_code != 200:
        print(f"  NWS points lookup failed: {resp.status_code}")
        return None
    return resp.json().get('properties', {})


def _avg_in_window(periods: list, target_date: date,
                   start: time = RACE_WINDOW_START,
                   end: time = RACE_WINDOW_END) -> dict | None:
    """
    Average hourly periods that fall within [start, end] local time on target_date.
    NWS hourly periods are 1 hour long; startTime includes local TZ offset.
    Returns aggregated dict or None if no hours matched.
    """
    matched = []
    for p in periods:
        try:
            st = datetime.fromisoformat(p['startTime'])
        except (KeyError, ValueError):
            continue
        # Use the local time from the ISO string (NWS provides TZ-aware timestamps)
        local_date = st.date()
        local_time = st.time()
        if local_date != target_date:
            continue
        if start <= local_time < end:
            matched.append(p)

    if not matched:
        return None

    def _num(val, default=0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    precips = [
        _num((p.get('probabilityOfPrecipitation') or {}).get('value'))
        for p in matched
    ]
    temps = [_num(p.get('temperature')) for p in matched if p.get('temperature') is not None]

    # Wind speed comes as string like "10 mph" or "5 to 10 mph" — pull max number.
    def _max_wind(ws: str) -> float:
        if not ws:
            return 0.0
        nums = re.findall(r'\d+', ws)
        return max((float(n) for n in nums), default=0.0)

    winds = [_max_wind(p.get('windSpeed', '')) for p in matched]

    # Pick the midpoint period as the representative short forecast
    mid = matched[len(matched) // 2]

    return {
        'precip_pct': int(round(max(precips) if precips else 0)),       # worst-case in window
        'precip_pct_avg': int(round(sum(precips) / len(precips))) if precips else 0,
        'temperature': f"{int(round(sum(temps) / len(temps)))}°{mid.get('temperatureUnit', 'F')}" if temps else 'Unknown',
        'wind': f"{int(round(max(winds)))} mph" if winds else 'Unknown',
        'forecast': mid.get('shortForecast', ''),
        'hours_matched': len(matched),
        'window': f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')} local",
    }


def _daily_fallback(periods: list, target_date: date) -> dict | None:
    """Original daily-period logic, used when hourly is unavailable."""
    best_period = None
    for period in periods:
        try:
            period_start = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
        except (KeyError, ValueError):
            continue
        if period_start.date() == target_date:
            if period['isDaytime'] or best_period is None:
                best_period = period
                if period['isDaytime']:
                    break

    if not best_period:
        for period in periods:
            try:
                period_start = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
            except (KeyError, ValueError):
                continue
            if period_start.date() >= target_date:
                best_period = period
                break

    if not best_period:
        return None

    precip_pct = (best_period.get('probabilityOfPrecipitation') or {}).get('value') or 0
    return {
        'precip_pct': int(precip_pct),
        'temperature': f"{best_period.get('temperature', '?')}°{best_period.get('temperatureUnit', 'F')}",
        'wind': best_period.get('windSpeed', 'Unknown'),
        'forecast': best_period.get('shortForecast', ''),
        'detailed': best_period.get('detailedForecast', ''),
        'period_name': best_period.get('name', ''),
    }


def get_nws_forecast(track: str, target_date: date = None) -> dict | None:
    """
    Get weather forecast from NWS API for a track location, averaged over the
    race-time window (default 12 PM–5 PM local). Falls back to daily period
    forecast if the hourly endpoint is unavailable.
    """
    meta = TRACKS.get(track.lower())
    if not meta:
        print(f"  Unknown track: {track}")
        return None

    if meta['state'] == 'ON':
        print(f"  NWS does not cover {track} (Canada). Skipping forecast.")
        return None

    if target_date is None:
        target_date = date.today()

    headers = {'User-Agent': 'horse-handicapper-tool (personal use)'}

    try:
        props = _nws_points(meta['lat'], meta['lon'], headers)
        if not props:
            return None

        hourly_url = props.get('forecastHourly')
        daily_url  = props.get('forecast')

        used_fallback = False
        agg = None

        # Try hourly + race-window average first
        if hourly_url:
            try:
                resp = requests.get(hourly_url, timeout=10, headers=headers)
                if resp.status_code == 200:
                    periods = resp.json().get('properties', {}).get('periods', [])
                    agg = _avg_in_window(periods, target_date)
                    if agg is None:
                        print(f"  NWS hourly: no hours matched {RACE_WINDOW_START.strftime('%H:%M')}–"
                              f"{RACE_WINDOW_END.strftime('%H:%M')} on {target_date}")
                else:
                    print(f"  NWS hourly fetch failed: {resp.status_code}")
            except Exception as e:
                print(f"  NWS hourly error: {e}")

        # Fallback to daily period
        if agg is None:
            used_fallback = True
            print(f"  ⚠ Falling back to daily-period forecast (less accurate for race-time estimates)")
            if not daily_url:
                print(f"  No daily forecast URL available")
                return None
            resp = requests.get(daily_url, timeout=10, headers=headers)
            if resp.status_code != 200:
                print(f"  NWS daily fetch failed: {resp.status_code}")
                return None
            periods = resp.json().get('properties', {}).get('periods', [])
            agg = _daily_fallback(periods, target_date)
            if agg is None:
                print(f"  No forecast available for {target_date}")
                return None

        precip_pct = agg['precip_pct']
        return {
            'source': 'nws_forecast',
            'dirt': estimate_dirt_condition(precip_pct),
            'turf': estimate_turf_condition(precip_pct),
            'precip_pct': precip_pct,
            'precip_pct_avg': agg.get('precip_pct_avg'),
            'temperature': agg['temperature'],
            'wind': agg['wind'],
            'forecast': agg['forecast'],
            'detailed': agg.get('detailed', ''),
            'period_name': agg.get('period_name', ''),
            'hours_matched': agg.get('hours_matched'),
            'window': agg.get('window'),
            'daily_fallback': used_fallback,
            'target_date': str(target_date),
        }

    except Exception as e:
        print(f"  NWS forecast failed: {e}")
        return None


# Probability-first condition estimators. Text matching removed — the NWS
# short-forecast string kept leaking overnight/far-off-window terms like
# "thunderstorm" into what should have been a fair-track call.

def estimate_dirt_condition(precip_pct: int) -> str:
    if precip_pct >= 80:
        return 'Sloppy (estimated)'
    elif precip_pct >= 60:
        return 'Muddy (estimated)'
    elif precip_pct >= 40:
        return 'Good (estimated)'
    else:
        return 'Fast (estimated)'


def estimate_turf_condition(precip_pct: int) -> str:
    if precip_pct >= 80:
        return 'Yielding (estimated) — turf races may come off'
    elif precip_pct >= 60:
        return 'Soft (estimated)'
    elif precip_pct >= 40:
        return 'Good (estimated)'
    else:
        return 'Firm (estimated)'


# ─── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt_text(info: dict, track: str) -> str:
    """Build a text block to inject into the handicapping prompt."""
    lines = [f"TRACK CONDITIONS for {track.title()}:"]

    if info.get('dirt'):
        lines.append(f"  Dirt: {info['dirt']}")
    if info.get('turf'):
        lines.append(f"  Turf: {info['turf']}")

    if info['source'] == 'nws_forecast':
        lines.append(f"  Weather: {info.get('forecast', 'Unknown')}")
        precip_line = f"  Precip chance: {info.get('precip_pct', '?')}%"
        if info.get('precip_pct_avg') is not None and info.get('precip_pct_avg') != info.get('precip_pct'):
            precip_line += f" (avg {info['precip_pct_avg']}%)"
        lines.append(precip_line)
        lines.append(f"  Temp: {info.get('temperature', '?')}, Wind: {info.get('wind', '?')}")
        if info.get('window') and not info.get('daily_fallback'):
            lines.append(f"  Window: {info['window']} ({info.get('hours_matched', '?')} hours)")
        if info.get('daily_fallback'):
            lines.append(f"  (⚠ daily-period fallback — hourly unavailable)")

        precip = info.get('precip_pct', 0)
        if precip >= 60:
            lines.append("")
            lines.append("HANDICAPPING NOTE: Significant rain expected. Adjust Beyer estimates:")
            lines.append("  - Dirt figures typically compress 5-10 points on off tracks")
            lines.append("  - Horses with proven wet-track form get a significant edge")
            lines.append("  - Front-runners on sealed sloppy tracks often outperform")
            lines.append("  - Turf races may be taken off the turf to dirt/all-weather")
        elif precip >= 40:
            lines.append("")
            lines.append("HANDICAPPING NOTE: Rain possible. Consider wet-track form as a tiebreaker.")
    elif info['source'] == 'equibase':
        if info.get('updated'):
            lines.append(f"  (Updated: {info['updated']})")

        dirt = (info.get('dirt') or '').lower()
        if dirt in ('sloppy', 'muddy', 'good', 'slow', 'heavy'):
            lines.append("")
            lines.append(f"HANDICAPPING NOTE: Off track ({dirt}). Adjust Beyer estimates:")
            lines.append("  - Dirt figures typically compress 5-10 points on off tracks")
            lines.append("  - Horses with proven wet-track form get a significant edge")
            lines.append("  - Front-runners on sealed sloppy tracks often outperform")

    return '\n'.join(lines)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def get_track_condition(track: str, target_date: date = None) -> dict:
    """
    Fetch track conditions, save to historical_meta.pkl, return info dict.
    Tries Equibase first (race day), falls back to NWS forecast.
    """
    track = track.lower()

    if target_date is None:
        target_date = date.today()

    is_today = (target_date == date.today())

    info = None

    # Try Equibase first (only makes sense for today)
    if is_today:
        print(f"Checking Equibase for {track}...")
        info = get_equibase_conditions(track)
        if info:
            print(f"  Equibase: Dirt={info.get('dirt')}, Turf={info.get('turf')}")
    else:
        print(f"  Equibase doesn't have future dates ({target_date})")

    # Fall back to NWS
    if not info:
        print(f"Fetching NWS forecast for {track} on {target_date}...")
        info = get_nws_forecast(track, target_date)
        if info:
            print(f"  NWS: Dirt={info.get('dirt')}, Turf={info.get('turf')}")
            print(f"  Forecast: {info.get('forecast')}, Precip: {info.get('precip_pct')}%")

    if not info:
        info = {
            'source': 'unavailable',
            'dirt': 'Unknown',
            'turf': 'Unknown',
        }
        print(f"  Could not get conditions for {track}")

    # Build prompt text
    info['prompt'] = build_prompt_text(info, track)
    info['fetched_at'] = datetime.now().isoformat()

    # Format date string (YYYY-MM-DD)
    date_str = target_date.strftime('%Y-%m-%d')

    # Save to historical_meta.pkl
    meta = load_meta()
    meta.setdefault(date_str, {})[track] = {
        k: v for k, v in info.items()
        if k != 'detailed'  # skip verbose NWS text
    }
    save_meta(meta)
    print(f"  Saved to {META_PATH}")

    return info


def parse_date(date_str: str) -> date:
    """Parse date from various formats. Prefers YYYY-MM-DD."""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', date_str)
    if m:
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    m = re.match(r'(\d{1,2})-(\d{1,2})-(\d{2})', date_str)
    if m:
        return date(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
    raise ValueError(f"Can't parse date: {date_str}")


def main():
    parser = argparse.ArgumentParser(description='Fetch and save track conditions')
    parser.add_argument('--track', type=str, required=True, help='Track name (e.g. keeneland)')
    parser.add_argument('--date', type=str, help='Target date (e.g. 2026-04-08). Defaults to today.')
    args = parser.parse_args()

    target = parse_date(args.date) if args.date else None
    info = get_track_condition(args.track, target)

    print(f"\n{'='*50}")
    print(info['prompt'])
    print(f"{'='*50}")

    # Show what's in the meta file
    meta = load_meta()
    print(f"\nhistorical_meta.pkl has {len(meta)} dates:")
    for d in sorted(meta.keys()):
        tracks = list(meta[d].keys())
        print(f"  {d}: {', '.join(tracks)}")


if __name__ == '__main__':
    main()