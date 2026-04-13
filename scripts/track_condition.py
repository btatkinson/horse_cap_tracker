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
from datetime import date, datetime

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = os.environ['DATA_PATH']
META_PATH = os.path.join(DATA_PATH, 'historical_meta.pkl')

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

def get_nws_forecast(track: str, target_date: date = None) -> dict | None:
    """Get weather forecast from NWS API for a track location."""
    meta = TRACKS.get(track.lower())
    if not meta:
        print(f"  Unknown track: {track}")
        return None

    if meta['state'] == 'ON':
        print(f"  NWS does not cover {track} (Canada). Skipping forecast.")
        return None

    lat, lon = meta['lat'], meta['lon']

    try:
        headers = {'User-Agent': 'horse-handicapper-tool (personal use)'}

        resp = requests.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=10, headers=headers)
        if resp.status_code != 200:
            print(f"  NWS points lookup failed: {resp.status_code}")
            return None

        forecast_url = resp.json()['properties']['forecast']
        resp = requests.get(forecast_url, timeout=10, headers=headers)
        if resp.status_code != 200:
            print(f"  NWS forecast fetch failed: {resp.status_code}")
            return None

        periods = resp.json()['properties']['periods']

        if target_date is None:
            target_date = date.today()

        # Find the best matching period (prefer daytime)
        best_period = None
        for period in periods:
            period_start = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
            if period_start.date() == target_date:
                if period['isDaytime'] or best_period is None:
                    best_period = period
                    if period['isDaytime']:
                        break

        if not best_period:
            for period in periods:
                period_start = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
                if period_start.date() >= target_date:
                    best_period = period
                    break

        if not best_period:
            print(f"  No forecast available for {target_date}")
            return None

        precip_pct = best_period.get('probabilityOfPrecipitation', {}).get('value', 0) or 0
        wind_speed = best_period.get('windSpeed', 'Unknown')
        temp = best_period.get('temperature', 'Unknown')
        temp_unit = best_period.get('temperatureUnit', 'F')
        short_forecast = best_period.get('shortForecast', '')
        detailed = best_period.get('detailedForecast', '')

        return {
            'source': 'nws_forecast',
            'dirt': estimate_dirt_condition(precip_pct, short_forecast, detailed),
            'turf': estimate_turf_condition(precip_pct, short_forecast, detailed),
            'precip_pct': precip_pct,
            'temperature': f"{temp}°{temp_unit}",
            'wind': wind_speed,
            'forecast': short_forecast,
            'detailed': detailed,
            'period_name': best_period.get('name', ''),
            'target_date': str(target_date),
        }

    except Exception as e:
        print(f"  NWS forecast failed: {e}")
        return None


def estimate_dirt_condition(precip_pct: int, short: str, detailed: str) -> str:
    text = (short + ' ' + detailed).lower()
    heavy_rain = any(w in text for w in ['heavy rain', 'thunderstorm', 'downpour', 'flood'])
    rain = any(w in text for w in ['rain', 'showers', 'drizzle', 'precipitation'])

    if heavy_rain or precip_pct >= 80:
        return 'Sloppy (estimated)'
    elif (rain and precip_pct >= 60) or precip_pct >= 70:
        return 'Muddy (estimated)'
    elif (rain and precip_pct >= 40) or precip_pct >= 50:
        return 'Good (estimated)'
    else:
        return 'Fast (estimated)'


def estimate_turf_condition(precip_pct: int, short: str, detailed: str) -> str:
    text = (short + ' ' + detailed).lower()
    heavy_rain = any(w in text for w in ['heavy rain', 'thunderstorm', 'downpour'])
    rain = any(w in text for w in ['rain', 'showers', 'drizzle'])

    if heavy_rain or precip_pct >= 80:
        return 'Yielding (estimated) — turf races may come off'
    elif (rain and precip_pct >= 60) or precip_pct >= 70:
        return 'Soft (estimated)'
    elif (rain and precip_pct >= 40) or precip_pct >= 50:
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
        lines.append(f"  Precip chance: {info.get('precip_pct', '?')}%")
        lines.append(f"  Temp: {info.get('temperature', '?')}, Wind: {info.get('wind', '?')}")

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