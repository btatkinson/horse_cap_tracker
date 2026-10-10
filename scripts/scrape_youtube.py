"""
scrape_youtube.py

Pull a YouTube video's transcript (auto-generated or manual captions)
and save it in the same directory layout as scrape_experts.py so
send_experts.py picks it up like any other expert source.

Output:
    {DATA_PATH}/raw_expert_picks/{date}/{track}/{expert_name}.txt

Usage:
    # Most explicit form
    python scrape_youtube.py \\
        --video EG8nJG9QRCo \\
        --expert matthew_desantis \\
        --track churchill-downs \\
        --date 2026-04-25

    # URL also works (we'll extract the ID)
    python scrape_youtube.py \\
        --video https://www.youtube.com/watch?v=EG8nJG9QRCo \\
        --expert matthew_desantis \\
        --track churchill-downs

    # Date defaults to today
    # Track defaults to nothing — must be specified

When captions don't exist for a video, the script exits with code 2
and prints what to do next. The current scope of this tool is YouTube
captions only — Whisper-based transcription of caption-less videos
is a separate, heavier path you can add later.

Library notes:
    youtube-transcript-api (v1.2+) uses an instance API:
        YouTubeTranscriptApi().fetch(video_id)
    Older tutorials online show YouTubeTranscriptApi.get_transcript(...),
    which is the pre-1.0 API. Don't be misled.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime, date

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_PATH = Path(os.environ['DATA_PATH'])

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
except ImportError:
    raise ImportError(
        "pip install youtube-transcript-api\n"
        "(version 1.2+ recommended; older versions have a different API)"
    )


def extract_video_id(value):
    """
    Accept either a raw video ID or a full URL and return the ID.

    YouTube IDs are 11 characters of [A-Za-z0-9_-]. We accept anything
    matching that pattern as-is, and try to pull an ID out of common
    URL forms otherwise.
    """
    # Already looks like an ID
    if re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
        return value

    # Try to extract from URL
    # Patterns: youtube.com/watch?v=ID, youtu.be/ID, youtube.com/shorts/ID,
    # youtube.com/embed/ID
    patterns = [
        r'(?:youtube\.com/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})',
        r'(?:youtu\.be/)([A-Za-z0-9_-]{11})',
        r'(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})',
        r'(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, value)
        if m:
            return m.group(1)

    raise ValueError(
        f"Couldn't extract a YouTube video ID from: {value!r}\n"
        f"Pass either an 11-character ID or a standard YouTube URL."
    )


def fetch_transcript(video_id, language_codes=('en', 'en-US', 'en-GB')):
    """
    Fetch a transcript, preferring manually-created over auto-generated
    when both exist (manual captions are usually more accurate).

    Returns: FetchedTranscript object (iterable of snippets with
             .text / .start / .duration attributes)

    Raises: TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
            and other exceptions from youtube_transcript_api.
    """
    api = YouTubeTranscriptApi()

    # The library's .fetch() will already prefer manual over generated
    # if you let it pick, but we want explicit control so we can tell
    # the user which one we got.
    try:
        transcript_list = api.list(video_id)
    except AttributeError:
        # Older library versions — fall back to .fetch() and hope for the best
        return api.fetch(video_id, languages=list(language_codes)), 'unknown'

    # Try manual first
    try:
        transcript = transcript_list.find_manually_created_transcript(language_codes)
        return transcript.fetch(), 'manual'
    except NoTranscriptFound:
        pass

    # Fall back to auto-generated
    try:
        transcript = transcript_list.find_generated_transcript(language_codes)
        return transcript.fetch(), 'auto-generated'
    except NoTranscriptFound:
        # Last resort: any English-ish transcript at all, even if marked
        # neither manual nor generated cleanly
        for t in transcript_list:
            if any(t.language_code.startswith(lc.split('-')[0]) for lc in language_codes):
                return t.fetch(), 'fallback'
        raise


def transcript_to_text(transcript, with_timestamps=False):
    """
    Convert a FetchedTranscript object to a plain text string.

    Without timestamps: snippets joined with spaces, then whitespace
    normalized. This is what you want for handing off to an LLM.

    With timestamps: one snippet per line, prefixed with [MM:SS].
    Useful for human review or debugging.
    """
    snippets = list(transcript)

    if with_timestamps:
        lines = []
        for s in snippets:
            mins = int(s.start // 60)
            secs = int(s.start % 60)
            lines.append(f"[{mins:02d}:{secs:02d}] {s.text}")
        return '\n'.join(lines)

    # Plain text: join snippets, collapse whitespace, fix YouTube's
    # tendency to break sentences across snippets without punctuation
    raw = ' '.join(s.text for s in snippets)
    # Collapse runs of whitespace
    raw = re.sub(r'\s+', ' ', raw)
    # Strip the [Music] / [Applause] tags that auto-captions sometimes leave
    raw = re.sub(r'\[(?:Music|Applause|Laughter|Inaudible)\]', '', raw,
                 flags=re.IGNORECASE)
    return raw.strip()


def parse_date_arg(date_str):
    """Match the format support of scrape_experts.py."""
    for fmt in ['%Y-%m-%d', '%m-%d-%Y', '%m-%d-%y']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Can't parse date: {date_str}")


def normalize_expert_name(name):
    """
    Convert "Matthew DeSantis" or "matt-desantis" -> "matthew_desantis".
    Keeps lowercase letters, replaces everything else with underscores,
    collapses runs of underscores. Matches the convention used in
    scrape_experts.py.
    """
    snake = re.sub(r"[^a-zA-Z]+", '_', name).strip('_').lower()
    return snake


def main():
    parser = argparse.ArgumentParser(
        description='Save a YouTube video transcript as an expert pick file.'
    )
    parser.add_argument('--video', required=True,
                        help='YouTube video ID or full URL')
    parser.add_argument('--expert', required=True,
                        help='Expert name (will be snake-cased; '
                             'e.g. "Matthew DeSantis" -> matthew_desantis)')
    parser.add_argument('--track', required=True,
                        help='Track key (e.g. churchill-downs, keeneland)')
    parser.add_argument('--date', default=None,
                        help='Target date YYYY-MM-DD; defaults to today')
    parser.add_argument('--with-timestamps', action='store_true',
                        help='Include [MM:SS] timestamps in saved text')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing file (default: skip)')
    args = parser.parse_args()

    target_date = parse_date_arg(args.date) if args.date else date.today()
    date_str = target_date.strftime('%Y-%m-%d')

    try:
        video_id = extract_video_id(args.video)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    expert_name = normalize_expert_name(args.expert)
    if not expert_name:
        print(f"ERROR: expert name '{args.expert}' normalized to empty string",
              file=sys.stderr)
        sys.exit(1)

    out_dir = DATA_PATH / 'raw_expert_picks' / date_str / args.track
    sent_dir = DATA_PATH / 'already_sent_eps' / date_str / args.track
    out_path = out_dir / f"{expert_name}.txt"

    # Honor the same skip rules as scrape_experts.py
    if (sent_dir / f"{expert_name}.txt").exists():
        print(f"Already sent: {sent_dir / (expert_name + '.txt')}")
        return
    if out_path.exists() and not args.overwrite:
        print(f"Already scraped: {out_path}")
        print("(Pass --overwrite to replace.)")
        return

    print(f"Fetching transcript for video {video_id}...")
    try:
        transcript, source = fetch_transcript(video_id)
    except TranscriptsDisabled:
        print(f"\nERROR: Captions are disabled for video {video_id}.")
        print("Next step: download audio with yt-dlp and transcribe via Whisper.")
        sys.exit(2)
    except NoTranscriptFound:
        print(f"\nERROR: No English transcript found for video {video_id}.")
        print("The video may have non-English captions only, or no captions.")
        print("Next step: download audio with yt-dlp and transcribe via Whisper.")
        sys.exit(2)
    except VideoUnavailable:
        print(f"\nERROR: Video {video_id} is unavailable (private/deleted/region-locked).")
        sys.exit(2)
    except Exception as e:
        # Catch-all for the library's various transient/internal errors.
        # The library has been known to break when YouTube changes things.
        print(f"\nERROR: Unexpected failure fetching transcript: "
              f"{type(e).__name__}: {e}")
        print("If this persists, the youtube-transcript-api library may need")
        print("an update (pip install -U youtube-transcript-api), or YouTube")
        print("has changed something. Whisper fallback is the alternative.")
        sys.exit(2)

    print(f"  Got {len(transcript)}-snippet transcript ({source})")
    text = transcript_to_text(transcript, with_timestamps=args.with_timestamps)

    if len(text) < 100:
        print(f"WARNING: transcript is suspiciously short ({len(text)} chars)")
        print("Inspect output before relying on it.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Saved {len(text)} chars to {out_path}")


if __name__ == '__main__':
    main()