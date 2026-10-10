"""
make_blind_forms.py

Copy chopped race PDFs with today's morning-line odds blacked out, for the
odds-blind LLM variant (send_openrouter.py --forms chopped_forms_blind).

Uses true PDF redaction: the black box is drawn AND the text underneath is
removed, so text extraction can't see the ML either.

The ML is found by its position in each horse's header, per PP format:
    DRF       bold span right after the program number, left margin (x < 60)
    DRF (old) large (>= 11pt) span after the saddle-cloth color, x < 40
    Equibase  bold span right before "Owner:"
    Brisnet   span right after the "Own:" line, left margin (x < 30)
Workout ranks like "23/38" share the odds pattern but are italic, so they
never match.

Usage:
    python make_blind_forms.py --date 2026-10-08 --track keeneland
    python make_blind_forms.py --all          # every card under chopped_forms/

Reads from:  {DATA_PATH}/chopped_forms/{date}/{track}/race_*.pdf
Writes to:   {DATA_PATH}/chopped_forms_blind/{date}/{track}/race_*.pdf
"""

import os, re, argparse
from pathlib import Path

import fitz  # pip install pymupdf
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '..' / '.env')

DATA_PATH = Path(os.environ['DATA_PATH'])
SRC = DATA_PATH / 'chopped_forms'
DST = DATA_PATH / 'chopped_forms_blind'

ODDS_RE = re.compile(r'^(?:\d{1,2}[-/]\d{1,2}|EVN?|Even)$')
PROGRAM_RE = re.compile(r'^\d{1,2}[A-Z]?$')


def find_ml_spans(page) -> list:
    """Return bboxes of morning-line odds spans on one page."""
    spans = [s for b in page.get_text('dict')['blocks']
             for l in b.get('lines', []) for s in l['spans'] if s['text'].strip()]
    hits = []
    for i, s in enumerate(spans):
        text = s['text'].strip()
        if not ODDS_RE.match(text) or 'Italic' in s['font']:
            continue
        prev = spans[i - 1]['text'].strip() if i else ''
        nxt = spans[i + 1]['text'].strip() if i + 1 < len(spans) else ''
        x0 = s['bbox'][0]
        drf = PROGRAM_RE.match(prev) and x0 < 60
        # Older DRF layout: large ML after the saddle-cloth color ("Red").
        drf_cloth = x0 < 40 and s['size'] >= 11
        equibase = nxt.startswith('Owner:')
        brisnet = prev.startswith('Own:') and x0 < 30
        if drf or drf_cloth or equibase or brisnet:
            hits.append(fitz.Rect(s['bbox']))
    return hits


def blind_pdf(src: Path, dst: Path) -> int:
    """Redact ML odds in one race PDF; return the number of redactions."""
    doc = fitz.open(str(src))
    n = 0
    for page in doc:
        for rect in find_ml_spans(page):
            page.add_redact_annot(rect + (-1, -1, 1, 1), fill=(0, 0, 0))
            n += 1
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dst), garbage=3, deflate=True)
    doc.close()
    return n


def blind_card(date: str, track: str) -> dict[int, int]:
    counts = {}
    for f in sorted((SRC / date / track).glob('race_*.pdf'),
                    key=lambda p: int(re.search(r'(\d+)', p.stem).group(1))):
        rn = int(re.search(r'(\d+)', f.stem).group(1))
        counts[rn] = blind_pdf(f, DST / date / track / f.name)
    return counts


def main():
    parser = argparse.ArgumentParser(description='Copy race PDFs with morning-line odds redacted')
    parser.add_argument('--date')
    parser.add_argument('--track')
    parser.add_argument('--all', action='store_true', help='Every card under chopped_forms/')
    args = parser.parse_args()

    if args.all:
        cards = [(p.parent.name, p.name) for p in sorted(SRC.glob('*/*')) if p.is_dir()]
    elif args.date and args.track:
        cards = [(args.date, args.track)]
    else:
        parser.error('give --date and --track, or --all')

    for date, track in cards:
        counts = blind_card(date, track)
        print(f"{date}/{track}: " + " ".join(f"R{r}:{n}" for r, n in counts.items()) +
              f"  ({sum(counts.values())} odds redacted)")


if __name__ == '__main__':
    main()
