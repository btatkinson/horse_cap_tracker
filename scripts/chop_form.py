"""
chop_form.py — Split a past-performance PDF into per-race PDFs.

Works with both DRF formats:
  - Gulfstream/1ST style  (footer: "GP, race 3, page:6")
  - Keeneland/Equibase style (footer: "04/04/2026 Race 5")

Usage:
    python chop_form.py                                    # auto-finds latest PDF
    python chop_form.py --pdf GP--04-05-2026.pdf
    python chop_form.py --pdf 4-4-26-kee-ppspdf.pdf --date 2026-04-04
    python chop_form.py --pdf GP--04-05-2026.pdf --track gulfstream-park

Reads from:  ../data/forms/
Writes to:   ../data/chopped_forms/{date}/{track}/race_N.pdf
"""

import os, re, argparse
from datetime import date
from pathlib import Path

try:
    import fitz  # pip install pymupdf
except ImportError:
    raise ImportError("pip install pymupdf")

SCRIPT_DIR = Path(__file__).parent
FORMS_DIR = SCRIPT_DIR / ".." / "data" / "forms"
OUT_BASE  = SCRIPT_DIR / ".." / "data" / "chopped_forms"


# ── helpers ──────────────────────────────────────────────────────────

def get_page_race(page) -> int | None:
    """Extract race number from a page's header/footer line (text-only sniff)."""
    text = page.get_text()
    # DRF/1ST style: "GP, race 3, page:6"  or  "GP, race: 1, page:1"
    m = re.search(r', race:?\s*(\d+), page:\d+', text)
    if m:
        return int(m.group(1))
    # Equibase style: "04/04/2026 Race 5"
    m = re.search(r'\d{2}/\d{2}/\d{4}\s+Race\s+(\d+)', text)
    if m:
        return int(m.group(1))
    # Brisnet style: header has a standalone line "Race N" (PyMuPDF emits the
    # top-right header tag as its own block, separate from the rest of the
    # banner). The body text never contains "Race N" as a standalone line —
    # it appears as "Races 1-2-3", "Such A Race Since Then", etc.
    if "Ultimate PP" in text or "brisnet" in text.lower():
        m = re.search(r'(?m)^Race\s+(\d+)\s*$', text)
        if m:
            return int(m.group(1))
    return None


def detect_track(text: str, filename: str) -> str:
    """Best-effort track detection from filename or content."""
    fn = filename.lower()
    mapping = {
        'kee': 'keeneland', 'cd': 'churchill-downs', 'sar': 'saratoga',
        'gp': 'gulfstream-park', 'sa': 'santa-anita', 'aq': 'aqueduct',
        'op': 'oaklawn', 'tam': 'tampa', 'fg': 'fair-grounds',
        'dmr': 'del-mar', 'pim': 'pimlico', 'lrl': 'laurel',
    }
    for code, name in mapping.items():
        if re.search(rf'(?:^|[^a-z]){code}(?:[^a-z]|$)', fn):
            return name
    header = text[:3000].lower()
    for name in ['keeneland', 'gulfstream-park', 'churchill downs', 'santa anita',
                 'saratoga', 'belmont', 'aqueduct', 'oaklawn', 'del mar',
                 'fair grounds', 'tampa bay', 'pimlico', 'laurel']:
        if name in header:
            return name.replace(' ', '-')
    return 'unknown'


def detect_date(filename: str) -> str:
    """Try to pull a date from the filename, fall back to today. Returns YYYY-MM-DD."""
    # "GP--04-05-2026.pdf"
    m = re.search(r'(\d{2})-(\d{2})-(\d{4})', filename)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    # "4-4-26-kee"  (2-digit year)
    m = re.search(r'(\d{1,2})-(\d{1,2})-(\d{2})(?=-|$)', filename)
    if m:
        yr = int(m.group(3))
        yr = yr + 2000 if yr < 100 else yr
        return f"{yr}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return date.today().strftime('%Y-%m-%d')


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chop PP PDF into per-race PDFs")
    parser.add_argument("--pdf",   help="PDF filename in ../data/forms/")
    parser.add_argument("--date",  help="Override date folder (e.g. 2026-04-05)")
    parser.add_argument("--track", help="Override track name")
    args = parser.parse_args()

    # ── locate PDF ───────────────────────────────────────────────────
    if args.pdf:
        pdf_path = FORMS_DIR / args.pdf
        if not pdf_path.exists():
            pdf_path = Path(args.pdf)
    else:
        pdfs = list(FORMS_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs in {FORMS_DIR}")
            return
        # Pick most recently modified — "the one Blake just downloaded"
        # rather than whatever sorts last alphabetically.
        pdf_path = max(pdfs, key=lambda p: p.stat().st_mtime)
        print(f"Auto-selected: {pdf_path.name}")

    if not pdf_path.exists():
        print(f"Not found: {pdf_path}")
        return

    # ── scan pages → race numbers ────────────────────────────────────
    src = fitz.open(str(pdf_path))
    page_map: dict[int, list[int]] = {}   # {race_num: [page_indices]}

    for i, page in enumerate(src):
        rn = get_page_race(page)
        if rn is not None:
            page_map.setdefault(rn, []).append(i)

    if not page_map:
        print("Could not detect race boundaries in this PDF.")
        return

    # ── resolve date & track ─────────────────────────────────────────
    date_str = args.date or detect_date(pdf_path.name)
    first_page_text = src[page_map[min(page_map)][0]].get_text()
    track = args.track or detect_track(first_page_text, pdf_path.name)

    out_dir = OUT_BASE / date_str / track
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── write per-race PDFs ──────────────────────────────────────────
    print(f"  {len(page_map)} races  →  {out_dir}/")
    for rn in sorted(page_map):
        out_path = out_dir / f"race_{rn}.pdf"
        dst = fitz.open()
        for page_idx in page_map[rn]:
            dst.insert_pdf(src, from_page=page_idx, to_page=page_idx)
        dst.save(str(out_path))
        dst.close()
        print(f"    race_{rn}.pdf  ({len(page_map[rn])} pages)")

    src.close()
    print("Done.")


if __name__ == "__main__":
    main()