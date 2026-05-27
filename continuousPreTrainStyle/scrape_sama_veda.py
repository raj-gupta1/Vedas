import os
import json
import re
import time
import html
import random
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL = "https://sacred-texts.com/hin/sv.htm"
WAYBACK_PREFIX = "https://web.archive.org/web/2023/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "sama_veda_pretrain.jsonl")
MAX_RETRIES = 5

RETRY_DELAY = 15
TIMEOUT = 60  # longer timeout — single large page

# Headers to mimic a regular browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Boilerplate text patterns to remove from extracted content
BOILERPLATE_PATTERNS = [
    r"Hymns of the Samaveda.*?Griffith.*?\d{4}",
    r"sacred-texts\.com",
    r"Sacred Texts\s+Hinduism\s+Index",
    r"Buy this Book at Amazon\.com",
    r"Click Here to Buy it Now",
    r"ISTA FLASH DRIVE.*",
    r"The World's Wisdom.*",
    r"in the Palm.*of Your Hand",
]


def fetch_page(url: str) -> str:
    """
    Fetch the full Sama Veda page with retries, trying Wayback Machine first
    then falling back to the direct URL.
    """
    wayback_url = WAYBACK_PREFIX + url
    urls_to_try = [wayback_url, url]

    for attempt in range(MAX_RETRIES):
        for try_url in urls_to_try:
            try:
                print(f"  📡 Trying {try_url[:80]}... (attempt {attempt + 1})")
                resp = requests.get(try_url, headers=HEADERS, timeout=TIMEOUT)
                if resp.status_code == 200:
                    print(f"  ✅ Successfully fetched ({len(resp.text):,} bytes)")
                    return resp.text
                if resp.status_code == 403 and try_url == url:
                    continue
                print(f"  ⚠ HTTP {resp.status_code} for {try_url}")
            except requests.RequestException as e:
                print(f"  ⚠ Request error: {e}")
                continue

        if attempt < MAX_RETRIES - 1:
            sleep_time = RETRY_DELAY * (3 ** attempt)
            print(f"  🔄 Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    raise RuntimeError(f"❌ Failed to fetch page after {MAX_RETRIES} attempts.")


def clean_text(text: str) -> str:
    """Remove boilerplate, excess whitespace, and navigation artifacts."""
    text = html.unescape(text)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove page markers like "p. a1", "p. 115"
    text = re.sub(r'\bp\.\s*[a-z]?\d+\b', '', text)

    # Clean up excessive whitespace while preserving paragraph breaks
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
        elif cleaned_lines and cleaned_lines[-1] != '':
            cleaned_lines.append('')

    text = '\n'.join(cleaned_lines).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def is_boilerplate_paragraph(text: str) -> bool:
    """Check if a paragraph is navigation/boilerplate that should be skipped."""
    skip_phrases = [
        "sacred-texts.com", "Buy this Book", "Amazon.com",
        "Sacred Texts", "Hinduism", "Next:", "Previous",
        "Griffith, [", "Index", "Click Here",
    ]
    for phrase in skip_phrases:
        if phrase in text:
            return True

    # Skip very short navigation artifacts
    if len(text.strip()) < 10:
        return True

    return False


def parse_sama_veda(page_html: str) -> list[dict]:
    """
    Parse the entire Sama Veda page into structured hymn records.

    The page structure is:
      FIRST PART:  h1 → h2(Book) → h3(Chapter) → h4(DECADE N Subject) → p(verses)
      SECOND PART: h1 → h2(Book) → h3(Chapter) → h4(N Subject)         → p(verses)
    """
    soup = BeautifulSoup(page_html, "html.parser")
    body = soup.find("body")
    if not body:
        raise ValueError("Could not find <body> in HTML")

    records = []

    # State tracking as we walk through elements in order
    current_part = ""       # "FIRST PART" or "PART SECOND"
    current_book = ""       # "BOOK I", "BOOK II", etc.
    current_chapter = ""    # "CHAPTER I", "CHAPTER II"
    current_hymn = ""       # "DECADE I" or "I", "II", etc.
    current_subject = ""    # "Agni", "Indra", "Soma Pavamana", etc.
    in_preface = False
    preface_parts = []
    hymn_counter = 0        # Global hymn counter for progress display

    # Iterate over all direct children and descendants of <body>
    for element in body.descendants:
        if not isinstance(element, Tag):
            continue

        tag_name = element.name
        tag_text = element.get_text(strip=True)

        # ── Track structural headers ──────────────────────────────────
        if tag_name == "h1":
            upper_text = tag_text.upper()
            if "FIRST PART" in upper_text:
                current_part = "FIRST PART"
                in_preface = False
                # Save preface if collected
                if preface_parts:
                    preface_text = '\n\n'.join(preface_parts)
                    preface_text = clean_text(preface_text)
                    if preface_text and len(preface_text) > 50:
                        records.append({
                            "text": "PREFACE\n\n" + preface_text,
                            "source": "sacred-texts.com",
                            "collection": "Sama Veda",
                            "translator": "Ralph T.H. Griffith",
                            "part": "Preface",
                            "book": 0,
                            "hymn": "PREFACE",
                            "title": "Preface",
                        })
                    preface_parts = []
                continue
            elif "PART SECOND" in upper_text or "SECOND PART" in upper_text:
                current_part = "SECOND PART"
                in_preface = False
                continue
            elif "PREFACE" in upper_text:
                in_preface = True
                continue
            elif "HYMNS OF THE SAMAVEDA" in upper_text:
                continue

        # Collect preface paragraphs
        if in_preface and tag_name == "p":
            p_text = element.get_text(separator='\n', strip=True)
            if p_text and not is_boilerplate_paragraph(p_text):
                preface_parts.append(p_text)
            continue

        if tag_name == "h2":
            book_match = re.search(r'BOOK\s+(\w+)', tag_text, re.IGNORECASE)
            if book_match:
                current_book = f"BOOK {book_match.group(1).upper()}"
                current_chapter = ""
                current_hymn = ""
                current_subject = ""
            continue

        if tag_name == "h3":
            chapter_match = re.search(r'CHAPTER\s+(\w+)', tag_text, re.IGNORECASE)
            if chapter_match:
                current_chapter = f"CHAPTER {chapter_match.group(1).upper()}"
                current_hymn = ""
                current_subject = ""
            continue

        if tag_name == "h4":
            # Parse the h4 to get hymn/decade label and subject
            h4_text = tag_text.strip()

            if current_part == "FIRST PART":
                # Format: "DECADE I Agni" or "DECADE IV Indra and others"
                decade_match = re.match(
                    r'(DECADE\s+[IXVLCDM]+)\s*(.*)',
                    h4_text, re.IGNORECASE
                )
                if decade_match:
                    current_hymn = decade_match.group(1).upper()
                    current_subject = decade_match.group(2).strip(" .")
                else:
                    current_hymn = h4_text
                    current_subject = ""
            else:
                # SECOND PART format: "I Soma Pavamana" or "VII Indra Agni"
                hymn_match = re.match(
                    r'([IXVLCDM]+\.?)\s*(.*)',
                    h4_text
                )
                if hymn_match:
                    current_hymn = f"HYMN {hymn_match.group(1).rstrip('.')}"
                    current_subject = hymn_match.group(2).strip(" .")
                else:
                    current_hymn = h4_text
                    current_subject = ""
            continue

        # ── Extract verse content from <p> tags ───────────────────────
        if tag_name == "p" and current_part and current_hymn:
            # Skip if this <p> is nested inside another <p> we already processed
            if element.parent and element.parent.name == "p":
                continue

            p_text = element.get_text(separator='\n', strip=True)

            if not p_text or is_boilerplate_paragraph(p_text):
                continue

            # Skip the "Om. Glory to the Samaveda..." invocations within hymns
            if re.match(r'^Om\.\s*Glory to the Samaveda', p_text):
                continue

            # Clean up spacing artifacts from HTML parsing
            p_text = re.sub(r' {2,}', ' ', p_text)

            # ── Build the formatted pretrain text ─────────────────────
            parts = []
            parts.append(f"{current_part}")

            if current_book:
                parts.append(current_book)
            if current_chapter:
                parts.append(current_chapter)
            if current_hymn:
                parts.append(current_hymn)
            if current_subject:
                parts.append(current_subject)

            parts.append("")  # blank line separator
            parts.append(p_text)

            text = '\n'.join(parts)
            text = clean_text(text)

            if text and len(text) > 20:
                # Determine book number for metadata
                book_num_match = re.search(r'BOOK\s+(\w+)', current_book)
                book_num = 0
                if book_num_match:
                    roman = book_num_match.group(1)
                    roman_map = {
                        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
                    }
                    book_num = roman_map.get(roman, 0)

                hymn_counter += 1
                records.append({
                    "text": text,
                    "source": "sacred-texts.com",
                    "collection": "Sama Veda",
                    "translator": "Ralph T.H. Griffith",
                    "part": current_part,
                    "book": book_num,
                    "chapter": current_chapter,
                    "hymn": current_hymn,
                    "title": current_subject,
                })

    return records


def save_records(records: list[dict], output_path: str):
    """Save records in JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def print_summary(records: list[dict]):
    """Analyze and print detailed stats and samples."""
    print(f"\n{'─' * 60}")
    print(f"📊 Summary Statistics")
    print(f"{'─' * 60}")
    print(f"   Total hymn records: {len(records)}")

    total_chars = sum(len(r["text"]) for r in records)
    print(f"   Total characters:   {total_chars:,}")
    if records:
        print(f"   Avg chars/record:   {total_chars // len(records):,}")

    # Per-part breakdown
    for part_name in ["Preface", "FIRST PART", "SECOND PART"]:
        part_records = [r for r in records if r.get("part") == part_name]
        if part_records:
            print(f"\n   {part_name}:")
            print(f"     Records: {len(part_records)}")
            part_chars = sum(len(r["text"]) for r in part_records)
            print(f"     Characters: {part_chars:,}")

            # Book-level counts
            book_counts = {}
            for r in part_records:
                b = r.get("book", 0)
                if b > 0:
                    book_counts[b] = book_counts.get(b, 0) + 1
            if book_counts:
                for b in sorted(book_counts):
                    print(f"       Book {b}: {book_counts[b]} records")

    # Show samples
    if records:
        print(f"\n{'─' * 60}")
        print(f"📝 Sample record (first hymn):")
        print(f"{'─' * 60}")
        # Skip preface, show first actual hymn
        sample = records[1] if len(records) > 1 else records[0]
        print(f"Text preview (first 400 chars):\n{sample['text'][:400]}")
        print(f"{'─' * 60}")
        meta_keys = ["part", "book", "chapter", "hymn", "title"]
        meta = {k: sample.get(k, "") for k in meta_keys}
        print(f"Metadata: {meta}")

        print(f"\n📝 Sample record (Second Part):")
        print(f"{'─' * 60}")
        part2 = [r for r in records if r.get("part") == "SECOND PART"]
        if part2:
            sample2 = part2[0]
            print(f"Text preview (first 400 chars):\n{sample2['text'][:400]}")
            print(f"{'─' * 60}")
            meta2 = {k: sample2.get(k, "") for k in meta_keys}
            print(f"Metadata: {meta2}")


def main():
    print("=" * 60)
    print("  Sama Veda Scraper — sacred-texts.com via Wayback Machine")
    print("=" * 60)
    print(f"Output file: {OUTPUT_PATH}\n")

    # Step 1: Fetch the full page
    print("📖 Fetching Sama Veda page...")
    page_html = fetch_page(BASE_URL)

    # Step 2: Parse hymns from the HTML
    print("\n🔍 Parsing hymns from HTML...")
    records = parse_sama_veda(page_html)
    print(f"   Parsed {len(records)} records")

    if not records:
        print("⚠️ No records extracted. Check the HTML structure.")
        return

    # Step 3: Save to JSONL
    save_records(records, OUTPUT_PATH)
    print(f"\n✅ Saved to {OUTPUT_PATH}")

    # Step 4: Print summary
    print_summary(records)

    print(f"\n🎉 Sama Veda scraping complete!")


if __name__ == "__main__":
    main()
