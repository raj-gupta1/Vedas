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
BLACK_YV_BASE = "https://sacred-texts.com/hin/yv/"
WHITE_YV_BASE = "https://sacred-texts.com/hin/wyv/"
WAYBACK_PREFIX = "https://web.archive.org/web/2023/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "yajur_veda_pretrain.jsonl")
MAX_RETRIES = 5


RETRY_DELAY = 10
TIMEOUT = 30

# Headers to mimic a regular browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Boilerplate patterns to remove from extracted text
BOILERPLATE_PATTERNS = [
    r"The Texts of the White Yajurveda.*Griffith.*",
    r"The Yajur Veda.*Arthur Berriedale Keith.*",
    r"sacred-texts\.com",
    r"Sacred Texts\s+Hinduism\s+Index",
    r"Buy this Book at Amazon\.com",
    r"Click Here to Buy it Now",
    r"ISTA FLASH DRIVE.*",
    r"The World's Wisdom.*",
    r"in the Palm.*of Your Hand",
    r"Previous\s+Next",
    r"Next:\s*Book\s+.*",
    r"Next:\s*BOOK\s+.*",
    r"Next:\s*Kanda\s+.*",
    r"Next:\s*KANDA\s+.*",
    r"Next:\s*Preface",
]

def fetch_page(url: str) -> str:
    """Fetch a page with retries, using Wayback Machine as primary source."""
    wayback_url = WAYBACK_PREFIX + url
    urls_to_try = [wayback_url, url]

    for attempt in range(MAX_RETRIES):
        for try_url in urls_to_try:
            try:
                # Add a tiny delay to avoid hitting rate limits
                time.sleep(random.uniform(1.0, 3.0))
                resp = requests.get(try_url, headers=HEADERS, timeout=TIMEOUT)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code == 403 and try_url == url:
                    continue
                print(f"  ⚠ HTTP {resp.status_code} for {try_url} (attempt {attempt + 1})")
            except requests.RequestException as e:
                print(f"  ⚠ Request error: {e} (attempt {attempt + 1})")
                continue

        if attempt < MAX_RETRIES - 1:
            sleep_time = RETRY_DELAY * (2 ** attempt)
            print(f"  🔄 Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    raise RuntimeError(f"❌ Failed to fetch page {url} after {MAX_RETRIES} attempts.")

def clean_text(text: str) -> str:
    """Remove boilerplate, excess whitespace, and navigation artifacts."""
    text = html.unescape(text)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove physical page markers like "[p. 1]" or "p. 1"
    text = re.sub(r'\[p\.\s*\d+\]', '', text)
    text = re.sub(r'\bp\.\s*[a-z]?\d+\b', '', text)

    # Clean up excess whitespace
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
        "Griffith, [", "Keith, [", "Index", "Click Here",
        "Title Page", "Contents", "Errata", "Corrigenda"
    ]
    for phrase in skip_phrases:
        if phrase in text:
            return True
    
    if len(text.strip()) < 10:
        return True
        
    return False

# ─── Black Yajur Veda (Taittiriya Samhita) Parser ────────────────────────────
def parse_black_yajur_veda(page_html: str, book_num: int) -> list[dict]:
    """
    Parse a Black Yajur Veda Kanda HTML page into structured Anuvaka records.
    Structure:
      <h1> → Kanda (e.g. KANDA I)
      <h2> → Prapathaka (e.g. PRAPATHAKA I)
      <i>  → Section Title (e.g. The New and Full Moon Sacrifices)
      <h3> → Anuvaka label (e.g. i. 1. 1.)
      <p>  → Verse content immediately following the <h3> tag.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    body = soup.find("body") or soup
    
    records = []
    
    current_kanda = f"KANDA {book_num}"
    current_prapathaka = ""
    current_section_title = ""
    current_anuvaka = ""
    
    # Track element iteration to collect paragraphs after each H3
    current_verses = []
    
    # We walk through all elements in order
    for element in body.descendants:
        if not isinstance(element, Tag):
            continue
            
        name = element.name
        
        if name == "h1":
            text = element.get_text(strip=True)
            if "KANDA" in text.upper():
                current_kanda = text.strip()
                current_prapathaka = ""
                current_section_title = ""
                current_anuvaka = ""
                
        elif name == "h2":
            text = element.get_text(strip=True)
            if "PRAPATHAKA" in text.upper():
                current_prapathaka = text.strip()
                current_section_title = ""
                current_anuvaka = ""
                
        elif name == "i":
            # Section title
            text = element.get_text(strip=True)
            if text and not is_boilerplate_paragraph(text):
                current_section_title = text.strip()
                
        elif name == "h3":
            # Before we start a new Anuvaka, save the previous one if we have content
            if current_anuvaka and current_verses:
                save_black_record(records, current_kanda, current_prapathaka, 
                                  current_section_title, current_anuvaka, current_verses, book_num)
                current_verses = []
                
            current_anuvaka = element.get_text(strip=True)
            
        elif name == "p" and current_anuvaka:
            if element.parent and element.parent.name == "p":
                continue
                
            text = element.get_text(separator='\n', strip=True)
            if not text or is_boilerplate_paragraph(text):
                continue
                
            # If the paragraph is in all-caps, it's likely a division title, not verse text
            if text.isupper() and len(text) < 100:
                current_section_title = text.strip()
                continue
                
            current_verses.append(text)
            
    # Save the very last record on the page
    if current_anuvaka and current_verses:
        save_black_record(records, current_kanda, current_prapathaka, 
                          current_section_title, current_anuvaka, current_verses, book_num)
                          
    return records

def save_black_record(records: list, kanda: str, prapathaka: str, title: str, anuvaka: str, verses: list, book_num: int):
    """Format and save a single Black Yajur Veda Anuvaka record."""
    verses_text = "\n\n".join(verses).strip()
    
    parts = ["BLACK YAJUR VEDA (TAITTIRIYA SAMHITA)"]
    if kanda:
        parts.append(kanda)
    if prapathaka:
        parts.append(prapathaka)
    if anuvaka:
        parts.append(f"ANUVAKA {anuvaka}")
    if title:
        parts.append(title)
        
    parts.append("")
    parts.append(verses_text)
    
    full_text = clean_text("\n".join(parts))
    
    # Try to parse prapathaka number
    prapathaka_num = 0
    if prapathaka:
        roman_match = re.search(r'PRAPATHAKA\s+([IXVLCDM]+)', prapathaka, re.IGNORECASE)
        if roman_match:
            roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
            prapathaka_num = roman_map.get(roman_match.group(1).upper(), 0)

    if full_text and len(verses_text) > 20:
        records.append({
            "text": full_text,
            "source": "sacred-texts.com",
            "collection": "Yajur Veda",
            "sub_collection": "Black Yajur Veda",
            "translator": "Arthur Berriedale Keith",
            "kanda": book_num,
            "prapathaka": prapathaka_num,
            "anuvaka": anuvaka,
            "title": title
        })

# ─── White Yajur Veda (Vajasaneya Samhita) Parser ────────────────────────────
def parse_white_yajur_veda(page_html: str, book_num: int) -> list[dict]:
    """
    Parse a White Yajur Veda Book HTML page into individual verse records.
    White Yajur Veda has numbered verses (e.g. 1 to N) inside physical book pages.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    body = soup.find("body") or soup
    
    # 1. Determine book title from h3 or title tags
    book_title = f"BOOK {book_num}"
    h3 = soup.find("h3")
    if h3:
        book_title = h3.get_text(strip=True)
        
    # 2. Extract and join all text paragraphs
    p_texts = []
    for p in body.find_all("p"):
        if p.parent and p.parent.name == "p":
            continue
        p_text = p.get_text(separator=' ', strip=True)
        if p_text and not is_boilerplate_paragraph(p_text):
            p_texts.append(p_text)
            
    full_book_text = " ".join(p_texts)
    
    # Clean some raw spacing/newlines
    full_book_text = re.sub(r'\s+', ' ', full_book_text).strip()
    
    # Remove page numbers like "[p. 2]" from the stream
    full_book_text = re.sub(r'\[p\.\s*\d+\]', '', full_book_text)
    full_book_text = re.sub(r'\bp\.\s*[a-z]?\d+\b', '', full_book_text)
    
    # 3. Split the text into individual numbered verses using a regular expression.
    # The first verse starts without a number, followed by "2 Verse 2 text... 3 Verse 3 text..."
    parts = re.split(r'\s*\b(\d+)\b\s*', full_book_text)
    
    raw_verses = []
    
    # First part corresponds to verse 1
    v1_text = parts[0].strip()
    if v1_text:
        raw_verses.append((1, v1_text))
        
    # Remaining parts are paired as (verse_num, verse_text)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            try:
                v_num = int(parts[i])
                v_text = parts[i+1].strip()
                if v_text:
                    raw_verses.append((v_num, v_text))
            except ValueError:
                continue
                
    # 4. Create formatted records
    records = []
    for v_num, v_text in raw_verses:
        formatted_parts = [
            "WHITE YAJUR VEDA (VAJASANEYA SAMHITA)",
            f"BOOK {book_num}",
            f"VERSE {v_num}",
            book_title,
            "",
            v_text
        ]
        
        full_text = clean_text("\n".join(formatted_parts))
        
        if full_text and len(v_text) > 15:
            records.append({
                "text": full_text,
                "source": "sacred-texts.com",
                "collection": "Yajur Veda",
                "sub_collection": "White Yajur Veda",
                "translator": "Ralph T.H. Griffith",
                "book": book_num,
                "verse": v_num,
                "title": book_title
            })
            
    return records

# ─── Main Script Orchestrator ────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Yajur Veda Scraper — Black and White Samhitas Unified")
    print("=" * 70)
    print(f"Output file: {OUTPUT_PATH}\n")
    
    all_records = []
    
    # ─── STEP 1: Scrape Black Yajur Veda (7 Kandas) ──────────────────────────
    print("📖 Scrape Black Yajur Veda (Taittiriya Samhita)...")
    for book_num in range(1, 8):
        url = f"{BLACK_YV_BASE}yv0{book_num}.htm"
        print(f"  📥 Fetching Kanda {book_num} ({url})...")
        try:
            page_html = fetch_page(url)
            records = parse_black_yajur_veda(page_html, book_num)
            all_records.extend(records)
            print(f"    ✅ Parsed {len(records)} Anuvaka records")
        except Exception as e:
            print(f"    ❌ Error scraping Kanda {book_num}: {e}")
            
    print(f"  ✨ Finished Black Yajur Veda. Total records so far: {len(all_records)}\n")
    
    # ─── STEP 2: Scrape White Yajur Veda (40 Books) ──────────────────────────
    print("📖 Scrape White Yajur Veda (Vajasaneya Samhita)...")
    for book_num in tqdm(range(1, 41), desc="White Yajur Veda", unit="book"):
        url = f"{WHITE_YV_BASE}wyvbk{book_num:02d}.htm"
        try:
            page_html = fetch_page(url)
            records = parse_white_yajur_veda(page_html, book_num)
            all_records.extend(records)
        except Exception as e:
            print(f"\n    ❌ Error scraping Book {book_num}: {e}")
            
    print(f"\n  ✨ Finished White Yajur Veda. Overall combined records: {len(all_records)}\n")
    
    # ─── STEP 3: Write to JSONL ──────────────────────────────────────────────
    if all_records:
        print(f"💾 Saving {len(all_records):,} records to {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"✅ Success! Pre-training dataset compiled successfully.")
        
        # ─── STEP 4: Summary Statistics ──────────────────────────────────────
        total_chars = sum(len(r["text"]) for r in all_records)
        black_records = [r for r in all_records if r.get("sub_collection") == "Black Yajur Veda"]
        white_records = [r for r in all_records if r.get("sub_collection") == "White Yajur Veda"]
        
        print("\n" + "─" * 70)
        print("📊 Dataset Summary Statistics")
        print("─" * 70)
        print(f"  Combined Records: {len(all_records):,}")
        print(f"  Combined Chars:   {total_chars:,}")
        print(f"  Avg Chars/Record: {total_chars // len(all_records):,}")
        print()
        print(f"  🖤 Black Yajur Veda: {len(black_records):,} Anuvaka records")
        print(f"  🤍 White Yajur Veda: {len(white_records):,} Verse records")
        print("─" * 70)
        
        # Print a sample of each to confirm formatting
        if black_records:
            print("\n📝 Sample Black Yajur Veda Record:")
            print("─" * 40)
            sample = black_records[0]
            print(sample["text"][:400])
            print("...\n" + "─" * 40)
            
        if white_records:
            print("\n📝 Sample White Yajur Veda Record:")
            print("─" * 40)
            sample = white_records[0]
            print(sample["text"][:400])
            print("...\n" + "─" * 40)
            
    else:
        print("⚠️ No records were successfully scraped or parsed.")
        
    print("\n🎉 Yajur Veda preparation complete!")

if __name__ == "__main__":
    main()
