import os
import json
import re
import time
import html
import random
from typing import Optional
import requests
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

# Roman numeral mapping for book numbers
ROMAN = {
    1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
    6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
}

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL = "https://sacred-texts.com/hin/rigveda/"
WAYBACK_PREFIX = "https://web.archive.org/web/2023/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "rig_veda_pretrain.jsonl")
REQUEST_DELAY = 5           # base delay in seconds between requests (higher to avoid rate limits)


MAX_RETRIES = 5            # max retry attempts per page
RETRY_DELAY = 15           # base retry delay (will use exponential backoff)
TIMEOUT = 30               # request timeout in seconds

# Boilerplate text patterns to remove from extracted content
BOILERPLATE_PATTERNS = [
    r"Rig Veda,?\s*tr\.? by Ralph T\.?H\.?\s*Griffith,?\s*\[1896\],?\s*at sacred-texts\.com",
    r"Buy this Book at Amazon\.com",
    r"Sacred Texts\s+Hinduism\s+Index",
    r"Book \d+ Index",
    r"Previous\s+Next",
    r"Next:\s*HYMN\s+.*",
    r"Next:\s*Hymn\s+.*",
    r"Click Here to Buy it Now",
    r"ISTA FLASH DRIVE.*",
    r"The World's Wisdom.*",
    r"in the Palm.*of Your Hand",
    r"Sanskrit",
    r"Rig-Veda, Book \d+ Index",
    r"Rig Veda: Rig-Veda, Book \d+:.*"
]

# Headers to mimic a regular browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_page(url: str) -> str:
    """
    Fetch a page with retries, trying Wayback Machine first then falling back
    to the direct URL. Raises RuntimeError on failure to prevent data loss.
    """
    wayback_url = WAYBACK_PREFIX + url
    
    # Introduce randomized request delay to reduce chance of rate limiting
    delay = REQUEST_DELAY + random.uniform(1.0, 4.0)
    time.sleep(delay)
    
    urls_to_try = [wayback_url, url]  # Wayback first, then direct
    
    for attempt in range(MAX_RETRIES):
        for try_url in urls_to_try:
            try:
                resp = requests.get(try_url, headers=HEADERS, timeout=TIMEOUT)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code == 403 and try_url == url:
                    # Direct URL blocked, skip to next attempt
                    continue
                print(f"  ⚠ HTTP {resp.status_code} for {try_url} (attempt {attempt + 1})")
            except requests.RequestException as e:
                print(f"  ⚠ Request error for {try_url}: {e} (attempt {attempt + 1})")
                continue  # Try the next URL in urls_to_try
        
        if attempt < MAX_RETRIES - 1:
            # Exponential backoff on retries: 15s, 45s, 135s, 405s
            sleep_time = RETRY_DELAY * (3 ** attempt)
            print(f"  🔄 Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    raise RuntimeError(f"❌ Failed to fetch page {url} after {MAX_RETRIES} attempts.")


def parse_book_index(book_html: str) -> list[dict]:
    """Parse a book index page and extract all hymn links with their titles."""
    soup = BeautifulSoup(book_html, "html.parser")
    hymns = []
    
    for link in soup.find_all("a"):
        href = link.get("href", "")
        text = link.get_text(strip=True)
        
        # Match hymn links like rv04001.htm
        filename_match = re.search(r'(rv\d{5}\.htm)', href)
        if filename_match and "HYMN" in text.upper():
            filename = filename_match.group(1)
            hymns.append({
                "filename": filename,
                "link_text": text,
                "url": BASE_URL + filename,
            })
    
    return hymns


def clean_text(text: str) -> str:
    """Remove boilerplate, excess whitespace, and navigation artifacts."""
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove boilerplate patterns
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove page markers like "p. a1", "p. a2", "p. 115" etc.
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
    
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def extract_hymn_content(hymn_html: str, book_num: int) -> dict:
    """
    Extract structured content from a hymn page.
    Returns dict with: book, hymn_label, subtitle, verses_text
    """
    soup = BeautifulSoup(hymn_html, "html.parser")
    
    # ── Extract metadata from the <title> tag ──
    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    
    book_name = f"BOOK {ROMAN.get(book_num, str(book_num))}"
    hymn_label = ""
    subtitle = ""
    
    # Extract Hymn title (e.g., "HYMN XXXIX Dadhikrās.")
    h3 = soup.find("h3")
    if h3:
        hymn_text = h3.get_text(strip=True)
        # Separate the HYMN number from the subtitle
        match = re.match(r'(HYMN\s+[IXVLCDM]+)\s*(.*)', hymn_text, re.IGNORECASE)
        if match:
            hymn_label = match.group(1).upper()
            subtitle = match.group(2).strip(" .")
        else:
            hymn_label = hymn_text
    
    # ── Extract the actual verse content ──
    content_parts = []
    
    # Find all <p> tags that contain verse text
    for p_tag in soup.find_all("p"):
        # We use separator='\n' so that <br> tags turn into newlines, preserving the verse lines
        p_text = p_tag.get_text(separator='\n', strip=True)
        
        # Skip boilerplate
        if not p_text:
            continue
        if any(skip in p_text for skip in [
            "sacred-texts.com", "Buy this Book", "Amazon.com",
            "Sacred Texts", "Hinduism", "Next:", "Previous",
            "Griffith, [1896]", "Sanskrit", "Index"
        ]):
            continue
        if "Rig Veda, tr. by Ralph T.H. Griffith" in p_text:
            continue
            
        # Skip page markers
        if re.match(r'^p\.\s*[a-z]?\d+$', p_text):
            continue
        
        if len(p_text) > 20:  # Skip very short navigation artifacts
            content_parts.append(p_text)
    
    # ── Clean up the verse text ──
    verses_text = '\n\n'.join(content_parts)
    
    # Clean up spacing artifacts from HTML parsing
    verses_text = re.sub(r' {2,}', ' ', verses_text)
    
    return {
        "book": book_name,
        "hymn_label": hymn_label,
        "subtitle": subtitle,
        "verses_text": verses_text.strip(),
        "title_tag": title_text,
    }


def format_pretrain_record(hymn_data: dict) -> str:
    """
    Format a hymn into the continuous pretrain text format.
    Includes metadata header with book, hymn number, and title.
    """
    parts = []
    
    book_name = hymn_data.get("book", "")
    if book_name:
        parts.append(book_name)
    
    if hymn_data["hymn_label"]:
        parts.append(hymn_data["hymn_label"])
    
    if hymn_data["subtitle"]:
        parts.append(hymn_data["subtitle"])
    
    parts.append("")
    
    if hymn_data["verses_text"]:
        parts.append(hymn_data["verses_text"])
    
    return '\n'.join(parts)


def load_existing_progress(output_path: str) -> set:
    """Load already scraped hymn filenames to support resume capabilities."""
    scraped_filenames = set()
    if not os.path.exists(output_path):
        return scraped_filenames
        
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "filename" in record:
                            scraped_filenames.add(record["filename"])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"  ⚠ Could not read existing progress: {e}. Starting fresh.")
    
    if scraped_filenames:
        print(f"  🔄 Found {len(scraped_filenames)} existing records in output file. Resuming...")
    return scraped_filenames


def scrape_all_books():
    """Scrape all 10 books and their hymns progressively with resume capability."""
    print("=" * 60)
    print("  Rig Veda Scraper — sacred-texts.com via Wayback Machine")
    print("=" * 60)
    
    completed_filenames = load_existing_progress(OUTPUT_PATH)
    mode = 'a' if completed_filenames else 'w'
    new_records_count = 0
    
    with open(OUTPUT_PATH, mode, encoding='utf-8') as out_file:
        for book_num in range(1, 11):
            book_filename = f"rvi{book_num:02d}.htm"
            book_url = BASE_URL + book_filename
            
            print(f"\n📖 Fetching Book {book_num} index ({book_filename})...")
            book_html = fetch_page(book_url)
            
            hymns = parse_book_index(book_html)
            print(f"  📑 Found {len(hymns)} hymns in Book {book_num}")
            
            if not hymns:
                print(f"  ⚠ No hymns found for Book {book_num}. Skipping.")
                continue
            
            for hymn_info in tqdm(hymns, desc=f"  Book {book_num}", unit="hymn"):
                filename = hymn_info["filename"]
                
                if filename in completed_filenames:
                    continue
                
                hymn_html = fetch_page(hymn_info["url"])
                hymn_data = extract_hymn_content(hymn_html, book_num)
                
                if not hymn_data["verses_text"]:
                    print(f"    ⚠ No verse content found in {filename}")
                    continue
                
                text = format_pretrain_record(hymn_data)
                text = clean_text(text)
                
                if text:
                    record = {
                        "text": text,
                        "source": "sacred-texts.com",
                        "collection": "Rig Veda",
                        "translator": "Ralph T.H. Griffith",
                        "book": book_num,
                        "hymn": hymn_data["hymn_label"],
                        "title": hymn_data["subtitle"],
                        "filename": filename,
                    }
                    
                    out_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                    out_file.flush()
                    new_records_count += 1
                    completed_filenames.add(filename)
                    
    return new_records_count


def print_summary(output_path: str):
    """Analyze the final output file and print detailed stats and samples."""
    if not os.path.exists(output_path):
        print("\n❌ Output file does not exist.")
        return
        
    records = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
    print(f"\n✅ All runs completed. Total hymns in {output_path}: {len(records)}")
    
    if records:
        print("\n" + "─" * 60)
        print("📝 Sample record (first hymn in file):")
        print("─" * 60)
        sample = records[0]
        print(f"Text preview (first 500 chars):\n{sample['text'][:500]}")
        print("─" * 60)
        print(f"Metadata: book={sample['book']}, hymn={sample['hymn']}, title={sample['title']}")
        
        total_chars = sum(len(r["text"]) for r in records)
        print(f"\n📊 Statistics:")
        print(f"   Total hymns: {len(records)}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average chars/hymn: {total_chars // len(records):,}")
        
        book_counts = {}
        for r in records:
            b = r["book"]
            book_counts[b] = book_counts.get(b, 0) + 1
        print(f"\n   Per-book hymn counts:")
        for b in sorted(book_counts):
            print(f"     Book {b:2d}: {book_counts[b]} hymns")


def main():
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Request delay: {REQUEST_DELAY}s between pages")
    print()
    
    new_scraped = scrape_all_books()
    print(f"\n🎉 Scraped {new_scraped} new hymns in this run.")
    
    print_summary(OUTPUT_PATH)


if __name__ == "__main__":
    main()
