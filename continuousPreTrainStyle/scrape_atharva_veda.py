"""
Atharva Veda Web Scraper
========================
Scrapes all 20 books and hymns from sacred-texts.com (via Wayback Machine)
and outputs a continuous pretrain data.jsonl file with metadata headers.

Output format: {"text": "Book I\nHYMN VI\nA charm to exercise evil spirits...\n\n1. Now may Vāchaspati..."}

Usage:
    pip install requests beautifulsoup4 tqdm
    python scrape_atharva_veda.py
"""

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
    11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV', 15: 'XV',
    16: 'XVI', 17: 'XVII', 18: 'XVIII', 19: 'XIX', 20: 'XX',
}

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL = "https://sacred-texts.com/hin/av/"
WAYBACK_PREFIX = "https://web.archive.org/web/2023/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "atharva_veda_pretrain.jsonl")
REQUEST_DELAY = 2.5        # base delay in seconds between requests
MAX_RETRIES = 5            # max retry attempts per page
RETRY_DELAY = 10           # base retry delay (will use exponential backoff)
TIMEOUT = 30               # request timeout in seconds


# Boilerplate text patterns to remove from extracted content
BOILERPLATE_PATTERNS = [
    r"Hymns of the Atharva Veda,?\s*by Ralph T\.?H\.?\s*Griffith,?\s*\[1895\],?\s*at sacred-texts\.com",
    r"Buy this Book at Amazon\.com",
    r"Sacred Texts\s+Hinduism\s+Index",
    r"Book \d+ Index",
    r"Previous\s+Next",
    r"Next:\s*Hymn\s+.*",
    r"Click Here to Buy it Now",
    r"ISTA FLASH DRIVE.*",
    r"The World's Wisdom.*",
    r"in the Palm.*of Your Hand",
]

# Headers to mimic a regular browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_page(url: str) -> str:
    """
    Fetch a page with retries, using Wayback Machine as primary source.
    Raises RuntimeError on failure to prevent data loss.
    """
    wayback_url = WAYBACK_PREFIX + url
    
    # Introduce randomized request delay to reduce chance of rate limiting
    delay = REQUEST_DELAY + random.uniform(0.5, 2.5)
    time.sleep(delay)
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(wayback_url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            print(f"  ⚠ HTTP {resp.status_code} for {wayback_url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            print(f"  ⚠ Request error for {wayback_url}: {e} (attempt {attempt + 1})")
        
        if attempt < MAX_RETRIES - 1:
            # Exponential backoff on retries: 10s, 30s, 90s, 270s, 810s
            sleep_time = RETRY_DELAY * (3 ** attempt)
            print(f"  🔄 Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    raise RuntimeError(f"❌ Failed to fetch page {url} after {MAX_RETRIES} attempts.")


def parse_book_index(book_html: str, book_num: int) -> list[dict]:
    """Parse a book index page and extract all hymn links with their titles."""
    soup = BeautifulSoup(book_html, "html.parser")
    hymns = []
    
    for link in soup.find_all("a"):
        href = link.get("href", "")
        text = link.get_text(strip=True)
        
        # Match hymn links like av08001.htm, av01035.htm
        # The href might have wayback prefix, so extract just the filename
        filename_match = re.search(r'(av\d{5}\.htm)', href)
        if filename_match and "Hymn" in text:
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


def extract_hymn_content(hymn_html: str) -> dict:
    """
    Extract structured content from a hymn page.
    Returns dict with: book, hymn_number, hymn_title, subtitle, verses_text
    """
    soup = BeautifulSoup(hymn_html, "html.parser")
    
    # ── Extract metadata from the <title> tag ──
    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    # Title format: "Atharva Veda: Book 1: Hymn 1: A prayer to Vāchaspati..."
    
    book_name = ""
    hymn_label = ""
    subtitle = ""
    
    # Extract Book heading (e.g., "BOOK I", "BOOK VIII")
    h1 = soup.find("h1")
    if h1:
        book_name = h1.get_text(strip=True)
    
    # Extract Hymn number (e.g., "HYMN I", "HYMN VI")
    h3 = soup.find("h3")
    if h3:
        hymn_label = h3.get_text(strip=True)
    
    # Extract subtitle/description (e.g., "A charm to exercise evil spirits...")
    h4_tags = soup.find_all("h4")
    for h4 in h4_tags:
        h4_text = h4.get_text(strip=True)
        # Skip the date line like "[1895-6]"
        if h4_text and not re.match(r'^\[?\d{4}', h4_text):
            subtitle = h4_text
            break
    
    # ── Extract the actual verse content ──
    # The verses are in <p> tags after the <h4> subtitle
    # Each verse starts with a <span class="margnote"> containing the verse number
    
    content_parts = []
    
    # Find all <p> tags that contain verse text
    for p_tag in soup.find_all("p"):
        p_text = ""
        
        # Check if this <p> contains verse markers (margnote spans)
        has_verses = p_tag.find("span", class_="margnote") is not None
        
        if has_verses:
            # Process verse-containing paragraph
            current_verse = []
            
            for child in p_tag.children:
                if isinstance(child, NavigableString):
                    text = str(child)
                    # Replace &nbsp; and clean
                    text = text.replace('\xa0', ' ')
                    if text.strip():
                        current_verse.append(text.strip())
                elif child.name == 'span' and 'margnote' in child.get('class', []):
                    # This is a verse number marker
                    verse_num = child.get_text(strip=True)
                    if current_verse:
                        content_parts.append(' '.join(current_verse))
                        current_verse = []
                    current_verse.append(f"{verse_num}.")
                elif child.name == 'br':
                    # Line breaks within verses — add a space
                    if current_verse:
                        last = current_verse[-1] if current_verse else ""
                        if last and not last.endswith(' '):
                            current_verse.append("")
                elif child.name == 'a' and child.get('name', '').startswith('page_'):
                    # Page marker anchor — skip
                    continue
                elif child.name == 'font' and child.get('color') == 'green':
                    # Page number in green — skip
                    continue
                elif child.name == 'a' and child.find('font', color='green'):
                    # Page number link — skip
                    continue
                else:
                    text = child.get_text()
                    text = text.replace('\xa0', ' ')
                    if text.strip():
                        # Skip the green attribution line
                        if "sacred-texts.com" in text:
                            continue
                        if "Griffith" in text and "1895" in text:
                            continue
                        current_verse.append(text.strip())
            
            if current_verse:
                content_parts.append(' '.join(current_verse))
        else:
            # Non-verse paragraph — check if it's relevant content
            p_text = p_tag.get_text(strip=True)
            
            # Skip boilerplate
            if not p_text:
                continue
            if any(skip in p_text for skip in [
                "sacred-texts.com", "Buy this Book", "Amazon.com",
                "Sacred Texts", "Hinduism", "Next:", "Previous",
                "Griffith, [1895]"
            ]):
                continue
            # Skip page markers
            if re.match(r'^p\.\s*[a-z]?\d+$', p_text):
                continue
            
            # Some hymns have prose sections without verse numbers
            if len(p_text) > 20:  # Skip very short navigation artifacts
                content_parts.append(p_text)
    
    # ── Clean up the verse text ──
    verses_text = '\n'.join(content_parts)
    
    # Clean up spacing artifacts from HTML parsing
    verses_text = re.sub(r' {2,}', ' ', verses_text)
    verses_text = re.sub(r'\n{3,}', '\n\n', verses_text)
    
    return {
        "book": book_name,
        "hymn_label": hymn_label,
        "subtitle": subtitle,
        "verses_text": verses_text.strip(),
        "title_tag": title_text,
    }


def format_pretrain_record(hymn_data: dict, book_num: int) -> str:
    """
    Format a hymn into the continuous pretrain text format.
    Includes metadata header with book, hymn number, and title.
    Always includes the book name even if the page HTML didn't have an <h1>.
    """
    parts = []
    
    # Always add book header — use parsed h1 if available, otherwise generate from book_num
    book_name = hymn_data.get("book", "").strip()
    if not book_name:
        book_name = f"BOOK {ROMAN.get(book_num, str(book_num))}"
    parts.append(book_name)
    
    # Add hymn label (e.g., "HYMN VI")
    if hymn_data["hymn_label"]:
        parts.append(hymn_data["hymn_label"])
    
    # Add subtitle/description (e.g., "A charm to exercise evil spirits who beset women")
    if hymn_data["subtitle"]:
        parts.append(hymn_data["subtitle"])
    
    # Add separator
    parts.append("")
    
    # Add the actual verse content
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
                        # We can reconstruct the unique filename from book and hymn label if needed,
                        # but storing a specific 'filename' metadata field or matching by book + hymn title is safer.
                        # Let's save the filename in the record metadata to make resume checking trivial and precise.
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
    """Scrape all 20 books and their hymns progressively with resume capability."""
    import os
    
    print("=" * 60)
    print("  Atharva Veda Scraper — sacred-texts.com via Wayback Machine")
    print("=" * 60)
    
    # Initialize/load progress
    completed_filenames = load_existing_progress(OUTPUT_PATH)
    
    # Open the file in append mode so we write progressively
    mode = 'a' if completed_filenames else 'w'
    
    new_records_count = 0
    
    with open(OUTPUT_PATH, mode, encoding='utf-8') as out_file:
        for book_num in range(1, 21):
            book_filename = f"avbook{book_num:02d}.htm"
            book_url = BASE_URL + book_filename
            
            # Check if all hymns in this book might already be done
            # (we still fetch the book index to get the list, which is fast)
            print(f"\n📖 Fetching Book {book_num} index ({book_filename})...")
            book_html = fetch_page(book_url)
            
            # Parse hymn links from book index
            hymns = parse_book_index(book_html, book_num)
            print(f"  📑 Found {len(hymns)} hymns in Book {book_num}")
            
            if not hymns:
                print(f"  ⚠ No hymns found for Book {book_num}. Skipping.")
                continue
            
            # Fetch each hymn
            for hymn_info in tqdm(hymns, desc=f"  Book {book_num}", unit="hymn"):
                filename = hymn_info["filename"]
                
                # Skip if already completed in a previous run
                if filename in completed_filenames:
                    continue
                
                hymn_html = fetch_page(hymn_info["url"])
                
                # Extract content
                hymn_data = extract_hymn_content(hymn_html)
                
                if not hymn_data["verses_text"]:
                    print(f"    ⚠ No verse content found in {filename}")
                    continue
                
                # Format the pretrain text
                text = format_pretrain_record(hymn_data, book_num)
                text = clean_text(text)
                
                if text:
                    record = {
                        "text": text,
                        "source": "sacred-texts.com",
                        "collection": "Atharva Veda",
                        "translator": "Ralph T.H. Griffith",
                        "book": book_num,
                        "hymn": hymn_data["hymn_label"],
                        "title": hymn_data["subtitle"],
                        "filename": filename,  # Kept for resume lookup
                    }
                    
                    # Write immediately and flush
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
        
        # Stats
        total_chars = sum(len(r["text"]) for r in records)
        print(f"\n📊 Statistics:")
        print(f"   Total hymns: {len(records)}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average chars/hymn: {total_chars // len(records):,}")
        
        # Per-book breakdown
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
    
    import os
    new_scraped = scrape_all_books()
    print(f"\n🎉 Scraped {new_scraped} new hymns in this run.")
    
    print_summary(OUTPUT_PATH)


if __name__ == "__main__":
    main()
