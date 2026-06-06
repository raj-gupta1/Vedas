"""
Charaka Samhita (English Translation) Scraper
Scrapes from wisdomlib.org chapter by chapter.
Outputs JSONL for continuous pretraining.

Usage:
    python charaka_scraper.py [--delay 2] [--output ./charaka_samhita_pretrain.jsonl]
"""

import argparse
import json
import os
import re
import time
import urllib.request
from html.parser import HTMLParser


# ─── Minimal HTML→text extractor ──────────────────────────────────────────────

class ContentExtractor(HTMLParser):
    """Extract text from HTML, skipping scripts/styles."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "iframe", "noscript"}

    def __init__(self):
        super().__init__()
        self.result = []
        self._skip_depth = 0
        self._in_h = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_h = True
            self.result.append("\n\n")
        elif tag == "p":
            self.result.append("\n\n")
        elif tag == "br":
            self.result.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_h = False

    def handle_data(self, data):
        if self._skip_depth:
            return
        self.result.append(data)

    def get_text(self):
        return "".join(self.result).strip()


# ─── HTML helpers ─────────────────────────────────────────────────────────────

def fetch(url, retries=3, delay=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  ⚠ Attempt {attempt+1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def extract_between(html, start_marker, end_marker):
    i = html.find(start_marker)
    if i == -1:
        return ""
    i += len(start_marker)
    j = html.find(end_marker, i)
    if j == -1:
        return html[i:]
    return html[i:j]


def extract_scontent(html):
    """Extract text content from the #scontent div."""
    marker = 'id="scontent"'
    idx = html.find(marker)
    if idx == -1:
        return ""
    start = html.find(">", idx) + 1
    depth = 1
    pos = start
    end = len(html)
    while depth > 0 and pos < len(html):
        next_open = html.find("<div", pos)
        next_close = html.find("</div>", pos)
        if next_close == -1:
            break
        if next_open != -1 and next_open < next_close:
            depth += 1
            pos = next_open + 4
        else:
            depth -= 1
            if depth == 0:
                end = next_close
            pos = next_close + 6

    content_html = html[start:end]
    extractor = ContentExtractor()
    extractor.feed(content_html)
    return extractor.get_text()


def extract_title(html):
    m = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
    if m:
        return re.sub(r'<[^>]+>', '', m.group(1)).strip()
    m = re.search(r'<title>(.*?)</title>', html)
    if m:
        return m.group(1).strip()
    return "Untitled"


def extract_links_from_index(html):
    idx_html = extract_between(html, 'id="indexList"', '</section>')
    links = []
    for m in re.finditer(
        r'<a\s+href="(/hinduism/book/charaka-samhita-english/d/doc\d+\.html)">(.*?)</a>',
        idx_html
    ):
        url = "https://www.wisdomlib.org" + m.group(1)
        title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        links.append((title, url))
    return links


def extract_child_links(html):
    idx_html = extract_between(html, 'id="indexList"', '</section>')
    if not idx_html:
        return []
    links = []
    for m in re.finditer(
        r'<a\s+href="(/hinduism/book/charaka-samhita-english/d/doc\d+\.html)">(.*?)</a>',
        idx_html
    ):
        url = "https://www.wisdomlib.org" + m.group(1)
        title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        links.append((title, url))
    return links


def is_section_page(html):
    has_index = 'id="indexList"' in html
    marker = 'id="scontent"'
    idx = html.find(marker)
    if idx == -1:
        return has_index
    start = html.find(">", idx) + 1
    close = html.find("</div>", start)
    content = html[start:close].strip() if close != -1 else ""
    return has_index and len(content) < 10


def clean_text(text):
    """Clean up extracted text for JSONL output."""
    text = text.replace("\xa0", " ")
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def make_record(section_title, chapter_title, content, url):
    """Build one JSONL record."""
    # Build the text block with hierarchy header
    header = f"Charaka Samhita (English Translation)\n{section_title}\n{chapter_title}"
    full_text = f"{header}\n\n{content}"

    return {
        "text": full_text,
        "source": "wisdomlib.org",
        "collection": "Charaka Samhita",
        "translator": "Shree Gulabkunverba Ayurvedic Society",
        "section": section_title,
        "chapter": chapter_title,
        "url": url,
    }


# ─── Main scraper ─────────────────────────────────────────────────────────────

def scrape_book(output_file="./charaka_samhita_pretrain.jsonl", delay=2.0):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    book_url = "https://www.wisdomlib.org/hinduism/book/charaka-samhita-english"
    print(f"📖 Fetching book index: {book_url}")
    book_html = fetch(book_url)

    all_links = extract_links_from_index(book_html)
    print(f"   Found {len(all_links)} entries in the table of contents\n")

    total_records = 0

    with open(output_file, "w", encoding="utf-8") as out:
        i = 0
        while i < len(all_links):
            title, url = all_links[i]
            print(f"🔍 Checking: {title}")
            time.sleep(delay)
            html = fetch(url)

            if is_section_page(html):
                section_title = title
                child_links = extract_child_links(html)
                print(f"📁 Section: {section_title} ({len(child_links)} chapters)")

                for ch_title, ch_url in child_links:
                    print(f"   📄 {ch_title}...")
                    time.sleep(delay)
                    ch_html = fetch(ch_url)
                    page_title = extract_title(ch_html)
                    content = extract_scontent(ch_html)
                    content = clean_text(content)

                    if not content:
                        # Sub-section with its own children
                        sub_children = extract_child_links(ch_html)
                        if sub_children:
                            for sub_title, sub_url in sub_children:
                                print(f"      📄 {sub_title}...")
                                time.sleep(delay)
                                sub_html = fetch(sub_url)
                                sub_page_title = extract_title(sub_html)
                                sub_content = extract_scontent(sub_html)
                                sub_content = clean_text(sub_content)
                                if sub_content:
                                    rec = make_record(section_title, sub_page_title, sub_content, sub_url)
                                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                    total_records += 1
                        continue

                    rec = make_record(section_title, page_title, content, ch_url)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_records += 1

                # Skip child entries in the flat list
                child_urls = {u for _, u in child_links}
                i += 1
                while i < len(all_links) and all_links[i][1] in child_urls:
                    i += 1
                print()
            else:
                # Standalone chapter
                page_title = extract_title(html)
                content = extract_scontent(html)
                content = clean_text(content)
                if content:
                    rec = make_record(page_title, page_title, content, url)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_records += 1
                i += 1

    print(f"\n🎉 Done! Wrote {total_records} records to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Charaka Samhita from wisdomlib.org → JSONL")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between requests in seconds (default: 2)")
    parser.add_argument("--output", type=str,
                        default="/Users/raj/PycharmProjects/VedaGPT/continuousPreTrainStyle/data/charaka_samhita_pretrain.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()

    scrape_book(output_file=args.output, delay=args.delay)
