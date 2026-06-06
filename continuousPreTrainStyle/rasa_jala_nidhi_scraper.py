"""
Rasa Jala Nidhi (English Translation) Scraper
Scrapes all 5 volumes from wisdomlib.org chapter by chapter.
Outputs JSONL for continuous pretraining.

Usage:
    python rasa_jala_nidhi_scraper.py [--delay 2] [--output ../continuousPreTrainStyle/data/rasa_jala_nidhi_pretrain.jsonl] [--limit 2]
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
    """Extract text from HTML, skipping scripts/styles and other unwanted blocks."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "iframe", "noscript", "section"}

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
    # Match any rasa jala nidhi volume chapter doc URL
    pattern = r'<a\s+href="(/hinduism/book/rasa-jala-nidhi-volume-\d+/d/doc\d+\.html)">(.*?)</a>'
    for m in re.finditer(pattern, idx_html):
        url = "https://www.wisdomlib.org" + m.group(1)
        title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        links.append((title, url))
    return links


def clean_text(text):
    """Clean up extracted text for JSONL output."""
    text = text.replace("\xa0", " ")
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def make_record(volume_title, chapter_title, content, url):
    """Build one JSONL record."""
    # Build the text block with hierarchy header
    header = f"Rasa Jala Nidhi\n{volume_title}\n{chapter_title}"
    full_text = f"{header}\n\n{content}"

    return {
        "text": full_text,
        "source": "wisdomlib.org",
        "collection": "Rasa Jala Nidhi",
        "translator": "Rasacharya Kaviraj Bhudeb Mookerji",
        "section": volume_title,
        "chapter": chapter_title,
        "url": url,
    }


# ─── Main scraper ─────────────────────────────────────────────────────────────

def scrape_rasa(output_file, delay=2.0, limit=None):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    volumes = [
        ("Rasa Jala Nidhi, volume 1", "https://www.wisdomlib.org/hinduism/book/rasa-jala-nidhi-volume-1"),
        ("Rasa Jala Nidhi, volume 2", "https://www.wisdomlib.org/hinduism/book/rasa-jala-nidhi-volume-2"),
        ("Rasa Jala Nidhi, volume 3", "https://www.wisdomlib.org/hinduism/book/rasa-jala-nidhi-volume-3"),
        ("Rasa Jala Nidhi, volume 4", "https://www.wisdomlib.org/hinduism/book/rasa-jala-nidhi-volume-4"),
        ("Rasa Jala Nidhi, volume 5", "https://www.wisdomlib.org/hinduism/book/rasa-jala-nidhi-volume-5")
    ]

    total_records = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for vol_title, vol_url in volumes:
            print(f"\n📖 Fetching index for {vol_title}...")
            time.sleep(delay)
            try:
                index_html = fetch(vol_url)
            except Exception as e:
                print(f"❌ Failed to fetch volume index {vol_url}: {e}")
                continue

            all_links = extract_links_from_index(index_html)
            print(f"   Found {len(all_links)} entries in index.")

            # Filter out obvious non-chapter / navigation pages
            filtered_links = []
            for title, url in all_links:
                title_lower = title.lower()
                if any(skip in title_lower for skip in ["title page", "plate", "errata"]):
                    print(f"   (Skipping navigation link: {title})")
                    continue
                filtered_links.append((title, url))

            print(f"   Scraping {len(filtered_links)} chapters (limit={limit if limit else 'None'})")

            vol_count = 0
            for title, url in filtered_links:
                if limit and vol_count >= limit:
                    print(f"   Reached limit of {limit} items for this volume.")
                    break

                print(f"   🔍 Fetching: {title} ...")
                time.sleep(delay)
                try:
                    ch_html = fetch(url)
                    page_title = extract_title(ch_html)
                    content = extract_scontent(ch_html)
                    content = clean_text(content)

                    if content:
                        rec = make_record(vol_title, page_title, content, url)
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_records += 1
                        vol_count += 1
                    else:
                        # If the page was empty (e.g. section title page), skip it without counting towards limit
                        print(f"   ⚠ No content found for {title} (skipping)")
                except Exception as e:
                    print(f"   ❌ Failed to scrape {title}: {e}")

    print(f"\n🎉 Done! Wrote {total_records} records to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Rasa Jala Nidhi from wisdomlib.org → JSONL")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between requests in seconds (default: 2)")
    parser.add_argument("--output", type=str,
                        default="/Users/raj/PycharmProjects/VedaGPT/continuousPreTrainStyle/data/rasa_jala_nidhi_pretrain.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of items to scrape per volume (for testing)")
    args = parser.parse_args()

    scrape_rasa(output_file=args.output, delay=args.delay, limit=args.limit)
