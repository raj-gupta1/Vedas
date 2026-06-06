"""
International Research Journal of Ayurveda and Yoga (IRJAY) Scraper
Scrapes all volumes/articles from wisdomlib.org.
Outputs JSONL for continuous pretraining.

Usage:
    python irjay_scraper.py [--delay 2] [--output ./data/irjay_pretrain.jsonl] [--limit 3]
"""

import argparse
import json
import os
import re
import time
import urllib.request
from html.parser import HTMLParser


# ─── HTML→text extractor ──────────────────────────────────────────────────────

class ContentExtractor(HTMLParser):
    """Extract text from HTML, skipping scripts/styles and footnotes section."""

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

def fetch(url, retries=3, delay=2, cookie=None, user_agent=None):
    headers = {
        "User-Agent": user_agent or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if cookie:
        headers["Cookie"] = cookie
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
    if not idx_html:
        return []
    links = []
    # Match any journal chapter/volume doc URL
    pattern = r'<a\s+href="(/hinduism/journal/international-research-journal-of-ayurveda-and-yoga/d/doc\d+\.html)">(.*?)</a>'
    for m in re.finditer(pattern, idx_html):
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
    return has_index and len(content) < 50


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
    header = f"International Research Journal of Ayurveda and Yoga\n{section_title}\n{chapter_title}"
    full_text = f"{header}\n\n{content}"

    return {
        "text": full_text,
        "source": "wisdomlib.org",
        "collection": "International Research Journal of Ayurveda and Yoga",
        "translator": "",
        "section": section_title,
        "chapter": chapter_title,
        "url": url,
    }


# ─── Main scraper ─────────────────────────────────────────────────────────────

def scrape_journal(output_file, delay=2.0, limit=None, cookie=None, user_agent=None):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    root_url = "https://www.wisdomlib.org/hinduism/journal/international-research-journal-of-ayurveda-and-yoga"
    print(f"📖 Fetching main journal page: {root_url}")
    try:
        root_html = fetch(root_url, cookie=cookie, user_agent=user_agent)
    except Exception as e:
        print(f"❌ Failed to fetch main journal page: {e}")
        return

    # Extract first-level links (volumes)
    vol_links = extract_links_from_index(root_html)
    print(f"   Found {len(vol_links)} top-level index entries.")

    # Queue of elements to crawl: (parent_section_title, url, depth)
    queue = []
    for title, url in vol_links:
        # Avoid navigation/extra links if any
        title_lower = title.lower()
        if any(skip in title_lower for skip in ["title page", "plate", "errata", "previous", "next"]):
            continue
        queue.append((title, url, 1))

    visited = set()
    total_records = 0

    with open(output_file, "w", encoding="utf-8") as out:
        while queue:
            if limit and total_records >= limit:
                print(f"\n✋ Reached global limit of {limit} records. Stopping crawl.")
                break

            parent_title, url, depth = queue.pop(0)

            if url in visited:
                continue
            visited.add(url)

            print(f"\n🔍 Fetching (depth {depth}): {parent_title} -> {url}")
            time.sleep(delay)

            try:
                html = fetch(url, cookie=cookie, user_agent=user_agent)
            except Exception as e:
                print(f"  ❌ Failed to fetch {url}: {e}")
                continue

            if is_section_page(html):
                # This is a container (like a volume or issue index page)
                sub_links = extract_links_from_index(html)
                print(f"  📁 Section page: found {len(sub_links)} sub-links.")
                current_title = extract_title(html)

                # Format hierarchy name (e.g. combining parent + current if appropriate)
                combined_title = parent_title
                if current_title and current_title != "Untitled" and current_title != parent_title:
                    combined_title = f"{parent_title} - {current_title}"

                # Queue the sub-links
                added_count = 0
                for sub_title, sub_url in sub_links:
                    sub_title_lower = sub_title.lower()
                    if any(skip in sub_title_lower for skip in ["title page", "plate", "errata", "previous", "next"]):
                        continue
                    if sub_url not in visited:
                        # Push to front of queue to do DFS-like traversal of volume/issue
                        queue.insert(added_count, (combined_title, sub_url, depth + 1))
                        added_count += 1
                print(f"  ➕ Queued {added_count} new links.")
            else:
                # This is an actual article page with content
                page_title = extract_title(html)
                content = extract_scontent(html)
                content = clean_text(content)

                if content:
                    rec = make_record(parent_title, page_title, content, url)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_records += 1
                    print(f"  ✅ Saved article: {page_title} (Total: {total_records})")
                else:
                    print(f"  ⚠ Empty page or no article content found at {url}")

    print(f"\n🎉 Done! Wrote {total_records} records to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape IRJAY journal from wisdomlib.org → JSONL")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between requests in seconds (default: 2)")
    parser.add_argument("--output", type=str,
                        default="/Users/raj/PycharmProjects/VedaGPT/continuousPreTrainStyle/data/irjay_pretrain.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of articles to scrape (for testing)")
    parser.add_argument("--cookie", type=str, default=None,
                        help="Optional Cookie header value to bypass Cloudflare challenge (copy from browser)")
    parser.add_argument("--user-agent", type=str, default=None,
                        help="Optional User-Agent matching the browser session that solved Cloudflare challenge")
    args = parser.parse_args()

    scrape_journal(
        output_file=args.output,
        delay=args.delay,
        limit=args.limit,
        cookie=args.cookie,
        user_agent=args.user_agent
    )
