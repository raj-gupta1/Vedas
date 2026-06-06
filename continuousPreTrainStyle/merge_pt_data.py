import os
import json
import re
import random
from collections import Counter, defaultdict

# Script base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_PRETRAIN_FILE = os.path.join(DATA_DIR, "continueousPreTrainData.jsonl")

# ── Boilerplate patterns to strip ──────────────────────────────────────────────
BOILERPLATE_PARAGRAPHS = [
    # Main boilerplate block
    r"History is littered with hundreds of conflicts over the future of a community.*?Our projects have ranged in size from a single volunteer downloading the data to a small-but-critical site, to over 100 volunteers stepping forward to acquire terabytes of user-created data to save for future generations\.",
    # archiveteam.org reference
    r"The main site for Archive Team is at\s*\n?archiveteam\.org\s*\n?and contains up to the date? information on various projects, manifestos, plans and walkthroughs\.",
    # Archive Team collection output
    r"This collection contains the output of many Archive Team projects, both ongoing and completed\. Thanks to the generous providing of disk space by the Internet Archive, multi-terabyte datasets can be made available, as well as in use by the\s*\n?Wayback Machine\s*\n?, providing a path back to lost websites and work\.",
    # Sub-collections note
    r"Our collection has grown to the point of having sub-collections for the type of data we acquire\. If you are seeking to browse the contents of these collections, the Wayback Machine is the best first stop\. Otherwise, you are free to dig into the stacks to see what you may find\.",
    # Panic Downloads
    r"The Archive Team Panic Downloads\s*\n?are full pulldowns of currently extant websites, meant to serve as emergency backups for needed sites that are in danger of closing, or which will be missed dearly if suddenly lost due to hard drive crashes or server failures\.",
    # ArchiveBot instructions
    r"To use ArchiveBot, drop by #archivebot on EFNet\. To interact with ArchiveBot, you issue commands by typing it into the channel\. Note you will need channel operator permissions in order to issue archiving jobs\. The dashboard shows the sites being downloaded currently\.",
    r"There is a dashboard running for the archivebot process at\s*\n?http://www\.archivebot\.com\s*\n?\.",
    r"ArchiveBot's source code can be found at\s*\n?https://github\.com/ArchiveTeam/ArchiveBot\s*\n?\.",
]

BOILERPLATE_PATTERN = re.compile("|".join(BOILERPLATE_PARAGRAPHS), re.DOTALL)

CONTAMINATION_KEYWORDS = [
    "Archive Team", "archiveteam.org", "ArchiveBot", "archivebot.com",
    "Wayback Machine", "Panic Downloads"
]

def clean_text(text: str) -> str:
    """Strip Archive Team boilerplate and clean up whitespaces."""
    cleaned = BOILERPLATE_PATTERN.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def has_contamination(text: str) -> bool:
    """Check if any contamination remains."""
    return any(kw in text for kw in CONTAMINATION_KEYWORDS)

def has_meaningful_content(text: str) -> bool:
    """Check if the text has valid content beyond just a header."""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    return len(lines) >= 2 and len(text) >= 100

def get_collection(text: str) -> str:
    """Extract source collection name from the formatted header."""
    if text.startswith("[["):
        header = text.split("]]")[0]
        parts = header.split("|")
        if len(parts) > 0:
            return parts[0].replace("[[ Collection:", "").strip()
    return "other"

def write_jsonl(filepath, entries):
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    print("🔗 Checking for *_pretrain.jsonl files in data directory to merge...")
    pretrain_files = sorted([
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
        if f.endswith("_pretrain.jsonl") and os.path.join(DATA_DIR, f) != RAW_PRETRAIN_FILE
    ])
    
    if not pretrain_files:
        print("⚠️ No *_pretrain.jsonl files found in data directory.")
        return
        
    print(f"🔗 Merging & Cleaning {len(pretrain_files)} files into {RAW_PRETRAIN_FILE}...")
    
    cleaned_entries = []
    seen_texts = set()
    stats = defaultdict(int)
    
    for file_path in pretrain_files:
        file_records = 0
        file_cleaned = 0
        with open(file_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                
                entry = json.loads(line)
                text = entry.get("text", "")
                
                # Clean boilerplate
                was_contaminated = has_contamination(text)
                cleaned = clean_text(text)
                
                # Aggressive backup check if regular clean missed something
                if has_contamination(cleaned):
                    paragraphs = cleaned.split("\n\n")
                    clean_paragraphs = [p for p in paragraphs if not any(kw in p for kw in CONTAMINATION_KEYWORDS)]
                    cleaned = "\n\n".join(clean_paragraphs).strip()
                    if has_contamination(cleaned):
                        stats["still_dirty_skipped"] += 1
                        continue
                
                if was_contaminated:
                    file_cleaned += 1
                    stats["total_contaminated_cleaned"] += 1
                
                # Check for empty/useless rows
                if not has_meaningful_content(cleaned):
                    stats["empty_skipped"] += 1
                    continue
                
                # Check for duplicates
                if cleaned in seen_texts:
                    stats["duplicates_removed"] += 1
                    continue
                
                seen_texts.add(cleaned)
                
                # Construct metadata header if it doesn't already start with [[ Collection:
                if not cleaned.startswith("[[ Collection:"):
                    col = entry.get("collection", "")
                    trans = entry.get("translator", "")
                    book = entry.get("book", "")
                    hymn = entry.get("hymn", "")
                    title = entry.get("title", "")
                    
                    header_parts = []
                    if col: header_parts.append(f"Collection: {col}")
                    if trans: header_parts.append(f"Translator: {trans}")
                    if book: header_parts.append(f"Book: {book}")
                    if hymn: header_parts.append(f"Hymn: {hymn}")
                    if title: header_parts.append(f"Title: {title}")
                    
                    header = "[[ " + " | ".join(header_parts) + " ]]\n\n"
                    cleaned = header + cleaned

                # Keep target format output matching continuousPreTrainData.jsonl format
                cleaned_entries.append({"text": cleaned})
                file_records += 1
                
        print(f"   📄 Processed {os.path.basename(file_path)}: kept {file_records} records (cleaned boilerplate in {file_cleaned})")
    
    # Save the full merged & cleaned dataset
    write_jsonl(RAW_PRETRAIN_FILE, cleaned_entries)
    print(f"✅ Saved total of {len(cleaned_entries)} cleaned records to: {RAW_PRETRAIN_FILE}")
    
    if stats["duplicates_removed"] > 0 or stats["empty_skipped"] > 0:
        print(f"      Deduplicated: {stats['duplicates_removed']} records")
        print(f"      Skipped empty/invalid: {stats['empty_skipped']} records")

if __name__ == "__main__":
    main()
