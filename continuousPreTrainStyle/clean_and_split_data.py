"""
Clean the continuousPreTrainData.jsonl file:
1. Remove Archive Team / archiveteam.org / ArchiveBot boilerplate contamination
2. Remove exact duplicates
3. Validate cleaned entries have meaningful content
4. Split into train/val/test (80/10/10) with stratification by source collection
"""

import json
import re
import os
import random
from collections import Counter, defaultdict

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "continueousPreTrainData.jsonl")
OUTPUT_DIR = DATA_DIR

# ── Boilerplate patterns to strip ──────────────────────────────────────────────
# These are complete paragraphs that appear between the hymn header and verse content
BOILERPLATE_PARAGRAPHS = [
    # Main boilerplate block (always starts with this)
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

# Compile a combined pattern
BOILERPLATE_PATTERN = re.compile(
    "|".join(BOILERPLATE_PARAGRAPHS),
    re.DOTALL
)

# Keyword check for any remaining contamination
CONTAMINATION_KEYWORDS = [
    "Archive Team",
    "archiveteam.org",
    "ArchiveBot",
    "archivebot.com",
    "Wayback Machine",
    "Panic Downloads",
    "Internet Archive, multi-terabyte",
    "sub-collections for the type of data",
]


def get_collection(text: str) -> str:
    """Extract the collection name from the structured header."""
    if text.startswith("[["):
        header = text.split("]]")[0]
        parts = header.split("|")
        if len(parts) > 0:
            return parts[0].replace("[[ Collection:", "").strip()
    return "other"


def clean_text(text: str) -> str:
    """Remove Archive Team boilerplate from a text entry."""
    cleaned = BOILERPLATE_PATTERN.sub("", text)
    
    # Clean up excessive whitespace left by removal (multiple blank lines → double newline)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def has_contamination(text: str) -> bool:
    """Check if text still contains any contamination keywords."""
    return any(kw in text for kw in CONTAMINATION_KEYWORDS)


def has_meaningful_content(text: str) -> bool:
    """Check if the text has meaningful hymn/verse/medical content beyond just a header."""
    # Must have at least some body text (not just a header line)
    lines = text.strip().split("\n")
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 2:
        return False
    # Must have at least 100 chars of actual content
    if len(text) < 100:
        return False
    return True


def stratified_split(entries_by_collection, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split entries into train/val/test with stratification by collection source."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    
    train, val, test = [], [], []
    
    for collection, entries in entries_by_collection.items():
        random.shuffle(entries)
        n = len(entries)
        n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        n_test = max(1, round(n * test_ratio)) if n >= 3 else 0
        n_train = n - n_val - n_test
        
        train.extend(entries[:n_train])
        val.extend(entries[n_train:n_train + n_val])
        test.extend(entries[n_train + n_val:])
    
    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def write_jsonl(filepath, entries):
    """Write entries to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    print("=" * 70)
    print("VedaGPT Data Cleaning & Splitting Pipeline")
    print("=" * 70)
    
    # ── Load ───────────────────────────────────────────────────────────────
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    
    print(f"\n📥 Loaded {len(raw_lines)} entries from source file")
    
    # ── Clean ──────────────────────────────────────────────────────────────
    cleaned_entries = []
    stats = {
        "total": len(raw_lines),
        "contaminated_cleaned": 0,
        "contaminated_still_dirty": 0,
        "empty_after_clean": 0,
        "exact_duplicates_removed": 0,
    }
    
    seen_texts = set()
    still_dirty = []
    
    for line in raw_lines:
        entry = json.loads(line)
        text = entry["text"]
        
        # Step 1: Clean boilerplate
        was_contaminated = has_contamination(text)
        cleaned = clean_text(text)
        
        # Step 2: Check if cleaning was successful
        if has_contamination(cleaned):
            stats["contaminated_still_dirty"] += 1
            still_dirty.append(cleaned[:200])
            # Still try to salvage - do a more aggressive clean
            # Remove any paragraph containing contamination keywords
            paragraphs = cleaned.split("\n\n")
            clean_paragraphs = []
            for p in paragraphs:
                if not any(kw in p for kw in CONTAMINATION_KEYWORDS):
                    clean_paragraphs.append(p)
            cleaned = "\n\n".join(clean_paragraphs).strip()
            
            # If still dirty after aggressive clean, skip
            if has_contamination(cleaned):
                continue
        
        if was_contaminated:
            stats["contaminated_cleaned"] += 1
        
        # Step 3: Check meaningful content
        if not has_meaningful_content(cleaned):
            stats["empty_after_clean"] += 1
            continue
        
        # Step 4: Deduplicate
        if cleaned in seen_texts:
            stats["exact_duplicates_removed"] += 1
            continue
        seen_texts.add(cleaned)
        
        cleaned_entries.append({"text": cleaned})
    
    print(f"\n🧹 Cleaning Results:")
    print(f"   Contaminated entries cleaned:    {stats['contaminated_cleaned']}")
    print(f"   Still dirty (aggressive clean):  {stats['contaminated_still_dirty']}")
    print(f"   Empty after cleaning:            {stats['empty_after_clean']}")
    print(f"   Exact duplicates removed:        {stats['exact_duplicates_removed']}")
    print(f"   Clean entries retained:          {len(cleaned_entries)}")
    
    # ── Analyze cleaned data ───────────────────────────────────────────────
    entries_by_collection = defaultdict(list)
    for entry in cleaned_entries:
        collection = get_collection(entry["text"])
        entries_by_collection[collection].append(entry)
    
    print(f"\n📊 Cleaned Data by Source:")
    for collection, entries in sorted(entries_by_collection.items(), key=lambda x: -len(x[1])):
        print(f"   {collection}: {len(entries)}")
    
    # ── Split ──────────────────────────────────────────────────────────────
    train, val, test = stratified_split(entries_by_collection)
    
    print(f"\n✂️  Split Results (80/10/10):")
    print(f"   Train: {len(train)} ({100*len(train)/len(cleaned_entries):.1f}%)")
    print(f"   Val:   {len(val)} ({100*len(val)/len(cleaned_entries):.1f}%)")
    print(f"   Test:  {len(test)} ({100*len(test)/len(cleaned_entries):.1f}%)")
    
    # ── Verify split distributions ─────────────────────────────────────────
    print(f"\n📊 Split Distribution by Source:")
    for split_name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
        dist = Counter(get_collection(e["text"]) for e in split_data)
        print(f"   {split_name}:")
        for collection, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"      {collection}: {count}")
    
    # ── Write output ───────────────────────────────────────────────────────
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_path = os.path.join(OUTPUT_DIR, "val.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test.jsonl")
    cleaned_full_path = os.path.join(OUTPUT_DIR, "continueousPreTrainData_cleaned.jsonl")
    
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)
    write_jsonl(cleaned_full_path, cleaned_entries)
    
    print(f"\n💾 Files Written:")
    print(f"   {train_path} ({len(train)} entries)")
    print(f"   {val_path} ({len(val)} entries)")
    print(f"   {test_path} ({len(test)} entries)")
    print(f"   {cleaned_full_path} ({len(cleaned_entries)} entries, full cleaned dataset)")
    
    # ── Spot check: show a before/after sample ─────────────────────────────
    print(f"\n🔍 Spot Check (before → after cleaning):")
    # Find a contaminated entry to show
    for line in raw_lines:
        entry = json.loads(line)
        if "Archive Team" in entry["text"] and "Rig Veda" in entry["text"]:
            print(f"\n   BEFORE (first 300 chars):")
            print(f"   {entry['text'][:300]}...")
            cleaned = clean_text(entry["text"])
            print(f"\n   AFTER (first 300 chars):")
            print(f"   {cleaned[:300]}...")
            break
    
    print(f"\n{'=' * 70}")
    print(f"✅ Pipeline complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
