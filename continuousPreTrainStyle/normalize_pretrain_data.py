"""
Normalize continueousPreTrainData.jsonl — embed all metadata into the text field
as a structured header, producing a uniform schema for training.

Before:
  {"text": "Om...", "collection": "Atharva Veda", "book": 2, "hymn": "HYMN V", "title": "Peace"}

After:
  {"text": "[[ Collection: Atharva Veda | Book: 2 | Hymn: HYMN V | Title: Peace ]]\n\nOm..."}

Handles all 8 source schemas: Rig Veda, Atharva Veda, Sama Veda, Yajur Veda,
Charaka Samhita, Sushruta Samhita, Rasa Jala Nidhi, and IRJAY.

Usage:
    python normalize_pretrain_data.py
"""
import json
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "continueousPreTrainData.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "continueousPreTrainData.jsonl")
BACKUP_FILE = os.path.join(DATA_DIR, "continueousPreTrainData_prenormalize.jsonl")

# ─── Metadata field mappings per collection ────────────────────────────────────
# Each collection type has different metadata fields.
# We define which fields to extract (in order) for the header.

# Fields to include in the header, in display order.
# "collection" is always first. Then source-specific fields.
FIELD_DISPLAY_ORDER = {
    # Vedas from sacred-texts.com
    "Rig Veda":      ["collection", "translator", "book", "hymn", "title"],
    "Atharva Veda":  ["collection", "translator", "book", "hymn", "title"],
    "Sama Veda":     ["collection", "translator", "part", "book", "hymn", "title"],
    "Yajur Veda":    ["collection", "sub_collection", "translator", "kanda", "prapathaka", "anuvaka", "title"],

    # Medical texts from wisdomlib.org
    "Charaka Samhita":  ["collection", "translator", "section", "chapter"],
    "Sushruta Samhita": ["collection", "translator", "section", "chapter"],
    "Rasa Jala Nidhi":  ["collection", "translator", "section", "chapter"],
    "International Research Journal of Ayurveda and Yoga": ["collection", "section", "chapter"],
}

# Human-readable labels for each field key
FIELD_LABELS = {
    "collection":     "Collection",
    "sub_collection": "Sub-Collection",
    "translator":     "Translator",
    "book":           "Book",
    "hymn":           "Hymn",
    "title":          "Title",
    "part":           "Part",
    "kanda":          "Kanda",
    "prapathaka":     "Prapathaka",
    "anuvaka":        "Anuvaka",
    "section":        "Section",
    "chapter":        "Chapter",
    "source":         "Source",
    "url":            "URL",
}

# Fields to exclude from the final output (metadata we don't want in training)
EXCLUDE_FIELDS = {"filename", "url", "source"}


def build_header(record):
    """Build a structured [[ ... ]] header from the record's metadata."""
    collection = record.get("collection", "Unknown")
    
    # Get the field order for this collection, fallback to a generic order
    field_order = FIELD_DISPLAY_ORDER.get(collection, None)
    
    if field_order is None:
        # Fallback: use all non-text, non-excluded fields
        field_order = [k for k in record.keys() if k not in ("text",) | EXCLUDE_FIELDS]
    
    parts = []
    for field_key in field_order:
        value = record.get(field_key)
        if value is None or value == "":
            continue
        label = FIELD_LABELS.get(field_key, field_key.replace("_", " ").title())
        parts.append(f"{label}: {value}")
    
    if not parts:
        return ""
    
    return "[[ " + " | ".join(parts) + " ]]"


def normalize_record(record):
    """Normalize a single JSONL record:
    1. Build structured header from metadata
    2. Strip any existing inline header from the text (the old format)
    3. Combine header + clean body text
    4. Return a clean record with only {"text": ...}
    """
    raw_text = record.get("text", "")
    collection = record.get("collection", "Unknown")
    
    # ── Strip existing inline headers from the text ──
    # The old scrapers prepend headers like:
    #   "Charaka Samhita (English Translation)\nSutrasthana...\nChapter 1...\n\n"
    #   "BOOK I\nHYMN I\nAgni\n\n1 I Laud..."
    #   "Rasa Jala Nidhi\nRasa Jala Nidhi, volume 1\nPreface\n\n..."
    # We want to remove these since we're replacing them with the [[ ]] header.
    
    body = raw_text
    
    # For wisdomlib scrapers: strip the first 2-3 lines which are the inline header
    if collection in ("Charaka Samhita", "Sushruta Samhita", "Rasa Jala Nidhi",
                       "International Research Journal of Ayurveda and Yoga"):
        # These texts have: "Collection\nSection\nChapter\n\nActual content..."
        # Find the first double-newline and take everything after it
        split_pos = body.find("\n\n")
        if split_pos != -1:
            body = body[split_pos:].strip()
    
    elif collection in ("Rig Veda", "Atharva Veda"):
        # Format: "BOOK I\nHYMN I\nAgni\n\n1 I Laud..."
        split_pos = body.find("\n\n")
        if split_pos != -1:
            body = body[split_pos:].strip()
    
    elif collection == "Sama Veda":
        # Some have "PREFACE\n\nThe Samaveda..."
        # Others have "FIRST PART\nBOOK II\nCHAPTER I\n..."
        split_pos = body.find("\n\n")
        if split_pos != -1:
            body = body[split_pos:].strip()
    
    elif collection == "Yajur Veda":
        # Format: "BLACK YAJUR VEDA (TAITTIRIYA SAMHITA)\nKANDA I\nPRAPATHAKA I\nANUVAKA i. 1. 1.\nTitle\n\nContent"
        split_pos = body.find("\n\n")
        if split_pos != -1:
            body = body[split_pos:].strip()
    
    # ── Build the new text ──
    header = build_header(record)
    if header:
        new_text = f"{header}\n\n{body}"
    else:
        new_text = body
    
    return {"text": new_text}


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return
    
    # Backup
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    print(f"✅ Backup saved to {BACKUP_FILE}")
    
    # Process
    records = []
    collection_counts = {}
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                coll = record.get("collection", "Unknown")
                collection_counts[coll] = collection_counts.get(coll, 0) + 1
                normalized = normalize_record(record)
                records.append(normalized)
    
    # Write
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\n📊 Processed {len(records)} records:")
    for coll in sorted(collection_counts.keys()):
        print(f"   {coll}: {collection_counts[coll]}")
    
    print(f"\n✅ Normalized data written to {OUTPUT_FILE}")
    print(f"   Each record now has only one field: 'text'")
    
    # Preview
    print("\n--- Sample outputs ---")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            rec = json.loads(line)
            preview = rec["text"][:250]
            print(f"\nRecord {i+1}:")
            print(preview)
            print("...")


if __name__ == "__main__":
    main()
