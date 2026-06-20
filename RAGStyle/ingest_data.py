import os
import json
import re
import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from colorama import Fore, init
from tqdm import tqdm

init(autoreset=True)
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI or "xxxxx" in MONGODB_URI:
    print(f"{Fore.RED}Error: Please set your MONGODB_URI in the .env file!{Fore.RESET}")
    exit(1)

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "continuousPreTrainStyle", "data", "continueousPreTrainData.jsonl")
DB_NAME = "vedic_rag"
COLLECTION_NAME = "scriptures"

# ── Upgraded Embedding Model ──
# BGE (BAAI General Embedding) is significantly better than MiniLM for semantic search.
# It outputs 768-dimensional vectors and understands complex queries much better.
# The "Represent this sentence:" prefix is recommended by the model authors for queries.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

print(f"🚀 {Fore.LIGHTCYAN_EX}Starting Enhanced Vector Data Ingestion...{Fore.RESET}")

# 1. Connect to MongoDB with Fallback
db = None
collection = None
is_local = False

try:
    print(f"🔌 Connecting to MongoDB configured URI: {MONGODB_URI}...")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Trigger connection test
    client.server_info()
    print("✅ Connected to Atlas MongoDB successfully!")
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
except Exception as e:
    print(f"⚠️  Failed to connect to configured MongoDB: {e}")
    print("🔄 Falling back to local MongoDB (mongodb://localhost:27017/)...")
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        client.server_info()
        print("✅ Connected to local MongoDB successfully!")
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        is_local = True
    except Exception as local_e:
        print(f"❌ Failed to connect to local MongoDB: {local_e}")
        exit(1)

# Drop old data from the previous embedding run
try:
    existing_count = collection.count_documents({})
    if existing_count > 0:
        print(f"⚠️  Found {existing_count} existing documents in {COLLECTION_NAME}. Dropping old collection for re-ingestion...")
        collection.drop()
        # Reinitialize collection object after drop
        collection = db[COLLECTION_NAME]
except Exception as e:
    print(f"❌ Error operating on collection: {e}")
    exit(1)

# 2. Load the Embedding Model
print(f"🧠 Loading Embedding Model: {EMBEDDING_MODEL}...")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

# 3. Load Data
if not os.path.exists(DATA_PATH):
    print(f"{Fore.RED}Error: Data file not found at {DATA_PATH}{Fore.RESET}")
    exit(1)

print(f"📖 Loading dataset from {DATA_PATH}...")


def parse_header(text):
    """
    Parse the [[ Collection: X | Book: Y | ... ]] header from the new data format.
    Returns (metadata_dict, body_text).
    """
    header_pattern = re.compile(r'^\[\[\s*(.*?)\s*\]\]\s*', re.DOTALL)
    match = header_pattern.match(text)

    if not match:
        return {}, text

    header_str = match.group(1)
    body = text[match.end():].strip()

    metadata = {}
    for field in header_str.split("|"):
        field = field.strip()
        if ":" in field:
            key, value = field.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            metadata[key] = value

    return metadata, body


raw_data = []
with open(DATA_PATH, 'r', encoding="utf-8") as f:
    for line in f:
        if line.strip():
            raw_data.append(json.loads(line))

# Parse headers and normalize into structured records
data = []
for entry in raw_data:
    text = entry.get("text", "")
    # Check if this record already has structured fields (old format)
    if entry.get("collection"):
        data.append(entry)
    else:
        # New format: parse the [[ ... ]] header
        metadata, body = parse_header(text)
        data.append({
            "text": body,
            "collection": metadata.get("collection", "Unknown"),
            "book": metadata.get("book", ""),
            "hymn": metadata.get("hymn", ""),
            "title": metadata.get("title", ""),
            "translator": metadata.get("translator", ""),
            "source": metadata.get("source", ""),
        })

print(f"Found {len(data)} records.")


# ── Smart Hymn-Aware Chunking ──
# Most hymns are ~78 words (median). We preserve whole hymns when possible.
# For longer texts (>250 words), we split on verse boundaries (numbered lines like "1.", "2.").
# This prevents cutting a verse in half, which destroys semantic meaning for the embedder.

def split_on_verse_boundaries(text, max_words=200, overlap_words=30):
    """Split text on verse boundaries (e.g., '1. ...', '2. ...') with overlap."""
    # Try to split on verse numbers like "1.", "2.", etc.
    verse_pattern = re.compile(r'\n(?=\d+[\.\)])')
    verses = verse_pattern.split(text)
    
    if len(verses) <= 1:
        # No verse structure found, fall back to sentence-level splitting
        return split_by_sentences(text, max_words, overlap_words)
    
    chunks = []
    current_chunk_verses = []
    current_word_count = 0
    
    for verse in verses:
        verse = verse.strip()
        if not verse:
            continue
        verse_words = len(verse.split())
        
        if current_word_count + verse_words > max_words and current_chunk_verses:
            chunks.append("\n".join(current_chunk_verses))
            # Overlap: keep last verse(s) that fit within overlap budget
            overlap_verses = []
            overlap_count = 0
            for v in reversed(current_chunk_verses):
                vw = len(v.split())
                if overlap_count + vw <= overlap_words:
                    overlap_verses.insert(0, v)
                    overlap_count += vw
                else:
                    break
            current_chunk_verses = overlap_verses
            current_word_count = overlap_count
        
        current_chunk_verses.append(verse)
        current_word_count += verse_words
    
    if current_chunk_verses:
        chunks.append("\n".join(current_chunk_verses))
    
    return chunks


def split_by_sentences(text, max_words=200, overlap_words=30):
    """Fallback: split on sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_count = 0
    
    for sent in sentences:
        sent_words = len(sent.split())
        if current_count + sent_words > max_words and current:
            chunks.append(" ".join(current))
            # Overlap
            overlap = []
            oc = 0
            for s in reversed(current):
                sw = len(s.split())
                if oc + sw <= overlap_words:
                    overlap.insert(0, s)
                    oc += sw
                else:
                    break
            current = overlap
            current_count = oc
        current.append(sent)
        current_count += sent_words
    
    if current:
        chunks.append(" ".join(current))
    return chunks


def smart_chunk(entry):
    """Hymn-aware chunking: keep short hymns whole, split long ones on verse boundaries."""
    text = entry.get("text", "").strip()
    if not text:
        return []
    
    words = text.split()
    word_count = len(words)
    
    # Short hymns (≤250 words): keep whole — this is the majority of data
    if word_count <= 250:
        return [text]
    
    # Longer texts: split on verse boundaries
    return split_on_verse_boundaries(text, max_words=200, overlap_words=30)


# ── Build metadata-enriched context prefix ──
def build_metadata_prefix(entry):
    """Creates a rich metadata string that gets prepended to help the embedder understand context."""
    parts = []
    if entry.get("collection"):
        parts.append(entry["collection"])
    if entry.get("book"):
        parts.append(f"Book {entry['book']}")
    if entry.get("hymn"):
        parts.append(entry["hymn"])
    if entry.get("title"):
        parts.append(f"— {entry['title']}")
    return " | ".join(parts)


documents_to_insert = []
texts_to_embed = []

print(f"✂️  Smart chunking and preparing texts...")
for idx, entry in enumerate(tqdm(data, desc="Chunking records")):
    chunks = smart_chunk(entry)
    metadata_prefix = build_metadata_prefix(entry)
    doc_id = f"doc_{idx}"
    
    for i, chunk in enumerate(chunks):
        text_for_embedding = f"{metadata_prefix}: {chunk}" if metadata_prefix else chunk
        texts_to_embed.append(text_for_embedding)
        
        doc = {
            "doc_id": doc_id,
            "text": chunk,                                       # Original text (for display)
            "text_with_context": text_for_embedding,             # Metadata-enriched (for search quality)
            "collection": entry.get("collection", "Unknown"),
            "book": entry.get("book", ""),
            "hymn": entry.get("hymn", ""),
            "title": entry.get("title", ""),
            "translator": entry.get("translator", ""),
            "source": entry.get("source", ""),
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        documents_to_insert.append(doc)

print(f"\n🧠 Generating embeddings in batch mode for {len(texts_to_embed)} chunks...")
embeddings = embedder.encode(texts_to_embed, batch_size=128, show_progress_bar=True)

print("💾 Assigning embeddings to documents...")
for doc, emb in zip(documents_to_insert, embeddings):
    doc["embedding"] = emb.tolist()

print(f"\n📊 Total chunks created: {len(documents_to_insert)} (from {len(data)} records)")

# 4. Insert into MongoDB
if documents_to_insert:
    print(f"💾 Uploading {len(documents_to_insert)} chunks to MongoDB Atlas...")
    
    batch_size = 500
    for i in tqdm(range(0, len(documents_to_insert), batch_size), desc="Uploading to MongoDB"):
        batch = documents_to_insert[i:i + batch_size]
        collection.insert_many(batch)
        
    print(f"\n{Fore.LIGHTGREEN_EX}✅ Successfully inserted {len(documents_to_insert)} vector chunks into MongoDB.{Fore.RESET}")
    print(f"\n{Fore.LIGHTYELLOW_EX}⚠️  IMPORTANT: Update your MongoDB Atlas Vector Search Index!{Fore.RESET}")
    print(f"   Your index 'vedic_index' must now use {Fore.LIGHTCYAN_EX}numDimensions: {EMBEDDING_DIM}{Fore.RESET}")
    print(f"   Go to Atlas → Database → Atlas Search → Edit Index → change numDimensions to {EMBEDDING_DIM}")
else:
    print(f"{Fore.YELLOW}No documents to insert.{Fore.RESET}")
