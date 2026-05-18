import json
import os
import hashlib
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm
from pypdf import PdfReader
from distillery.config import load_settings
from distillery.pipeline import PipelineConfig, run_pipeline
from distillery.providers.embeddings import build_embedder
from distillery.providers.llm import build_provider
from distillery.export.jsonl import export_jsonl
from distillery.export.split import train_eval_split
from distillery.types import Chunk, Example

# Configuration for this run
START_PAGE = 520
END_PAGE = 530

# Page-wise chunking with overlap
def extract_pages(pdf_path: str, start: int, end: int) -> list[tuple[int, str]]:
    """Extract text from specified pages of the PDF, returning (page_num, text)."""
    reader = PdfReader(pdf_path)
    pages = []
    # Ensure we don't go out of bounds
    end = min(end, len(reader.pages))
    for page_num in tqdm(range(start, end), desc="Extracting pages", unit="page"):
        text = (reader.pages[page_num].extract_text() or "").strip()
        if text:
            pages.append((page_num, text))
    return pages

def chunk_pages(pages: list[tuple[int, str]], source: str) -> list[Chunk]:
    """
    Split each page into 3 roughly equal chunks.
    """
    if not pages:
        return []

    chunks = []
    idx = 0

    for page_num, text in pages:
        text = text.strip()
        if not text:
            continue
        
        # Split page text into 3 roughly equal parts
        part_size = max(1, len(text) // 3)
        parts = [text[i:i+part_size] for i in range(0, len(text), part_size)]
        
        # In case integer division gives more than 3 parts (e.g., remaining characters at the end),
        # merge the last few parts into the 3rd part
        if len(parts) > 3:
            parts[2] = "".join(parts[2:])
            parts = parts[:3]

        for part_idx, part_text in enumerate(parts):
            part_text = part_text.strip()
            if not part_text:
                continue
                
            digest = hashlib.sha1(part_text.encode("utf-8")).hexdigest()[:10]
            chunk_id = f"{source}:p{page_num}_part{part_idx+1}:{digest}"

            chunks.append(Chunk(
                id=chunk_id,
                text=part_text,
                source=source,
                index=idx,
                metadata={
                    "page_number": page_num,
                    "part_number": part_idx + 1,
                    "char_count": len(part_text),
                },
            ))
            idx += 1

    return chunks

# Checkpoint helpers

def save_chunks_checkpoint(chunks: list[Chunk], path="chunks_checkpoint.json"):
    """Save chunks to JSON file."""
    serializable = [asdict(c) for c in chunks]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Checkpoint saved: {path}")

def load_chunks_checkpoint(path="chunks_checkpoint.json") -> list[Chunk] | None:
    """Load chunks from JSON file if exists."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = [Chunk(**d) for d in data]
        print(f"Loaded {len(chunks)} chunks from checkpoint {path}")
        return chunks
    return None

def save_pipeline_checkpoint(examples, path="pipeline_checkpoint.jsonl"):
    """Save generated examples incrementally (overwrites)."""
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + '\n')
    print(f"Pipeline checkpoint saved: {path}")

def load_pipeline_checkpoint(path="pipeline_checkpoint.jsonl") -> list[Example] | None:
    """Load previously generated examples, filtering out unknown fields."""
    if os.path.exists(path):
        examples = []
        valid_fields = {f.name for f in Example.__dataclass_fields__.values()}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Skip header lines (distillery checkpoints start with a config line)
                if "instruction" not in data:
                    continue
                # Only pass keys that exist in the Example dataclass
                filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                examples.append(Example(**filtered_data))
        print(f"Loaded {len(examples)} examples from checkpoint {path}")
        return examples
    return None

# Main pipeline

def main():
    # Load settings and build provider/embedder
    settings = load_settings()
    provider = build_provider(settings)
    embedder = build_embedder(settings.embedding_model)

    # Point to the rootbooks folder relative to qaAlpacaStyle/
    pdf_path = "../books/atharvaVedHindiSanskrit.pdf"

    # Step 1: Page-wise chunking with overlap
    print(f"Loading PDF from {pdf_path} (Pages {START_PAGE} to {END_PAGE})...")
    pages = extract_pages(pdf_path, START_PAGE, END_PAGE)
    print(f"Extracted {len(pages)} non-empty pages")

    print("Creating chunks (3 chunks per page)...")
    chunks = chunk_pages(
        pages,
        source=pdf_path
    )
    print(f"Created {len(chunks)} chunks")

    # Show first chunk sample to verify
    if chunks:
        sample = chunks[0].text[:300]
        print(f"\n--- First chunk sample (page {chunks[0].metadata['page_number']}, part {chunks[0].metadata['part_number']}) ---")
        print(sample)
        print("--------------------------\n")

    # Step 2: Run Distillery pipeline
    checkpoint_name = f"pipeline_checkpoint_{START_PAGE}_{END_PAGE}.jsonl"
    existing_examples = load_pipeline_checkpoint(checkpoint_name)
    if existing_examples:
        print(f"Found existing pipeline checkpoint {checkpoint_name}. Skipping generation.")
        result_examples = existing_examples
    else:
        print("Starting Distillery pipeline (this may take hours)...")
        result = run_pipeline(
            config=PipelineConfig(
                description="""You are a scholarly assistant specializing in Sanskrit and Hindi texts. 
                Answer questions based STRICTLY on the provided book content which is organized into Kandas (sections) and Sutras (aphorisms).
                
                Guidelines:
                - When referencing answers, mention the specific Kanda and Sutra number as context
                - Preserve Sanskrit diacritics and special characters accurately
                - For Sanskrit text, maintain grammatical cases (vibhakti) and sandhi combinations
                - If the query is in Sanskrit, respond in Sanskrit; if in Hindi, respond in Hindi
                - For technical terms, provide the original Sanskrit term followed by explanation in the query's language
                - Do not add external knowledge or interpretations beyond the given text""",

                target_examples=10000,  # High number to process all chunks
                min_judge_score=6,
                seeds_per_chunk=1,
                diversity_threshold=0.65,
                min_hallucination_overlap=0.5,
                checkpoint_path=Path(checkpoint_name),
                concurrency=1,
                generator_temperature=0.2,
                answer_max_tokens=768,
                language="sa",
            ),
            chunks=chunks,
            provider=provider,
            embedder=embedder,
        )
        result_examples = result.examples
        # Save pipeline checkpoint
        save_pipeline_checkpoint(result_examples, checkpoint_name)

    # Step 3: Append final datasets (Appends directly to local QAData.jsonl inside qaAlpacaStyle/)
    with open("QAData.jsonl", "a", encoding="utf-8") as f:
        for ex in result_examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + '\n')
    print(f"Appended {len(result_examples)} examples to QAData.jsonl")

    # Print statistics (if we have a result object; otherwise compute from examples)
    if 'result' in locals():
        stats = result.stats.to_dict()
        print("\n=== Pipeline Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("\n=== Dataset Statistics ===")
        print(f"Total examples loaded from checkpoint: {len(result_examples)}")

    print(f"\n✅ Dataset generation complete.")
    print(f"   Training examples generated this run: {len(result_examples)}")

if __name__ == "__main__":
    main()
