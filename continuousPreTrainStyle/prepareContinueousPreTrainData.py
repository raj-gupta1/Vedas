import json
import os
from tqdm import tqdm
from pypdf import PdfReader

# Configuration for this run (Updated to point to root parent folders relative to continuousPreTrainStyle/)
START_PAGE = 0
END_PAGE = None  # Set to None to extract the ENTIRE book, or a number (e.g. 600) for a specific end page
CHUNKS_PER_PAGE = 4  # Number of chunks per page (e.g. 4 or 5)
OVERLAP_CHARACTERS = 150  # Overlap in characters between consecutive chunks to ensure context isn't lost
PDF_PATH = "../books/Translation-Of-The-Sama-Veda.pdf"
OUTPUT_PATH = "Translation_Of_The_Sama_Veda.jsonl"

def extract_pages(pdf_path: str, start: int, end: int | None) -> list[tuple[int, str]]:
    """Extract text from specified pages of the PDF, returning (page_num, text)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
    reader = PdfReader(pdf_path)
    pages = []
    
    # If end is None, default to the total pages of the book
    if end is None:
        end = len(reader.pages)
    else:
        end = min(end, len(reader.pages))
    
    for page_num in tqdm(range(start, end), desc="Extracting pages for Pre-training", unit="page"):
        text = (reader.pages[page_num].extract_text() or "").strip()
        if text:
            pages.append((page_num, text))
    return pages

def chunk_page(text: str, chunks_count: int, overlap: int = 150) -> list[str]:
    """Split page text into chunks_count parts with an overlap between consecutive parts."""
    text = text.strip()
    if not text:
        return []
        
    # Split text into chunks_count parts with overlap
    part_size = max(1, len(text) // chunks_count)
    parts = []
    
    for j in range(chunks_count):
        start_idx = j * part_size
        # The end index is (j + 1) * part_size plus some overlap
        # If it's the last chunk, we take everything up to the end of the text
        if j == chunks_count - 1:
            end_idx = len(text)
        else:
            end_idx = min(len(text), (j + 1) * part_size + overlap)
            
        chunk_text = text[start_idx:end_idx].strip()
        if chunk_text:
            parts.append(chunk_text)
            
    return parts

def save_pt_data(pages: list[tuple[int, str]], output_path: str, chunks_count: int, overlap: int = 150):
    """Chunk and save extracted page texts in Pre-training JSONL format."""
    total_chunks = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for page_num, text in pages:
            chunks = chunk_page(text, chunks_count, overlap)
            for chunk_text in chunks:
                record = {"text": chunk_text}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_chunks += 1
            
    print(f"\n✅ Pre-training data saved to: {output_path}")
    print(f"📊 Extracted {len(pages)} pages -> Generated {total_chunks} chunks (with {overlap} chars overlap).")

def main():
    print(f"Starting Pre-training (PT) data extraction from {PDF_PATH}...")
    print(f"Processing pages {START_PAGE} to {END_PAGE or 'End of Book'}...")
    print(f"Chunking rate: {CHUNKS_PER_PAGE} chunks per page with {OVERLAP_CHARACTERS} character overlap.")
    
    try:
        pages = extract_pages(PDF_PATH, START_PAGE, END_PAGE)
        if pages:
            save_pt_data(pages, OUTPUT_PATH, CHUNKS_PER_PAGE, OVERLAP_CHARACTERS)
        else:
            print("⚠️ No non-empty pages extracted.")
    except Exception as e:
        print(f"❌ Error during extraction: {e}")

if __name__ == "__main__":
    main()
