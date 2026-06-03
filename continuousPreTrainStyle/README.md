# Continuous Pre-Training Data Pipeline for Vedas

This directory contains scripts, tools, and raw dataset structures designed to fetch, parse, clean, and format the four primary Hindu Vedas (**Rig Veda**, **Sama Veda**, **Yajur Veda**, and **Atharva Veda**) into high-quality JSONL pre-training data files.

## 📁 Directory Structure

*   `data/`: Contains final parsed continuous pre-training JSONL datasets.
    *   `rig_veda_pretrain.jsonl` (Rig Veda English)
    *   `sama_veda_pretrain.jsonl` (Sama Veda English)
    *   `yajur_veda_pretrain.jsonl` (Unified Black & White Yajur Veda English)
    *   `atharva_veda_pretrain.jsonl` (Atharva Veda English)
    *   `continueousPreTrainData.jsonl` (Merged master pre-training dataset)
    *   `README.md` (Detailed dataset composition, authorship, and schema description)
*   `scrape_rig_veda.py`: Web scraper for the Rig Veda (Ralph T.H. Griffith English translation).
*   `scrape_sama_veda.py`: Web scraper for the Sama Veda (Ralph T.H. Griffith English translation).
*   `scrape_yajur_veda.py`: Web scraper for both the Black (Keith) and White (Griffith) Yajur Vedas.
*   `scrape_atharva_veda.py`: Web scraper for the Atharva Veda (Ralph T.H. Griffith English translation).
*   `prepareContinueousPreTrainData.py`: PDF parser and chunker for bilingual/PDF sources.

---

## 🛠️ Usage

All scrapers are designed to resolve output paths absolute to this script directory, automatically placing files inside the `data/` subdirectory.

To run any scraper, run it using Python:

```bash
# Run the Rig Veda scraper
python3 scrape_rig_veda.py

# Run the Sama Veda scraper
python3 scrape_sama_veda.py

# Run the Yajur Veda scraper (scrapes and merges both Black & White Yajur Vedas)
python3 scrape_yajur_veda.py

# Run the Atharva Veda scraper
python3 scrape_atharva_veda.py
```

---

## 🏛️ Dataset & Chunking Overview

*   **Rig Veda & Atharva Veda**: Extracted at the **Hymn (Sukta)** level, creating rich narrative units for continuous pre-training.
*   **Sama Veda**: Scraped at **Decade/Hymn** level, or page-chunked via PDF (4 chunks per page with 150-char overlap).
*   **Yajur Veda**: Integrates the **Black Yajur Veda** (grouped at Anuvaka level) and the **White Yajur Veda** (grouped at Verse level) into one file.
*   **PDF Pipeline**: For printed manuscripts, parses text page-by-page, chunking each into 4 overlapping parts (150-character overlap) to preserve translation pairs without context clipping.

For specific dataset parameters, schemas, and licence details, see [data/README.md](file:///Users/raj/PycharmProjects/VedaGPT/continuousPreTrainStyle/data/README.md).
