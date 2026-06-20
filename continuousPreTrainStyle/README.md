# Continuous Pre-Training & Fine-Tuning Pipeline for Vedas & Ayurveda

This directory contains scripts, tools, and raw dataset structures designed to fetch, parse, clean, and format the four primary Hindu Vedas (**Rig Veda**, **Sama Veda**, **Yajur Veda**, and **Atharva Veda**) as well as Ayurvedic texts (**Charaka Samhita**, **Sushruta Samhita**, etc.) into high-quality JSONL pre-training data files. It also houses the **local** and **cloud-based (Modal)** fine-tuning and inference logic.

## 📁 Directory Structure

### 1. Data Preparation & Scrapers
*   `data/`: Contains final parsed continuous pre-training JSONL datasets.
*   `scrape_*.py`: Web scrapers for the Rig Veda, Sama Veda, Yajur Veda, and Atharva Veda.
*   `charaka_scraper.py` / `sushruta_scraper.py` / `rasa_jala_nidhi_scraper.py` / `irjay_scraper.py`: Ayurvedic text extractors.
*   `prepareContinueousPreTrainData.py` & `splitContinueousPretrainData.py`: Tools for parsing PDF manuscripts and splitting the train/val datasets.
*   `clean_and_split_data.py` / `normalize_pretrain_data.py` / `merge_pt_data.py`: Data cleaning, normalization, and merging utilities.
*   `fix_hf_config.py`: HF configuration corrector for vLLM compatibility.

### 2. Fine-Tuning Scripts
*   **`modalFineTune.py` (New ✨)**: **Cloud/Serverless Fine-Tuning (Base).** A fully automated script running on [Modal](https://modal.com) that provisions an L40S GPU, fine-tunes `Qwen/Qwen2.5-14B-Instruct` using Unsloth, merges the LoRA adapters, and pushes directly to Hugging Face.
*   **`modalLORAfineTune.py` (New ✨)**: **Cloud/Serverless Fine-Tuning (Resumed LoRA).** Resumes training from an existing 14B LoRA adapter on an L40S GPU on Modal with full dataset coverage to maximize knowledge override.
*   **`qloraFineTune.ipynb` (Classic)**: **Local/Colab Fine-Tuning.** A Jupyter Notebook designed to run Unsloth QLoRA fine-tuning on local GPUs or Google Colab environments (typically using `Llama-3.2-3B`).
*   `finetuneContinueousPretrain.py` & `evaluateContinueousPretrain.py`: [DEPRECATED] Unused local training scripts.

### 3. Inference Scripts
*   **`modalinference.py` (New ✨)**: **Cloud/Serverless Inference.** Blazing fast inference hosted on Modal using the **vLLM** engine. Features continuous batching, PagedAttention, and an interactive CLI prompt (`modal run -q continuousPreTrainStyle/modalinference.py`).
*   **`inference.py`**: **Local Inference.** Standard `transformers`-based script to test LoRA adapters on your local machine.

### 4. Local Web UI
*   **`webui.py` (New ✨)**: Local Python web server that connects Vector Search (MongoDB) and the deployed Modal model.
*   **`frontend/index.html`**: Premium, glassmorphic UI interface built with HTML, Tailwind-alternative custom CSS, and vanilla JS.

---

## 🛠️ Usage

### Running the Scrapers
All scrapers are designed to resolve output paths absolute to this script directory, automatically placing files inside the `data/` subdirectory.

```bash
# Run Vedic scrapers
python3 scrape_rig_veda.py
python3 scrape_sama_veda.py
python3 scrape_yajur_veda.py
python3 scrape_atharva_veda.py

# Run Ayurvedic scrapers
python3 charaka_scraper.py
python3 sushruta_scraper.py
```

### Running Fine-Tuning
Depending on your compute environment, pick one of the tracks:

```bash
# Track A: Modal Serverless Cloud GPU (Base Model)
modal run continuousPreTrainStyle/modalFineTune.py

# Track B: Modal Serverless Cloud GPU (Resume existing LoRA)
# Start resumed training:
modal run --detach continuousPreTrainStyle/modalLORAfineTune.py::train
# Test resumed inference:
modal run continuousPreTrainStyle/modalLORAfineTune.py::infer
# Export merged 16bit, GGUF and push to HF:
modal run continuousPreTrainStyle/modalLORAfineTune.py::push_export

# Track C: Google Colab / Local
# Open continuousPreTrainStyle/qloraFineTune.ipynb in your Jupyter environment
```

### Running Inference & Web UI
```bash
# Option 1: Modal Serverless vLLM CLI
modal run -q continuousPreTrainStyle/modalinference.py

# Option 2: Local HuggingFace Transformers CLI
python3 continuousPreTrainStyle/inference.py

# Option 3: RAG Web UI (Directly with Python)
source continuousPreTrainStyle/vedaFineTune/bin/activate
python3 continuousPreTrainStyle/webui.py

# Option 4: RAG Web UI (Dockerized with Local MongoDB)
docker-compose up --build
```

---

## 🏛️ Dataset & Chunking Overview

*   **Rig Veda & Atharva Veda**: Extracted at the **Hymn (Sukta)** level, creating rich narrative units for continuous pre-training.
*   **Sama Veda**: Scraped at **Decade/Hymn** level.
*   **Yajur Veda**: Integrates the **Black Yajur Veda** (grouped at Anuvaka level) and the **White Yajur Veda** (grouped at Verse level) into one file.
*   **Ayurvedic Texts**: Hierarchical chunking retaining Sloka and Chapter context.
*   **PDF Pipeline**: For printed manuscripts, parses text page-by-page, chunking each into overlapping parts (150-character overlap) to preserve translation pairs without context clipping.

For specific dataset parameters, schemas, and licence details, see [data/README.md](file:///Users/raj/PycharmProjects/VedaGPT/continuousPreTrainStyle/data/README.md).
