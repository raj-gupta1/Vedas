# VedaGPT Data Preparation Pipeline 🪶

A premium data curation, extraction, and evaluation pipeline for fine-tuning and pre-training Large Language Models (LLMs) on ancient Indian Vedic literature (Sanskrit and Hindi), starting with the Atharva Veda.

---

## 🎯 Project Objectives & Roadmap
This repository forms the foundation for data curation to enable:
* **Model Fine-Tuning**: Supervised Fine-Tuning (SFT) and Continued Pre-training (CPT) of open-source LLMs on custom, high-fidelity Sanskrit-Hindi Vedic datasets.
* **Quantization**: Quantizing the fine-tuned LLM models (e.g., GGUF, AWQ, GPTQ) for efficient, local, and low-latency inference.
* **Rigorous Evaluation & Comparative Analysis**: Systematic benchmarking of performance and accuracy across:
  * **Base Model** (pre-trained, unmodified)
  * **Fine-Tuned Model** (specifically trained on Vedic nuances)
  * **Quantized Model** (evaluating trade-offs in speed vs. diacritic/Sanskrit fidelity)
  * **RAG (Retrieval-Augmented Generation)** (comparing fine-tuning against vector database search context)

---

## 🚀 Key Features

### 1. SFT Dataset Generation (`prepareData.py`)
* **Intelligent LLM-in-the-loop Extraction**: Generates diverse, high-quality question-answering (Instruction-Output) pairs directly from complex multi-lingual PDF layouts.
* **Auto-Judging & Evaluation**: Integrated evaluation step that scores each generated pair from `1` to `10` on relevance, context consistency, and hallucination metrics.
* **Granular Progress Saving**: Saves progress incrementally into range-specific checkpoint files (`pipeline_checkpoint_x_y.jsonl`) to ensure zero progress loss.
* **Master Registry**: Appends the rich-metadata generated records directly into your master `train.jsonl`.

### 2. Dataset Cleaning & Post-Processing (`clean_dataset.py`)
* **Zero-Loss Filtering**: Reads your master `train.jsonl` and extracts only the `"instruction"` and `"output"` fields required for training.
* **Separate Output**: Saves the cleaned records into a new file **`cleaned_train.jsonl`** while keeping the original `train.jsonl` with all its valuable audit metrics (hallucination scores, reasoning, and page sources) untouched.

### 3. Continued Pre-Training (PT) Preparation (`prepareContinueousPreTrainData.py`)
* **Full-Book Extraction**: Reads unstructured raw texts across any specified page range (or the entire book).
* **Sliding Window Overlap Chunks**: Splices page text into a configurable number of chunks per page (e.g. 4 or 5) with a sliding character-overlap window (defaults to `150` characters). This ensures sentence structures and mantras are not truncated at chunk boundaries.
* **Standard PT Format**: Automatically formats chunks into standard `{"text": "..."}` structures inside **`continueousPreTrainData.jsonl`**.

---

## 📂 Project Structure

```text
├── books/                             # PDF source books (Atharva Veda, etc.)
├── pipeline_checkpoint_*.jsonl       # Incremental SFT generation checkpoints
├── train.jsonl                        # Master SFT dataset (with rich audit metadata)
├── cleaned_train.jsonl                # Clean SFT training file (instruction, output only)
├── continueousPreTrainData.jsonl      # Overlapping raw text chunks for pre-training
├── prepareData.py                     # SFT generation and evaluation pipeline
├── clean_dataset.py                   # SFT training dataset cleaner
├── prepareContinueousPreTrainData.py  # Pre-training sliding overlap data preparer
├── .gitignore                         # Git files exclusion configuration
└── README.md                          # Project overview and documentation
```

---

## 🛠️ Quick Start & Usage

Ensure you have your virtual environment activated:
```bash
source vedas/bin/activate
```

### 1. Generate SFT Dataset
To process new pages and add high-quality instruction-output pairs with LLM scoring, configure `START_PAGE` and `END_PAGE` in `prepareData.py` and run:
```bash
python prepareData.py
```

### 2. Clean the SFT Dataset for Fine-Tuning
To strip the metadata and prepare `cleaned_train.jsonl` for training:
```bash
python clean_dataset.py
```

### 3. Generate Pre-Training Chunks
To parse the book and extract overlapping text blocks for continued pre-training:
```bash
python prepareContinueousPreTrainData.py
```

---

## 📝 Configuration Variables
Both python data scripts are fully customizable by editing their top-level constants:

* **SFT Pipeline (`prepareData.py`)**:
  * `START_PAGE` / `END_PAGE`: Process specific page segments.
  * `CHUNKS_PER_PAGE`: Control granularity of page splitting.
  * `SAMPLES_PER_CHUNK`: Number of instruction pairs to generate per chunk.

* **PT Pipeline (`prepareContinueousPreTrainData.py`)**:
  * `CHUNKS_PER_PAGE`: Number of text chunks per page.
  * `OVERLAP_CHARACTERS`: Number of overlapping characters between consecutive chunks.
