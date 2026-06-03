# Continuous Pre-Training & QLoRA Fine-Tuning for Vedas

This directory contains the pipeline components, notebook, and tools used to scrape, prepare, chunk, and fine-tune Large Language Models (LLMs) on the four primary Hindu Vedas: **Rig Veda**, **Sama Veda**, **Yajur Veda**, and **Atharva Veda**.

---

## 🚀 GPU Fine-Tuning Process (`qloraFineTune.ipynb`)

Rather than running local CPU training, a high-performance GPU fine-tuning workflow was executed:
1. **Interactive GPU Environment**: All fine-tuning was carried out using `qloraFineTune.ipynb` (e.g., inside Google Colab using a T4 or higher GPU).
2. **Double Quantization**: The base model `unsloth/Llama-3.2-3B-Instruct` was loaded in 4-bit (`load_in_4bit=True`) using Unsloth's optimized double quantization config.
3. **Data Preparation**: 
   * Raw text was chunked with a **20% sliding window overlap** (`chunk_with_overlap`) to ensure no sacred verses or mantras were cut in half.
   * Chunks were wrapped inside an assistant-role chat template (`formatting_pretrain_function`) so that the model learned the book content while retaining its standard conversational instruction-following capability.
4. **Entire Book Training**: Once training parameters converged and loss was stable, the model was shown the **entire book** corpus for a complete epoch.
5. **Multi-Format Deployment**:
   * Pushed raw LoRA adapters to Hugging Face Hub: `shinigamiRaj/Vedas-Llama-3.2-3B-LoRA`
   * Pushed merged 16-bit float model to Hugging Face Hub: `shinigamiRaj/Vedas-Llama-3.2-3B-Merged`
   * Exported GGUF format (`q4_k_m` quantization) for local consumption (e.g., via LM Studio / Ollama).

> [!WARNING]
> **Deprecated Scripts**: 
> The local scripts `finetuneContinueousPretrain.py` and `evaluateContinueousPretrain.py` are **deprecated** and are **not used** for the final model training. They remain in the codebase for reference only.

---

## 📁 Folder Structure

* **`qloraFineTune.ipynb`**: [ACTIVE] The primary Jupyter Notebook used for GPU fine-tuning, merging, and Hugging Face deployment.
* **`inference.py`**: Runs local interactive inference using the base model combined with your LoRA adapters (requires `HF_TOKEN` in `.env`).
* **`prepareContinueousPreTrainData.py`**: Extracts text from bilingual/PDF source manuscripts (e.g., Sama Veda PDF) and chunks them with configurable overlap.
* **`splitContinueousPretrainData.py`**: Splits raw JSONL continuous pre-train datasets into training and validation sets.
* **`data/`**: Contains parsed continuous pre-training JSONL datasets:
  * `rig_veda_pretrain.jsonl` (Rig Veda English)
  * `sama_veda_pretrain.jsonl` (Sama Veda English)
  * `yajur_veda_pretrain.jsonl` (Unified Black & White Yajur Veda English)
  * `atharva_veda_pretrain.jsonl` (Atharva Veda English)
  * `continueousPreTrainData.jsonl` (Merged master pre-training dataset)
* **`scrape_*.py`**: Scripts to scrape Rig Veda, Sama Veda, Yajur Veda, and Atharva Veda from sacred-texts.com.

---

## 🛠️ Usage Instructions

### 1. Data Generation
Extract overlapping text blocks from PDF sources:
```bash
python3 prepareContinueousPreTrainData.py
```

### 2. Fine-Tuning
Upload `qloraFineTune.ipynb` to your GPU environment (like Google Colab), ensure your `.env` contains your Hugging Face Hub credentials, and run the notebook cells to train and push your model to Hugging Face.

### 3. Local Adapter Inference
Run the local tester script (fetches Hugging Face adapters or loads local adapters if present):
```bash
python3 inference.py
```
