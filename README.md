# VedaGPT: Continued Pre-Training, Fine-Tuning & RAG Pipeline 🪶

A premium data curation, model fine-tuning, deployment, and Retrieval-Augmented Generation (RAG) evaluation pipeline for ancient Indian Vedic literature (Sanskrit, Hindi, and English), starting with the four primary Vedas: **Rig Veda**, **Sama Veda**, **Yajur Veda**, and **Atharva Veda**.

---

## 🎯 Architecture & Roadmap
The project contains two distinct architectures designed to run in a sequential workflow:
1. **Continuous Pre-Training & QLoRA Fine-Tuning**: Infusing deep domain-specific Vedic knowledge into the model parameters using high-performance GPU fine-tuning.
2. **Retrieval-Augmented Generation (RAG) & Evaluation**: Supplementing the custom fine-tuned and quantized models with a vector database (MongoDB Atlas) and evaluating the performance using the Ragas framework.

```mermaid
graph TD
    A[Vedic PDF/Web Corpus] -->|prepareData / Scrapers| B[Raw text chunks]
    B -->|Jupyter Notebook on GPU| C[QLoRA Fine-Tuning]
    C -->|Entire book shown| D[LoRA Adapter & Merged Model]
    D -->|Pushed to Hub| E[Hugging Face Hub]
    E -->|GGUF Export| F[Local Ollama / Inference]
    
    B -->|Ingestion & Embedding| G[(MongoDB Vector DB)]
    G -->|Vector Search & Re-ranking| H[RAG Retrieval Pipeline]
    H -->|Prompt Context| I[Local Fine-Tuned LLM]
    I -->|Answers| J[Ragas Evaluation Suite]
    J -->|Evaluator LLM| K[Local Ollama: Gemma 3]
```

---

## 💻 Tech Stack & Models Used

### 1. Base Model
* **Model**: `unsloth/Llama-3.2-3B-Instruct`
* Loaded in **4-bit quantization** (`nf4` double quantization) via Unsloth during fine-tuning, and via `BitsAndBytesConfig` during local inference for high-speed computation on consumer GPUs.

### 2. Fine-Tuning (GPU-Powered)
* **Pipeline**: Handled entirely inside the Jupyter Notebook `continuousPreTrainStyle/qloraFineTune.ipynb` on a high-performance GPU (Google Colab / Jupyter environment).
* **Training strategy**: Once initial parameters and hyperparameters were stable, the model was shown the **entire book** to ensure thorough semantic comprehension and alignment.
* *Note: The scripts `continuousPreTrainStyle/finetuneContinueousPretrain.py` and `continuousPreTrainStyle/evaluateContinueousPretrain.py` are deprecated and not used.*

### 3. Model Deployment & Quantization
* **Hugging Face Hub Repositories**:
  * **LoRA Adapters**: `shinigamiRaj/Vedas-Llama-3.2-3B-LoRA`
  * **Merged 16-bit Model**: `shinigamiRaj/Vedas-Llama-3.2-3B-Merged`
* **Local Quantization**: 
  * Exported to **GGUF format** using the `q4_k_m` quantization method for efficient, CPU/GPU-split local inference (via Ollama or LM Studio).
  * 4-bit local quantization runs on non-Apple devices via `bitsandbytes`.

### 4. RAG (Retrieval-Augmented Generation)
* **Vector Database**: **MongoDB Atlas Vector Search** (Index name: `vedic_index`).
* **Embedding Model**: `BAAI/bge-base-en-v1.5` (outputs 768-dimensional vectors).
* **Two-Stage Retrieval Pipeline**:
  * **Stage 1 (Vector Search)**: Fast, broad recall using cosine similarity vector search on MongoDB Atlas.
  * **Stage 2 (Cross-Encoder Re-ranking)**: Precise re-scoring of candidate passages using `BAAI/bge-reranker-v2-m3` to yield the top relevant contexts.

### 5. RAG Evaluation (Ragas Framework)
* **Evaluation Toolkit**: **Ragas** (evaluating Context Precision, Context Recall, Faithfulness, and Answer Relevancy).
* **Evaluator LLM**: Local Ollama model **`gemma3:4b-it-qat`** (Quantization-Aware Trained model running locally on Ollama at `http://127.0.0.1:11434`, bypassing OpenAI API costs).

---

## 🛠️ Step-by-Step Execution Guide

To run the pipeline, you must follow the execution order below. The workflow is divided into two phases: **1. Continuous Pre-Training Style** followed by **2. RAG Style**.

```mermaid
graph TD
    subgraph Phase 1: Continuous Pre-Training Style
        direction TB
        P1_1[1. Run Web Scrapers:<br>scrape_*.py] --> P1_2[2. Run PDF Extractor:<br>prepareContinueousPreTrainData.py]
        P1_2 --> P1_3[3. Run Train/Val Split:<br>splitContinueousPretrainData.py]
        P1_3 --> P1_4[4. Fine-Tune on GPU:<br>qloraFineTune.ipynb]
        P1_4 --> P1_5[5. Run Local Inference:<br>inference.py]
        P1_5 --> P1_6[6. Evaluate Models:<br>evaluate_all_models.py]
    end

    subgraph Phase 2: RAG Style
        direction TB
        P1_6 --> P2_1[7. Setup MongoDB Atlas Manually<br>Create DB, Collection & Index]
        P2_1 --> P2_2[8. Ingest Vector Chunks:<br>ingest_data.py]
        P2_2 --> P2_3[9. Run RAG Inference:<br>rag_inference.py]
        P2_3 --> P2_4[10. Run RAG Evaluation:<br>evaluate_rag.py]
    end
```

### Phase 1: Continuous Pre-Training & Inference

1. **Configure Environment Variables**:
   Ensure you have configured your environment keys in the root `.env` (and `RAGStyle/.env`):
   ```env
   HF_TOKEN=your_huggingface_write_token
   MONGODB_URI="your_mongodb_connection_string"
   ```

2. **Scrape Web Sources**:
   Run the web scrapers inside the `continuousPreTrainStyle` folder to fetch the scripture texts:
   ```bash
   python3 continuousPreTrainStyle/scrape_rig_veda.py
   python3 continuousPreTrainStyle/scrape_sama_veda.py
   python3 continuousPreTrainStyle/scrape_yajur_veda.py
   python3 continuousPreTrainStyle/scrape_atharva_veda.py
   ```

3. **Extract PDF Data**:
   Extract and chunk bilingual text from PDF manuscripts with sliding window overlap:
   ```bash
   python3 continuousPreTrainStyle/prepareContinueousPreTrainData.py
   ```

4. **Split Train and Validation Datasets**:
   Run the split script to prepare the continuous pre-training corpus:
   ```bash
   python3 continuousPreTrainStyle/splitContinueousPretrainData.py
   ```

5. **Run GPU Fine-Tuning**:
   * Open `continuousPreTrainStyle/qloraFineTune.ipynb` in Google Colab or another GPU environment.
   * Run the cells to mount Google Drive, install Unsloth/Transformers dependencies, chunk the dataset with overlap, apply the Llama-3 chat template, fine-tune the model on the **entire book**, and push adapters/merged weights to Hugging Face.

6. **Run Local Inference**:
   Test the custom fine-tuned model adapters directly on your local device:
   ```bash
   python3 continuousPreTrainStyle/inference.py
   ```

7. **Evaluate All Model Adapters**:
   Compare the Base Model against the CPT adapter and the QA adapter:
   ```bash
   python3 evaluate_all_models.py
   ```

---

### Phase 2: RAG Ingestion, Retrieval & Evaluation

1. **Set Up MongoDB Atlas Manually**:
   * Create a cluster in MongoDB Atlas.
   * Create a database named `vedic_rag` and a collection named `scriptures`.
   * Create a Vector Search index named `vedic_index` with the following definition (ensuring dimensions match the BGE embedding model):
     ```json
     {
       "fields": [
         {
           "numDimensions": 768,
           "path": "embedding",
           "similarity": "cosine",
           "type": "vector"
         }
       ]
     }
     ```

2. **Ingest and Embed Data**:
   Embed and upload the prepared scriptures to MongoDB Atlas:
   ```bash
   python3 RAGStyle/ingest_data.py
   ```

3. **Run Interactive RAG CLI**:
   Query the fine-tuned model augmented by vector search and cross-encoder re-ranking:
   ```bash
   python3 RAGStyle/rag_inference.py
   ```

4. **Start Local Ollama Server & Pull Evaluator LLM**:
   Make sure Ollama is installed and running, then pull the evaluator model:
   ```bash
   ollama pull gemma3:4b-it-qat
   ```

5. **Run RAG Evaluation (Ragas)**:
   Benchmark your RAG pipeline's context precision, recall, faithfulness, and answer relevance:
   ```bash
   python3 RAGStyle/evaluate_rag.py
   ```

---

## 🐳 Docker Setup
A Dockerfile is provided to quickly build a containerized environment with all system and Python libraries installed.

### 1. Build the Docker Image
```bash
docker build -t vedagpt-pipeline .
```

### 2. Run the Container
You can run any script inside the container by overriding the command. Mount your `.env` file to pass credentials:
```bash
# Run RAG Inference (Default CMD)
docker run -it --env-file .env vedagpt-pipeline

# Run Data Ingestion
docker run -it --env-file .env vedagpt-pipeline python RAGStyle/ingest_data.py

# Run RAG Evaluation (Ensure Ollama is running and accessible from the container)
docker run -it --env-file .env vedagpt-pipeline python RAGStyle/evaluate_rag.py
```

---

## 📂 Project Structure
```text
├── books/                             # PDF source books (Sama Veda, etc.)
├── continuousPreTrainStyle/           # Continued Pre-training files
│   ├── data/                          # Folder for pre-train datasets
│   ├── lora_model/                    # Local LoRA adapter files
│   ├── qloraFineTune.ipynb            # [ACTIVE] GPU Fine-tuning & deployment notebook
│   ├── prepareContinueousPreTrainData.py # PDF overlapping page-chunk extractor
│   ├── inference.py                   # Local fine-tuned model tester
│   ├── finetuneContinueousPretrain.py # [DEPRECATED] Unused local training script
│   └── evaluateContinueousPretrain.py # [DEPRECATED] Unused local evaluation script
├── RAGStyle/                          # RAG and Evaluation files
│   ├── .env                           # RAG environment secrets
│   ├── ingest_data.py                 # MongoDB vector search ingestor
│   ├── rag_inference.py               # Interactive RAG search & chat script
│   └── evaluate_rag.py                # Ragas evaluation runner (via Ollama)
├── Dockerfile                         # Container setup for libraries installation
├── requirements.txt                   # Master Python library dependency list
├── .env                               # Root environment secrets
└── README.md                          # Main project documentation
```
