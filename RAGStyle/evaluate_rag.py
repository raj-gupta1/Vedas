"""
VedaGPT RAG Evaluation using RAGAS
===================================
Evaluates the RAG pipeline on a curated Vedic knowledge test set using:
  - Context Precision   → Are the retrieved passages relevant?
  - Context Recall      → Did we retrieve all needed passages?
  - Faithfulness        → Is the answer grounded in the retrieved context?
  - Response Relevancy  → Does the answer actually address the question?

Uses local Ollama (gemma3) as the evaluator LLM — no OpenAI key needed.
"""

import os
import json
import warnings

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymongo import MongoClient
from colorama import Fore, init

from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
)
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from openai import OpenAI

init(autoreset=True)
load_dotenv()

# Suppress deprecation warnings from ragas LangchainLLMWrapper
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# ═══════════════════════════════════════════════════════
# CONFIGURATION — Must match rag_inference.py exactly
# ═══════════════════════════════════════════════════════

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI or "xxxxx" in MONGODB_URI:
    print(f"{Fore.RED}Error: Set MONGODB_URI in .env!{Fore.RESET}")
    exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DATASET_PATH = os.path.join(SCRIPT_DIR, "eval_dataset.json")

BASE_MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/Users/raj/PycharmProjects/VedaGPT/continuousPreTrainStyle/lora_model"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DB_NAME = "vedic_rag"
COLLECTION_NAME = "scriptures"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

INITIAL_CANDIDATES = 150
VECTOR_TOP_K = 15
RERANK_TOP_K = 9

# Ollama evaluator LLM (running locally — no API key needed)
OLLAMA_MODEL = "gemma3:4b-it-qat"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


# ═══════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════

print(f"🚀 {Fore.LIGHTCYAN_EX}VedaGPT RAG Evaluation Suite{Fore.RESET}")
print(f"{'═' * 60}")

# MongoDB
print(f"🔌 Connecting to MongoDB Atlas...")
client = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Embedding + Re-ranker
print(f"🧠 Loading Bi-Encoder: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print(f"🎯 Loading Cross-Encoder Re-ranker: {RERANKER_MODEL}...")
reranker = CrossEncoder(RERANKER_MODEL)

# LLM + LoRA
print(f"📂 Loading tokenizer: {BASE_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"📂 Loading base model: {BASE_MODEL_PATH}...")
if DEVICE == "mps":
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, device_map=DEVICE, torch_dtype=torch.float16,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, device_map=DEVICE,
    )

print(f"📂 Applying LoRA adapters: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print(f"{Fore.LIGHTGREEN_EX}✅ All models loaded.{Fore.RESET}\n")


# ═══════════════════════════════════════════════════════
# RETRIEVAL PIPELINE (reused from rag_inference.py)
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are VedaGPT, an expert scholar of the ancient Indian Vedas (Rig Veda, Sama Veda, Yajur Veda, and Atharva Veda).

You will be given scripture passages retrieved from the Vedic texts along with the user's question.

Follow this Chain of Thought reasoning process:

**Step 1 — Analyze the Evidence:**
Read each retrieved passage carefully. Identify which passages are directly relevant to the question and which are tangential.

**Step 2 — Extract Key Insights:**
Pull out the specific verses, concepts, deities, rituals, or teachings that directly address the question. Note the source (which Veda, Book, Hymn).

**Step 3 — Synthesize Your Answer:**
Combine insights from the relevant passages into a clear, scholarly answer. Always cite the specific source (e.g., "Rig Veda, Book 10, Hymn CXCI").

**Step 4 — Supplement if Needed:**
If the retrieved passages do not fully answer the question, you may supplement with your own knowledge of the Vedas, but clearly distinguish between what comes from the passages and what comes from your broader knowledge.

Rules:
- Always cite specific Veda, Book, and Hymn numbers when referencing passages.
- Be scholarly yet accessible — explain concepts for a modern reader.
- If a question cannot be answered from the Vedas, say so honestly.
- Do NOT fabricate verse numbers or citations.
- Keep answers focused and well-structured."""


def retrieve_and_rerank(query, vector_top_k=VECTOR_TOP_K, rerank_top_k=RERANK_TOP_K):
    """Two-stage retrieval: vector search → cross-encoder re-ranking."""
    query_vector = embedder.encode(f"Represent this sentence: {query}").tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vedic_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": INITIAL_CANDIDATES,
                "limit": vector_top_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "text_with_context": 1,
                "collection": 1,
                "book": 1,
                "hymn": 1,
                "title": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    candidates = list(collection.aggregate(pipeline))
    if not candidates:
        return []

    pairs = [(query, doc.get("text_with_context", doc["text"])) for doc in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, doc in enumerate(candidates):
        doc["rerank_score"] = float(rerank_scores[i])

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:rerank_top_k]


def generate_answer(query, retrieved_docs):
    """Generate an answer using the RAG LLM given the query and retrieved contexts."""
    if not retrieved_docs:
        context_block = "No scripture passages were retrieved from the database."
    else:
        context_block = ""
        for i, doc in enumerate(retrieved_docs):
            src = doc.get('collection', 'Veda')
            bk = doc.get('book', '?')
            hymn = doc.get('hymn', '?')
            title = doc.get('title', '')
            text = doc.get('text', '')

            header = f"[{src}, Book {bk}, {hymn}"
            if title:
                header += f" — {title}"
            header += "]"
            context_block += f"--- Passage {i+1} {header} ---\n{text}\n\n"

    user_message = f"""Here are the retrieved scripture passages:

{context_block}

Question: {query}

Now follow the Chain of Thought reasoning process to answer this question."""

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}\n\nAnswer:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ═══════════════════════════════════════════════════════
# RUN EVALUATION
# ═══════════════════════════════════════════════════════

def main():
    # 1. Load eval dataset
    print(f"📖 Loading evaluation dataset from {EVAL_DATASET_PATH}...")
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"   {len(eval_data)} questions loaded.\n")

    # 2. Run each question through the RAG pipeline
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(eval_data):
        q = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i+1}/{len(eval_data)}] {Fore.LIGHTYELLOW_EX}{q}{Fore.RESET}")

        # Retrieve
        docs = retrieve_and_rerank(q)
        ctx_texts = [doc.get("text", "") for doc in docs]

        # Generate
        answer = generate_answer(q, docs)
        print(f"    → {Fore.LIGHTBLACK_EX}{answer[:120]}...{Fore.RESET}\n")

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx_texts)
        ground_truths.append(gt)

    # 3. Build HuggingFace Dataset for RAGAS
    print(f"\n{'═' * 60}")
    print(f"📊 {Fore.LIGHTCYAN_EX}Running RAGAS Evaluation...{Fore.RESET}")
    print(f"   Evaluator LLM: {OLLAMA_MODEL} (via Ollama)")
    print(f"{'═' * 60}\n")

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # 4. Configure RAGAS evaluator LLM (local Ollama)
    openai_client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    evaluator_llm = llm_factory(OLLAMA_MODEL, client=openai_client)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    # 5. Run RAGAS evaluation
    metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # Calculate average scores
    avg_scores = {}
    for metric in metrics:
        name = metric.name
        scores_list = result[name]
        valid_scores = [s for s in scores_list if s is not None and s == s]
        avg_scores[name] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # 6. Print results
    print(f"\n{'═' * 60}")
    print(f"📈 {Fore.LIGHTGREEN_EX}VEDAGPT RAG EVALUATION REPORT{Fore.RESET}")
    print(f"{'═' * 60}")
    print(f"{'Metric':<25} | {'Score':<10}")
    print(f"{'-' * 40}")

    for metric_name, score in avg_scores.items():
        color = Fore.LIGHTGREEN_EX if score >= 0.7 else (Fore.LIGHTYELLOW_EX if score >= 0.4 else Fore.LIGHTRED_EX)
        print(f"{metric_name:<25} | {color}{score:.4f}{Fore.RESET}")

    print(f"{'═' * 60}")

    # 7. Save detailed results
    results_path = os.path.join(SCRIPT_DIR, "eval_results.json")
    results_df = result.to_pandas()
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"\n💾 Detailed per-question results saved to: {Fore.LIGHTCYAN_EX}{results_path}{Fore.RESET}")

    # 8. Also save a human-readable summary
    summary_path = os.path.join(SCRIPT_DIR, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("VedaGPT RAG Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluator LLM: {OLLAMA_MODEL}\n")
        f.write(f"Embedding Model: {EMBEDDING_MODEL}\n")
        f.write(f"Re-ranker: {RERANKER_MODEL}\n")
        f.write(f"Questions evaluated: {len(eval_data)}\n\n")
        f.write("Scores:\n")
        for metric_name, score in avg_scores.items():
            f.write(f"  {metric_name}: {score:.4f}\n")
        f.write("\n\nPer-question breakdown:\n")
        f.write("-" * 50 + "\n")
        for i, row in results_df.iterrows():
            f.write(f"\nQ{i+1}: {row.get('question', 'N/A')}\n")
            f.write(f"  Answer (first 200 chars): {str(row.get('answer', ''))[:200]}\n")
            for metric_name in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                if metric_name in row:
                    f.write(f"  {metric_name}: {row[metric_name]:.4f}\n")

    print(f"📝 Human-readable summary saved to: {Fore.LIGHTCYAN_EX}{summary_path}{Fore.RESET}")
    print(f"\n{Fore.LIGHTGREEN_EX}✅ Evaluation complete!{Fore.RESET}\n")


if __name__ == "__main__":
    main()
