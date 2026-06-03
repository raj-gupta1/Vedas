import os
import gc
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymongo import MongoClient
from colorama import Fore, init

init(autoreset=True)
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI or "xxxxx" in MONGODB_URI:
    print(f"{Fore.RED}Error: Please set your MONGODB_URI in the .env file!{Fore.RESET}")
    exit(1)

# ── Configuration ──
BASE_MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct" 
ADAPTER_PATH = "/Users/raj/PycharmProjects/VedaGPT/continuousPreTrainStyle/lora_model"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DB_NAME = "vedic_rag"
COLLECTION_NAME = "scriptures"

# Same embedding model used during ingestion — MUST match!
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Cross-encoder re-ranker: takes (query, passage) pairs and scores relevance much more accurately
# than vector similarity alone. This is the secret weapon for RAG quality.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Retrieval settings
INITIAL_CANDIDATES = 150   # broad vector search pool
VECTOR_TOP_K = 15          # retrieve top 15 from vector search
RERANK_TOP_K = 9           # keep top 5 after re-ranking

print(f"🚀 {Fore.LIGHTCYAN_EX}Starting VedaGPT Enhanced RAG on {DEVICE.upper()}...{Fore.RESET}")

# ── 1. Connect to MongoDB Atlas ──
print(f"🔌 Connecting to MongoDB Atlas...")
client = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# ── 2. Load Embedding + Re-ranker ──
print(f"🧠 Loading Bi-Encoder: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print(f"🎯 Loading Cross-Encoder Re-ranker: {RERANKER_MODEL}...")
reranker = CrossEncoder(RERANKER_MODEL)

# ── 3. Load LLM + LoRA ──
print(f"📂 Loading tokenizer from: {BASE_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"📂 Loading base model from: {BASE_MODEL_PATH}...")
if DEVICE == "mps":
    print("💡 Apple Silicon detected: Loading in float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map=DEVICE,
        torch_dtype=torch.float16,
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map=DEVICE,
        quantization_config=quant_config if DEVICE != "cpu" else None,
    )

print(f"📂 Applying LoRA adapters from: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print(f"\n{Fore.LIGHTGREEN_EX}✅ Enhanced RAG System loaded! Type 'quit' or 'exit' to stop.{Fore.RESET}\n")


# ════════════════════════════════════════════════════
# RETRIEVAL PIPELINE: Vector Search → Cross-Encoder Re-Rank
# ════════════════════════════════════════════════════

def retrieve_and_rerank(query, vector_top_k=VECTOR_TOP_K, rerank_top_k=RERANK_TOP_K):
    """
    Two-stage retrieval:
      Stage 1 — Bi-encoder vector search (fast, broad recall)
      Stage 2 — Cross-encoder re-ranking  (slow, precise relevance)
    """
    # BGE recommends prepending "Represent this sentence:" for queries
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
    
    # Stage 2: Cross-encoder re-ranking
    # The cross-encoder sees both the query AND the passage together,
    # so it understands relevance far better than cosine similarity.
    pairs = [(query, doc.get("text_with_context", doc["text"])) for doc in candidates]
    rerank_scores = reranker.predict(pairs)
    
    for i, doc in enumerate(candidates):
        doc["rerank_score"] = float(rerank_scores[i])
    
    # Sort by re-rank score (highest = most relevant)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return candidates[:rerank_top_k]


# ════════════════════════════════════════════════════
# CHAIN OF THOUGHT SYSTEM PROMPT
# ════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════

while True:
    try:
        user_query = input(f"{Fore.LIGHTYELLOW_EX}Enter your question: {Fore.RESET}")
        if user_query.lower() in ["quit", "exit"]:
            break
        if not user_query.strip():
            continue

        # ── Stage 1+2: Retrieve & Re-rank ──
        print(f"🔍 Searching sacred texts & re-ranking...")
        retrieved_docs = retrieve_and_rerank(user_query)
        
        if not retrieved_docs:
            print(f"{Fore.YELLOW}No relevant scriptures found in the database.{Fore.RESET}")
            context_block = "No scripture passages were retrieved from the database."
        else:
            print(f"📖 Retrieved & re-ranked {len(retrieved_docs)} passages.")
            
            # Show retrieval debug info
            for i, doc in enumerate(retrieved_docs):
                src = doc.get('collection', '?')
                bk = doc.get('book', '?')
                hymn = doc.get('hymn', '?')
                title = doc.get('title', '')
                vs = doc.get('score', 0)
                rs = doc.get('rerank_score', 0)
                print(f"   {Fore.LIGHTBLACK_EX}#{i+1} [{src} Bk{bk} {hymn}] vec={vs:.3f} rerank={rs:.3f}{Fore.RESET}")
            
            # Format context for the LLM
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

        # ── Build Chain-of-Thought RAG Prompt ──
        user_message = f"""Here are the retrieved scripture passages:

{context_block}

Question: {user_query}

Now follow the Chain of Thought reasoning process to answer this question."""

        # Format with chat template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}\n\nAnswer:\n"

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
        
        print(f"✨ Reasoning & generating answer...\n")
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
        
        print(f"{Fore.LIGHTBLUE_EX}VedaGPT:{Fore.RESET}")
        print(response.strip())
        print(f"\n{'═' * 70}\n")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"{Fore.RED}Error during RAG generation: {e}{Fore.RESET}")
        import traceback
        traceback.print_exc()
