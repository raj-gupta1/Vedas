"""
VedaGPT Local Web UI with RAG
==============================

Runs a local web server that:
  1. Embeds the user's query with BGE
  2. Retrieves relevant passages from MongoDB Atlas (vector search)
  3. Re-ranks with a cross-encoder
  4. Sends the RAG-augmented prompt to shinigamiRaj/IndicVedas on Modal
  5. Returns the answer + sources

Usage:
    source continuousPreTrainStyle/vedaFineTune/bin/activate
    python continuousPreTrainStyle/webui.py

Then open http://localhost:8080 in your browser.
"""

import os
import json
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import modal
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAGStyle", ".env"))

MONGODB_URI = os.getenv("MONGODB_URI")

# ── Configuration ────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
PORT = 8080
DB_NAME = "vedic_rag"
COLLECTION_NAME = "scriptures"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Retrieval settings
INITIAL_CANDIDATES = 150
VECTOR_TOP_K = 15
RERANK_TOP_K = 3  # Retrieve top 3 reconstructed parent documents for context size safety

# ── Globals ──────────────────────────────────────────────────────────────────
model_ready = False
mongo_ready = False
is_local_mongo = False
warmup_status = "not_started"
generate_fn = None
embedder = None
reranker = None
collection = None
in_memory_docs = None

# ── RAG System Prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are VedaGPT, an expert scholar of the ancient Indian Vedas (Rig Veda, Sama Veda, Yajur Veda, Atharva Veda), Charaka Samhita, Sushruta Samhita, Rasa Jala Nidhi, and IRJAY.

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
- Keep answers focused and well-structured.
- Maintain the style of writing as per the ancient Vedic texts where required.
- Do not output literal step headers (e.g., 'Step 1 — Analyze the Evidence', 'Step 2') in your final response. Conduct the step-by-step analysis internally and output only the final scholarly synthesized answer."""


# ═══════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def reconstruct_parent_documents(candidates):
    """
    Parent-document sliding window retrieval: For each candidate chunk, fetch 
    its sibling chunks and combine the matching chunk with at most 1 chunk before
    and 1 chunk after to preserve context without exceeding token limits.
    """
    seen_docs = set()
    reconstructed = []
    
    for doc in candidates:
        doc_id = doc.get("doc_id")
        if not doc_id:
            # Fallback if doc_id doesn't exist in older database entries
            reconstructed.append(doc)
            continue
            
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        
        # Query sibling chunks from MongoDB using the unique doc_id
        sibling_chunks = list(collection.find({"doc_id": doc_id}).sort("chunk_index", 1))
        
        if sibling_chunks:
            matching_idx = doc.get("chunk_index", 0)
            
            # Context window: matching chunk +/- 1 sibling chunk
            start_idx = max(0, matching_idx - 1)
            end_idx = min(len(sibling_chunks) - 1, matching_idx + 1)
            
            selected_chunks = sibling_chunks[start_idx : end_idx + 1]
            full_text = "\n\n".join([c["text"] for c in selected_chunks])
            
            reconstructed_doc = doc.copy()
            reconstructed_doc["text"] = full_text
            reconstructed_doc["text_with_context"] = f"{doc.get('text_with_context', '')} (Context Window)"
            reconstructed.append(reconstructed_doc)
        else:
            reconstructed.append(doc)
            
    return reconstructed


def retrieve_and_rerank(query):
    """
    Two-stage retrieval:
      Stage 1 — Bi-encoder vector search (fast, broad recall)
                Uses Atlas $vectorSearch if available, otherwise falls back
                to in-memory numpy cosine similarity search.
      Stage 2 — Cross-encoder re-ranking  (slow, precise relevance)
    """
    global in_memory_docs
    query_vector = embedder.encode(f"Represent this sentence: {query}").tolist()
    candidates = []

    # Stage 1: Vector search
    # Attempt Atlas Vector Search first if not running on local MongoDB
    if not is_local_mongo:
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vedic_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": INITIAL_CANDIDATES,
                        "limit": VECTOR_TOP_K,
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "doc_id": 1,
                        "text": 1,
                        "text_with_context": 1,
                        "collection": 1,
                        "book": 1,
                        "hymn": 1,
                        "title": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            candidates = list(collection.aggregate(pipeline))
            print(f"    [Atlas Search] Retrieved {len(candidates)} candidates via $vectorSearch")
        except Exception as e:
            print(f"    ⚠️  Atlas $vectorSearch failed: {e}. Falling back to local in-memory search...")
            candidates = []

    # Fallback to local in-memory cosine similarity search
    if not candidates:
        if in_memory_docs is None:
            print("    💾 Loading documents from MongoDB into memory for local search...")
            in_memory_docs = list(collection.find({}, {
                "_id": 0,
                "doc_id": 1,
                "text": 1,
                "text_with_context": 1,
                "collection": 1,
                "book": 1,
                "hymn": 1,
                "title": 1,
                "embedding": 1
            }))
            print(f"    💾 Loaded {len(in_memory_docs)} documents.")

        if in_memory_docs:
            import numpy as np
            query_arr = np.array(query_vector)
            embeddings_matrix = np.array([doc["embedding"] for doc in in_memory_docs])
            
            # Normalize vectors for cosine similarity
            query_norm = query_arr / (np.linalg.norm(query_arr) or 1.0)
            embeddings_norms = np.linalg.norm(embeddings_matrix, axis=1)
            embeddings_norms[embeddings_norms == 0] = 1.0
            normalized_embeddings = embeddings_matrix / embeddings_norms[:, np.newaxis]
            
            similarities = np.dot(normalized_embeddings, query_norm)
            top_indices = np.argsort(similarities)[::-1][:VECTOR_TOP_K]
            
            for idx in top_indices:
                doc = in_memory_docs[idx].copy()
                doc.pop("embedding", None)
                doc["score"] = float(similarities[idx])
                candidates.append(doc)
            print(f"    [Local Search] Retrieved {len(candidates)} candidates via in-memory search")

    if not candidates:
        return []

    # Stage 2: Cross-encoder re-ranking
    pairs = [(query, doc.get("text_with_context", doc["text"])) for doc in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=4, max_length=512)

    for i, doc in enumerate(candidates):
        doc["rerank_score"] = round(float(rerank_scores[i]), 4)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Reconstruct parent documents for the top RERANK_TOP_K candidates
    top_candidates = candidates[:RERANK_TOP_K]
    return reconstruct_parent_documents(top_candidates)


def build_rag_prompt(query, retrieved_docs):
    """Build the RAG prompt formatted with Llama 3/3.2 chat template."""
    if not retrieved_docs:
        context_block = "No scripture passages were retrieved from the database."
    else:
        context_block = ""
        for i, doc in enumerate(retrieved_docs):
            src = doc.get("collection", "Veda")
            bk = doc.get("book", "?")
            hymn = doc.get("hymn", "?")
            title = doc.get("title", "")
            text = doc.get("text", "")

            header = f"[{src}, Book {bk}, {hymn}"
            if title:
                header += f" — {title}"
            header += "]"

            context_block += f"--- Passage {i+1} {header} ---\n{text}\n\n"

    user_message = f"""Here are the retrieved scripture passages:

{context_block}

Question: {query}

Now follow the Chain of Thought reasoning process to answer this question."""

    # Format the prompt using standard Llama 3/3.2 chat template syntax
    # System and user messages are explicitly separated for the chat-tuned model
    formatted_chat_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return formatted_chat_prompt


# ═══════════════════════════════════════════════════════════════════════
# HTTP HANDLER
# ═══════════════════════════════════════════════════════════════════════

class VedaGPTHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/health":
            self._health_check()
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/generate":
            self._generate()
        else:
            self.send_error(404)

    def _serve_html(self):
        html_path = os.path.join(FRONTEND_DIR, "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _health_check(self):
        data = {
            "status": "ok",
            "environment_correct": model_ready and mongo_ready,
            "model": "shinigamiRaj/IndicVedas",
            "warmup_status": warmup_status,
            "mongodb_connected": mongo_ready,
            "rag_enabled": mongo_ready and embedder is not None,
        }
        self._send_json(data)

    def _generate(self):
        global model_ready, warmup_status
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body)
            prompt = payload.get("prompt", "")

            if not prompt:
                self._send_json({"error": "No prompt provided"}, status=400)
                return

            # Stage 1+2: Retrieve & Re-rank from MongoDB
            sources = []
            if mongo_ready and embedder is not None:
                print(f"  🔍 Retrieving & re-ranking for: {prompt[:60]}...")
                t_rag = time.time()
                retrieved_docs = retrieve_and_rerank(prompt)
                rag_elapsed = time.time() - t_rag
                print(f"  📖 Retrieved {len(retrieved_docs)} passages in {rag_elapsed:.1f}s")

                # Build RAG-augmented prompt
                full_prompt = build_rag_prompt(prompt, retrieved_docs)

                # Prepare sources for frontend
                for doc in retrieved_docs:
                    sources.append({
                        "collection": doc.get("collection", ""),
                        "book": doc.get("book", ""),
                        "hymn": doc.get("hymn", ""),
                        "title": doc.get("title", ""),
                        "text": doc.get("text", "")[:300],
                        "rerank_score": doc.get("rerank_score", 0),
                    })
            else:
                full_prompt = prompt

            # Send to Modal for generation
            t0 = time.time()
            # Use "completion" mode so we send the full RAG prompt as-is
            # (the system prompt + context is already in the prompt)
            if sources:
                response = generate_fn.remote(full_prompt, mode="completion", max_new_tokens=1500)
            else:
                response = generate_fn.remote(prompt, mode="chat")
            elapsed = time.time() - t0

            model_ready = True
            warmup_status = "ready"

            print(f"  ✅ Generated in {elapsed:.1f}s")
            self._send_json({
                "response": response,
                "sources": sources,
                "elapsed": round(elapsed, 2),
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        msg = str(args[0]) if args else ""
        if "/api/" in msg or "GET / " in msg:
            super().log_message(format, *args)


# ═══════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════

def warmup_model():
    """Warm up the Modal container in the background."""
    global model_ready, warmup_status
    try:
        warmup_status = "loading"
        print("  ⏳ Warming up Modal container (may take 1-2 min on cold start)...")
        generate_fn.remote("warmup", mode="completion", max_new_tokens=1)
        model_ready = True
        warmup_status = "ready"
        print("  ✅ Model is ready!")
    except Exception as e:
        warmup_status = f"failed: {e}"
        print(f"  ⚠️  Warmup failed: {e}")


CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


if __name__ == "__main__":
    print(f"""
{BOLD}{CYAN}╔════════════════════════════════════════════════════════════╗
║    🙏  VedaGPT — Local Web Interface with RAG  🙏       ║
║    Model: shinigamiRaj/IndicVedas (via Modal)            ║
║    RAG: MongoDB Atlas + BGE + Cross-Encoder              ║
╚════════════════════════════════════════════════════════════╝{RESET}
""")

    # ── 1. Connect to MongoDB ──
    print(f"  {DIM}Connecting to MongoDB Atlas: {MONGODB_URI}...{RESET}")
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        client.server_info()
        collection = client[DB_NAME][COLLECTION_NAME]
        doc_count = collection.count_documents({})
        mongo_ready = True
        is_local_mongo = False
        print(f"  {GREEN}✅ Atlas MongoDB connected! ({doc_count} documents in collection){RESET}")
    except Exception as e:
        print(f"  {YELLOW}⚠️  Atlas MongoDB connection failed: {e}{RESET}")
        print(f"  {DIM}🔄 Falling back to local MongoDB (mongodb://localhost:27017/)...{RESET}")
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
            client.server_info()
            collection = client[DB_NAME][COLLECTION_NAME]
            doc_count = collection.count_documents({})
            mongo_ready = True
            is_local_mongo = True
            print(f"  {GREEN}✅ Local MongoDB connected! ({doc_count} documents in collection){RESET}")
        except Exception as local_e:
            print(f"  {YELLOW}❌ Local MongoDB connection failed: {local_e}{RESET}")
            print(f"  {YELLOW}   RAG will be disabled, using model-only mode.{RESET}")

    # ── 2. Load Embedding + Reranker ──
    if mongo_ready:
        print(f"  {DIM}Loading embedding model: {EMBEDDING_MODEL}...{RESET}")
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  {GREEN}✅ Embedder loaded!{RESET}")

        print(f"  {DIM}Loading cross-encoder reranker on CPU (prevents local GPU OOM): {RERANKER_MODEL}...{RESET}")
        reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
        print(f"  {GREEN}✅ Reranker loaded!{RESET}")

    # ── 3. Look up deployed Modal function ──
    print(f"  {DIM}Looking up deployed Modal app...{RESET}")
    try:
        VedaGPTCls = modal.Cls.from_name(
            "vedagpt-vllm-inference",
            "VedaGPTInference",
        )
        generate_fn = VedaGPTCls().generate
        print(f"  {GREEN}✅ Found deployed Modal function!{RESET}")
    except Exception as e:
        print(f"""
  {BOLD}❌ Modal app not deployed yet!{RESET}

  Run this first:
    modal deploy continuousPreTrainStyle/modalinference.py

  Error: {e}
""")
        exit(1)

    # ── 4. Warmup in background ──
    warmup_thread = threading.Thread(target=warmup_model, daemon=True)
    warmup_thread.start()

    # ── 5. Start HTTP server ──
    server = HTTPServer(("0.0.0.0", PORT), VedaGPTHandler)
    print(f"\n  {BOLD}{GREEN}🌐 Open in your browser:{RESET} {BOLD}http://localhost:{PORT}{RESET}")
    print(f"  {DIM}Press Ctrl+C to stop.{RESET}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  {CYAN}🙏 Dhanyavad! Server stopped.{RESET}\n")
        server.server_close()
