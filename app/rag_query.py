import os
import json
import asyncio
import httpx
import numpy as np
import logging
import torch
 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
 
# -------------------------------------------------
# CONFIG
# -------------------------------------------------
 
logging.basicConfig(level=logging.INFO)
 
load_dotenv()
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")
CATEGORY_PATH = os.path.join(BASE_DIR, "data", "category.json")
 
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
 
TOP_K = 20
FINAL_TOP_K = 5
CONFIDENCE_THRESHOLD = 0.35
 
# -------------------------------------------------
# LOAD CATEGORY
# -------------------------------------------------
 
with open(CATEGORY_PATH, "r") as f:
    CATEGORY_SCHEMA = json.load(f)
 
HINTS = CATEGORY_SCHEMA["hints"]
DOMAIN_DEFAULTS = CATEGORY_SCHEMA["domain_mapping_defaults"]
 
# -------------------------------------------------
# LOAD MODELS (Singleton Pattern)
# -------------------------------------------------
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
db = None
 
def load_vector_db():
    global db
    if db is None:
        db = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return db
 
# -------------------------------------------------
# CLASSIFICATION
# -------------------------------------------------
 
def classify_query(query):
    text_lower = query.lower()
 
    for label, hint in HINTS.items():
        keywords = [str(x).lower() for x in hint.get("keywords", [])]
 
        if any(k in text_lower for k in keywords):
            group = label.split("/")[0]
            return {
                "label": label,
                "category": label.split("/")[-1],
                "domain": DOMAIN_DEFAULTS.get(group, "General")
            }
 
    return {"label": "Unknown", "category": None, "domain": None}
 
# -------------------------------------------------
# LLM CALL (ROBUST)
# -------------------------------------------------
 
async def call_llm(prompt, retries=3):
    async with httpx.AsyncClient(timeout=None) as client:  # remove timeout
        for attempt in range(retries):
            try:
                res = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False
                    }
                )
 
                print("STATUS:", res.status_code)
                print("RAW:", res.text)
 
                res.raise_for_status()
                data = res.json()
 
                return data.get("response", "No response")
 
            except Exception as e:
                logging.error(f"LLM attempt {attempt+1} failed: {e}")
 
    return "LLM failed completely"
 
# -------------------------------------------------
# QUERY EXPANSION
# -------------------------------------------------
 
async def expand_query(query):
    prompt = f"""
Generate 3 different search queries for better document retrieval.
 
Original Query: {query}
 
Return only queries (one per line).
"""
 
    try:
        response = await call_llm(prompt)
        queries = [q.strip() for q in response.split("\n") if q.strip()]
        return list(set([query] + queries))
    except:
        return [query]
 
# -------------------------------------------------
# SEARCH
# -------------------------------------------------
 
async def search_one(db, query, metadata):
    loop = asyncio.get_event_loop()
 
    return await loop.run_in_executor(
        None,
        lambda: db.similarity_search(
            query,
            k=TOP_K,
            filter={"category": metadata["category"]} if metadata["category"] else None
        )
    )
 
async def hybrid_search(db, queries, metadata):
    tasks = [search_one(db, q, metadata) for q in queries]
    results = await asyncio.gather(*tasks)
 
    docs = []
    for r in results:
        docs.extend(r)
 
    return docs
 
# -------------------------------------------------
# RERANK
# -------------------------------------------------
 
def rerank_docs(docs, query):
    if not docs:
        return [], []
 
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
 
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
 
    top_docs = [d for d, _ in ranked[:FINAL_TOP_K]]
    top_scores = [float(s) for _, s in ranked[:FINAL_TOP_K]]
 
    return top_docs, top_scores
 
# -------------------------------------------------
# CONFIDENCE
# -------------------------------------------------
 
def calculate_confidence(scores):
    return float(np.mean(scores)) if scores else 0.0
 
# -------------------------------------------------
# PROMPT
# -------------------------------------------------
def build_prompt(query, docs):
    """
    Forces JSON-only output with this schema:
 
    {
      "answer": "...",
      "department": "...",
      "category": "...",
      "source": "...",
      "domain": "...",
      "doc_category": "...",
      "label": "..."
    }
 
    Notes:
    - JSON cannot contain duplicate keys. So we use "category" for business category
      and "doc_category" for the document metadata category.
    - Values must be derived ONLY from context/metadata; otherwise "Unknown".
    """
 
    context_blocks = []
    for i, d in enumerate(docs):
        meta = getattr(d, "metadata", {}) or {}
 
        src = meta.get("source", f"doc_{i+1}")
        dom = meta.get("domain", "Unknown")
        cat = meta.get("category", "Unknown")
        label = meta.get("label", "Unknown")
 
        text = (d.page_content or "").strip()
        snippet = text[:800]
 
        context_blocks.append(
            f"[DOC {i+1}] source={src} | domain={dom} | category={cat} | label={label}\n"
            f"{snippet}"
        )
 
    context = "\n\n".join(context_blocks)
 
    return f"""
You are a highly accurate Enterprise AI Assistant designed for document-based question answering.
 
=====================
🔒 STRICT RULES
=====================
1. Use ONLY the provided context to answer.
2. Do NOT use external knowledge.
3. If the answer is not found in the context, set:
   "answer": "I don't know based on the provided documents"
4. Do NOT guess or assume anything.
5. Do NOT fabricate data.
6. Output MUST be valid JSON ONLY (no markdown, no extra text).
7. JSON keys MUST appear exactly as specified below.
 
=====================
📚 CONTEXT
=====================
{context}
 
=====================
❓ USER QUESTION
=====================
{query}
 
=====================
🧠 OUTPUT FORMAT (STRICT JSON)
=====================
Return ONLY one JSON object with EXACTLY these keys:
 
1) "answer":
   - Answer using ONLY the context.
   - If not found, use EXACT fallback:
     "I don't know based on the provided documents"
 
2) "department":
   - Use the best matching document's metadata "domain" as department IF no explicit department exists.
   - If department is not determinable, set "Unknown".
   - Do NOT invent.
 
3) "category":
   - This is the BUSINESS category (e.g., Dispatch).
   - Prefer extracting from the document text if explicitly stated.
   - If not explicitly stated, use best matching document's metadata category if it represents business category.
   - Otherwise "Unknown".
 
4) "source":
   - Use best matching document metadata "source".
   - Otherwise "Unknown".
 
5) "domain":
   - Use best matching document metadata "domain".
   - Otherwise "Unknown".
 
6) "doc_category":
   - Use best matching document metadata "category" (document classification category).
   - Otherwise "Unknown".
 
7) "label":
   - Use best matching document metadata "label".
   - Otherwise "Unknown".
 
IMPORTANT:
- Choose the SINGLE most relevant document from the context blocks to populate source/domain/doc_category/label.
- Keep strings as plain text. No extra fields. No trailing commentary.
 
=====================
✅ RESPONSE (JSON ONLY)
=====================
""".strip()

# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
 
async def ask_question(query):
    logging.info(f"Query: {query}")
 
    db = load_vector_db()
 
    # Step 1: Expand query
    queries = await expand_query(query)
 
    # Step 2: Classify
    metadata = classify_query(query)
 
    # Step 3: Search
    docs = await hybrid_search(db, queries, metadata)
 
    if not docs:
        return {"answer": "No documents found", "confidence": 0}
 
    # Step 4: Rerank
    docs, scores = rerank_docs(docs, query)
 
    # Step 5: Confidence
    confidence = calculate_confidence(scores)
 
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I could not find relevant information",
            "confidence": confidence
        }
 
    # Step 6: LLM
    prompt = build_prompt(query, docs)
    answer = await call_llm(prompt)
 
    # Step 7: Sources
    sources = [
        {
            "file": d.metadata.get("source"),
            "category": d.metadata.get("category"),
            "domain": d.metadata.get("domain")
        }
        for d in docs
    ]
 
    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources
    }

# CLI TEST
# -------------------------------------------------
 
if __name__ == "__main__":
 
    while True:
 
        q = input("\nAsk (exit to quit): ")
 
        if q.lower() == "exit":
            break
 
        res = asyncio.run(ask_question(q))
 
        print("\nAnswer:\n", res["answer"])
        print(f"\nConfidence: {res['confidence']:.2f}")
 
        if "sources" in res:
            print("\nSources:")
            for s in res["sources"]:
                print(f"- {s['file']} | {s['category']} | {s['domain']}")