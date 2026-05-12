import os
import json
import requests
import numpy as np
import logging
from functools import lru_cache
 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
 
# -------------------------------------------------
# CONFIG
# -------------------------------------------------
 
load_dotenv()
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")
CATEGORY_PATH = os.path.join(BASE_DIR, "data", "category.json")
 
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
 
TOP_K = 20
FINAL_TOP_K = 5
CONFIDENCE_THRESHOLD = 0.3
 
logging.basicConfig(level=logging.INFO)
 
# -------------------------------------------------
# LOAD CATEGORY
# -------------------------------------------------
 
with open(CATEGORY_PATH, "r") as f:
    CATEGORY_SCHEMA = json.load(f)
 
HINTS = CATEGORY_SCHEMA["hints"]
DOMAIN_DEFAULTS = CATEGORY_SCHEMA["domain_mapping_defaults"]
 
# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
 
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
# -------------------------------------------------
# CLASSIFICATION
# -------------------------------------------------
 
def classify_query(query):
    text_lower = query.lower()
 
    for label, hint in HINTS.items():
        keyword_rules = [str(x).lower() for x in hint.get("keywords", [])]
 
        if any(rule in text_lower for rule in keyword_rules):
            group = label.split("/")[0]
            return {
                "label": label,
                "category": label.split("/")[-1],
                "domain": DOMAIN_DEFAULTS.get(group, "General")
            }
 
    return {"label": "Unknown", "category": None, "domain": None}
 
# -------------------------------------------------
# LOAD VECTOR DB
# -------------------------------------------------
 
def load_vector_db():
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
 
# -------------------------------------------------
# QUERY EXPANSION
# -------------------------------------------------
 
def expand_query(query):
    prompt = f"""
Generate 3 variations of this query for better document retrieval.
 
Query: {query}
 
Return only queries.
"""
 
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        ).json()["response"]
 
        queries = [q.strip() for q in response.split("\n") if q.strip()]
        return list(set([query] + queries))
 
    except Exception as e:
        logging.error(f"Query expansion failed: {e}")
        return [query]
 
# -------------------------------------------------
# HYBRID SEARCH
# -------------------------------------------------
 
def hybrid_search(db, queries, metadata):
    all_docs = []
 
    for q in queries:
        try:
            if metadata["category"]:
                docs = db.similarity_search(q, k=TOP_K, filter={"category": metadata["category"]})
            else:
                docs = db.similarity_search(q, k=TOP_K)
 
            all_docs.extend(docs)
 
        except Exception as e:
            logging.error(f"Search failed: {e}")
 
    return all_docs
 
# -------------------------------------------------
# CROSS-ENCODER RE-RANKING
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
# CONFIDENCE SCORE
# -------------------------------------------------
 
def calculate_confidence(scores):
    if not scores:
        return 0.0
    return float(np.mean(scores))
 
# -------------------------------------------------
# PROMPT BUILDER
# -------------------------------------------------
 
def build_prompt(query, docs):
 
    context = "\n\n".join([doc.page_content[:800] for doc in docs])
 
    return f"""
You are an enterprise AI assistant.
 
STRICT RULES:
- Answer ONLY from context
- If answer not found → say "I don't know"
- Do NOT hallucinate
- Keep answer concise
- Mention source file names if possible
 
Context:
{context}
 
Question:
{query}
 
Answer:
"""
 
# -------------------------------------------------
# LLM CALL WITH RETRY
# -------------------------------------------------
 
def call_llm(prompt, retries=3):
 
    for attempt in range(retries):
        try:
            res = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            )
            return res.json()["response"]
 
        except Exception as e:
            logging.error(f"LLM call failed (attempt {attempt+1}): {e}")
 
    return "LLM failed. Please try again."
 
# -------------------------------------------------
# MAIN PIPELINE (CACHED)
# -------------------------------------------------
 
@lru_cache(maxsize=100)
def ask_question(query):
 
    logging.info(f"Query: {query}")
 
    db = load_vector_db()
 
    # 1️⃣ Expand Query
    queries = expand_query(query)
 
    # 2️⃣ Classify
    metadata = classify_query(query)
 
    # 3️⃣ Retrieve
    docs = hybrid_search(db, queries, metadata)
 
    if not docs:
        return {"answer": "No documents found", "confidence": 0}
 
    # 4️⃣ Re-rank
    docs, scores = rerank_docs(docs, query)
 
    # 5️⃣ Confidence
    confidence = calculate_confidence(scores)
 
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I could not find relevant information",
            "confidence": confidence
        }
 
    # 6️⃣ LLM
    prompt = build_prompt(query, docs)
    answer = call_llm(prompt)
 
    # 7️⃣ Sources
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
 
# -------------------------------------------------
# CLI
# -------------------------------------------------
 
if __name__ == "__main__":
 
    while True:
 
        q = input("\nAsk (exit to quit): ")
 
        if q.lower() == "exit":
            break
 
        res = ask_question(q)
 
        print("\nAnswer:\n", res["answer"])
        #print(f"\nConfidence: {res['confidence']:.2f}")
 
        if "sources" in res:
            print("\nSources:")
            for s in res["sources"]:
                print(f"- {s['file']} | {s['category']} | {s['domain']}")