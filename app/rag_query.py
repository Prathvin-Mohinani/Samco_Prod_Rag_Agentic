import os
import json
import asyncio
import httpx
import numpy as np
import logging
import torch
import re
 
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
 
# IMPORTANT:
# Replace localhost with your Azure VM Private IP if running from Azure Web App
OLLAMA_URL = "http://4.210.114.244:11434/api/generate"
 
OLLAMA_MODEL = "llama3:8b"
 
TOP_K = 20
FINAL_TOP_K = 5
CONFIDENCE_THRESHOLD = 0.25
 
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
 
    return {
        "label": "Unknown",
        "category": None,
        "domain": None
    }
 
 
# -------------------------------------------------
# CLEAN LLM RESPONSE
# -------------------------------------------------
 
def clean_json_response(response_text):
    """
    Cleans LLM response and extracts valid JSON only.
    Removes:
    - markdown
    - extra text
    - \n characters
    - duplicate JSON wrappers
    """
 
    if not response_text:
        return {}
 
    # Remove markdown blocks
    response_text = response_text.replace("```json", "")
    response_text = response_text.replace("```", "")
 
    # Find first JSON object
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
 
    if not match:
        return {}
 
    json_text = match.group(0)
 
    try:
        return json.loads(json_text)
 
    except Exception as e:
        logging.error(f"JSON parsing failed: {e}")
        return {}
 
 
# -------------------------------------------------
# LLM CALL (ROBUST)
# -------------------------------------------------
 
async def call_llm(prompt, retries=3):
 
    async with httpx.AsyncClient(timeout=None) as client:
 
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
 
                res.raise_for_status()
 
                data = res.json()
 
                raw_response = data.get("response", "")
 
                return clean_json_response(raw_response)
 
            except Exception as e:
 
                logging.error(f"LLM attempt {attempt+1} failed: {e}")
 
    return {
        "answer": "LLM failed completely",
        "department": "Unknown",
        "category": "Unknown",
        "source": "Unknown",
        "domain": "Unknown",
        "doc_category": "Unknown",
        "label": "Unknown"
    }
 
 
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
 
        async with httpx.AsyncClient(timeout=60.0) as client:
 
            res = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )
 
            data = res.json()
 
            response = data.get("response", "")
 
            queries = [
                q.strip()
                for q in response.split("\n")
                if q.strip()
            ]
 
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
            filter={
                "category": metadata["category"]
            } if metadata["category"] else None
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
 
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
 
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
 
    context_blocks = []
 
    for i, d in enumerate(docs):
 
        meta = getattr(d, "metadata", {}) or {}
 
        src = meta.get("source", "Unknown")
        dom = meta.get("domain", "Unknown")
        cat = meta.get("category", "Unknown")
        label = meta.get("label", "Unknown")
 
        text = (d.page_content or "").strip()
 
        snippet = text[:2500]
 
        context_blocks.append(
            f"""
[DOC {i+1}]
source={src}
domain={dom}
category={cat}
label={label}
 
CONTENT:
{snippet}
"""
        )
 
    context = "\n\n".join(context_blocks)
 
    return f"""
You are an Enterprise AI Assistant.
 
STRICT RULES:
1. Use ONLY the provided context.
2. NEVER use external knowledge.
3. NEVER generate explanations before or after JSON.
4. NEVER say:
   - "Here is the JSON"
   - "Below is the answer"
   - markdown
   - code blocks
5. Output MUST be VALID JSON ONLY.
6. Do NOT include \\n or escaped formatting inside values.
7. Return ONLY ONE JSON object.
8. Do NOT duplicate answers.
9. If answer not found:
   "answer": "I don't know based on the provided documents"
 
CONTEXT:
{context}
 
USER QUESTION:
{query}
 
RETURN STRICT JSON FORMAT:
 
{{
    "answer": "",
    "department": "",
    "category": "",
    "source": "",
    "domain": "",
    "doc_category": "",
    "label": ""
}}
 
IMPORTANT:
- All fields must be plain text.
- No nested JSON.
- No extra commentary.
- No markdown.
- No escaped characters.
- No duplicate JSON.
""".strip()
 
 
# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
 
async def ask_question(query):
 
    logging.info(f"Query: {query}")
 
    db = load_vector_db()
 
    # Step 1: Expand query
    queries = await expand_query(query)
 
    # Step 2: Classification
    metadata = classify_query(query)
 
    # Step 3: Search
    docs = await hybrid_search(db, queries, metadata)
 
    if not docs:
 
        return {
            "answer": "No documents found",
            "department": "Unknown",
            "category": "Unknown",
            "source": "Unknown",
            "domain": "Unknown",
            "doc_category": "Unknown",
            "label": "Unknown",
            "confidence": 0,
            "sources": []
        }
 
    # Step 4: Rerank
    docs, scores = rerank_docs(docs, query)
 
    # Step 5: Confidence
    confidence = calculate_confidence(scores)
 
    if confidence < CONFIDENCE_THRESHOLD:
 
        return {
            "answer": "I could not find relevant information",
            "department": "Unknown",
            "category": "Unknown",
            "source": "Unknown",
            "domain": "Unknown",
            "doc_category": "Unknown",
            "label": "Unknown",
            "confidence": round(confidence, 2),
            "sources": []
        }
 
    # Step 6: LLM
    prompt = build_prompt(query, docs)
 
    llm_response = await call_llm(prompt)
 
    # Step 7: Sources
    sources = []
 
    for d in docs:
 
        sources.append({
            "file": d.metadata.get("source", "Unknown"),
            "category": d.metadata.get("category", "Unknown"),
            "domain": d.metadata.get("domain", "Unknown")
        })
 
    # FINAL CLEAN RESPONSE
    return {
        "answer": llm_response.get(
            "answer",
            "I don't know based on the provided documents"
        ),
        "department": llm_response.get("department", "Unknown"),
        "category": llm_response.get("category", "Unknown"),
        "source": llm_response.get("source", "Unknown"),
        "domain": llm_response.get("domain", "Unknown"),
        "doc_category": llm_response.get("doc_category", "Unknown"),
        "label": llm_response.get("label", "Unknown"),
        "confidence": round(confidence, 2),
        "sources": sources
    }
 
 
# -------------------------------------------------
# CLI TEST
# -------------------------------------------------
 
if __name__ == "__main__":
 
    while True:
 
        q = input("\nAsk (exit to quit): ")
 
        if q.lower() == "exit":
            break
 
        res = asyncio.run(ask_question(q))
 
        print(res["answer"]) 
        print("Department :", res["department"])