import logging
 
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
 
from app.rag_query import ask_question
from app.auth import verify_api_key
 
# -------------------------------------------------
# CONFIG
# -------------------------------------------------
 
logging.basicConfig(level=logging.INFO)
 
app = FastAPI(
    title="Enterprise RAG API",
    description="Async RAG system with FastAPI",
    version="1.0"
)
 
# -------------------------------------------------
# CORS
# -------------------------------------------------
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# -------------------------------------------------
# REQUEST / RESPONSE MODELS
# -------------------------------------------------
 
class QueryRequest(BaseModel):
    question: str
 
 
class SourceModel(BaseModel):
    file: str | None = None
    category: str | None = None
    domain: str | None = None
 
 
class QueryResponse(BaseModel):
 
    answer: str
 
    department: str | None = None
    category: str | None = None
    source: str | None = None
    domain: str | None = None
    doc_category: str | None = None
    label: str | None = None
 
    confidence: float
 
    sources: list[SourceModel] | None = None
 
 
# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
 
@app.get("/")
async def root():
    return {"message": "RAG API is running 🚀"}
 
 
@app.get("/health")
async def health():
    return {"status": "healthy"}
 
 
# -------------------------------------------------
# MAIN RAG ENDPOINT
# -------------------------------------------------
 
@app.post("/ask", response_model=QueryResponse)
async def ask(
    query: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
 
    logging.info(f"Incoming query: {query.question}")
 
    try:
 
        result = await ask_question(query.question)
 
        return {
            "answer": result.get("answer", ""),
 
            "department": result.get("department", "Unknown"),
            "category": result.get("category", "Unknown"),
            "source": result.get("source", "Unknown"),
            "domain": result.get("domain", "Unknown"),
            "doc_category": result.get("doc_category", "Unknown"),
            "label": result.get("label", "Unknown"),
 
            "confidence": result.get("confidence", 0.0),
 
            "sources": result.get("sources", [])
        }
 
    except Exception as e:
 
        logging.error(f"Error in /ask: {e}")
 
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
 
 
# -------------------------------------------------
# BULK QUERY ENDPOINT
# -------------------------------------------------
 
@app.post("/ask-bulk")
async def ask_bulk(
    queries: list[QueryRequest],
    api_key: str = Depends(verify_api_key)
):
 
    results = []
 
    for q in queries:
 
        try:
 
            res = await ask_question(q.question)
 
            results.append(res)
 
        except Exception as e:
 
            results.append({
                "error": str(e)
            })
 
    return results