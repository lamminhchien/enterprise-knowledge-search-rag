"""
FastAPI REST Service Endpoint for Enterprise Semantic Search and RAG.
Provides high-speed HTTP interface for internal intranet & enterprise search portals.
"""

from typing import List, Optional
import os
import sys

# Append root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.security.pii_osmosis_guard import PIIOsmosisGuard
from src.security.ephemeral_vault import EphemeralOrthogonalVault
from src.embeddings.dense_embedder import DenseEmbedder
from src.vector_store.faiss_index_manager import VectorIndexManager
from src.retriever.hybrid_retriever import HybridRetriever
from src.generator.rag_engine import RAGEngine

app = FastAPI(
    title="Enterprise Knowledge Search & RAG API",
    version="1.0.0",
    description="Secure, ephemeral-vaulted semantic search engine for enterprise document governance."
)

# Global in-memory components (Initialized at startup)
guard = PIIOsmosisGuard()
vault = EphemeralOrthogonalVault(dim=384, ttl_seconds=300.0)
embedder = DenseEmbedder(dim=384)
vector_store = VectorIndexManager(dim=384)
retriever = HybridRetriever(vector_store, vault)
rag_engine = RAGEngine(guard)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    user_role: Optional[str] = "employee"


class SearchResponse(BaseModel):
    query: str
    sanitized_query: str
    results: List[dict]


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[dict]
    confidence: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vault_active": not vault.is_expired,
        "indexed_chunks": len(vector_store._metadata_store)
    }


@app.post("/api/v1/search", response_model=SearchResponse)
def semantic_search_endpoint(req: QueryRequest):
    # 1. Inbound PII Osmosis filtering
    clean_query, _ = guard.filter_and_tokenize(req.query)

    # 2. Dense Embedding & Secure Retrieval
    q_vec = embedder.embed_texts(clean_query)[0]
    hits = retriever.retrieve(clean_query, q_vec, top_k=req.top_k)

    return {
        "query": req.query,
        "sanitized_query": clean_query,
        "results": hits
    }


@app.post("/api/v1/ask", response_model=AskResponse)
def ask_rag_endpoint(req: QueryRequest):
    clean_query, _ = guard.filter_and_tokenize(req.query)
    q_vec = embedder.embed_texts(clean_query)[0]
    hits = retriever.retrieve(clean_query, q_vec, top_k=req.top_k)

    is_authorized = req.user_role in ["compliance_officer", "ciso", "admin"]
    rag_result = rag_engine.generate_grounded_answer(
        query=req.query,
        retrieved_contexts=hits,
        user_authorized=is_authorized
    )

    return {
        "query": req.query,
        "answer": rag_result["answer"],
        "citations": rag_result["citations"],
        "confidence": rag_result["confidence"]
    }
