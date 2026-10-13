"""
FastAPI Enterprise Agentic AI & RAG Gateway.
Provides high-performance endpoints for:
1. Agentic Chat with LangGraph State Machine & Server-Sent Events (SSE) Streaming.
2. Anthropic Model Context Protocol (MCP) tool discovery & execution.
3. Multimodal OCR document ingestion & PII redaction.
4. Ephemeral Orthogonal Matrix (Q^T Q = I) semantic search.
"""

from typing import List, Optional, Dict, Any
import os
import sys
import time
import json
import asyncio

# Append root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.security.pii_osmosis_guard import PIIOsmosisGuard
from src.security.ephemeral_vault import EphemeralOrthogonalVault
from src.security.prompt_guard import PromptSecurityGuard
from src.embeddings.dense_embedder import DenseEmbedder
from src.vector_store.faiss_index_manager import VectorIndexManager
from src.retriever.hybrid_retriever import HybridRetriever
from src.generator.rag_engine import RAGEngine
from src.tools.mcp_server import MCPServerRegistry
from src.tools.ocr_tool import DocumentOCRTool
from src.agent.langgraph_agent import EnterpriseAgenticWorkflow

app = FastAPI(
    title="Enterprise Agentic AI & Secure RAG Gateway",
    version="2.0.0",
    description="Agentic StateGraph, MCP Tool Calling, Zero-Trust Guardrails & Ephemeral Orthogonal Search."
)

# Global in-memory components (Initialized at startup)
guard = PIIOsmosisGuard()
prompt_guard = PromptSecurityGuard()
vault = EphemeralOrthogonalVault(dim=384, ttl_seconds=600.0)
embedder = DenseEmbedder(dim=384)
vector_store = VectorIndexManager(dim=384)
retriever = HybridRetriever(vector_store, vault)
rag_engine = RAGEngine(guard)
ocr_tool = DocumentOCRTool()
mcp_registry = MCPServerRegistry(hybrid_retriever=retriever, embedder=embedder, security_guard=prompt_guard)
agent_engine = EnterpriseAgenticWorkflow(mcp_registry=mcp_registry, pii_guard=guard, security_guard=prompt_guard)


# --- Request & Response Models ---
class AgentChatRequest(BaseModel):
    query: str
    user_role: Optional[str] = "employee"
    session_id: Optional[str] = "default_session"
    stream: Optional[bool] = False


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    user_role: Optional[str] = "employee"


# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise Agentic Gateway",
        "vault_active": not vault.is_expired,
        "ttl_remaining_s": max(0.0, vault.ttl_seconds - (time.time() - vault.created_at)),
        "registered_mcp_tools": len(mcp_registry.list_tools())
    }


@app.get("/api/v1/mcp/tools")
def list_mcp_tools():
    """Returns Anthropic MCP compliant tool schemas."""
    return {
        "protocol": "Model Context Protocol (MCP) v1.0",
        "tools": mcp_registry.list_tools()
    }


@app.post("/api/v1/agent/chat")
async def agent_chat_endpoint(req: AgentChatRequest):
    """
    Executes LangGraph Agentic Workflow:
    Input Guardrail -> Intent Router -> MCP Tool Calling -> Grounding Verification -> Rehydration.
    Supports Server-Sent Events (SSE) streaming when stream=True.
    """
    # Streaming mode via Server-Sent Events (SSE)
    if req.stream:
        async def event_generator():
            yield f"data: {json.dumps({'event': 'agent_started', 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.05)

            # Run Agentic StateGraph
            result = agent_engine.run(
                user_query=req.query,
                user_role=req.user_role,
                session_id=req.session_id
            )

            # Stream intermediate tokens
            yield f"data: {json.dumps({'event': 'guardrail_passed', 'is_safe': result['is_safe'], 'pii_scrubbed': list(result['detected_pii'].keys())})}\n\n"
            await asyncio.sleep(0.05)

            yield f"data: {json.dumps({'event': 'tool_executed', 'tools': [t['name'] for t in result['selected_tools']]})}\n\n"
            await asyncio.sleep(0.05)

            # Stream words
            for word in result["grounded_answer"].split(" "):
                yield f"data: {json.dumps({'event': 'token', 'token': word + ' '})}\n\n"
                await asyncio.sleep(0.02)

            yield f"data: {json.dumps({'event': 'done', 'citations': result['citations'], 'confidence': result['confidence']})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Standard JSON Response
    result = agent_engine.run(
        user_query=req.query,
        user_role=req.user_role,
        session_id=req.session_id
    )
    return {
        "session_id": result["session_id"],
        "user_role": result["user_role"],
        "is_safe": result["is_safe"],
        "intent": result["intent"],
        "tools_called": [t["name"] for t in result["selected_tools"]],
        "answer": result["grounded_answer"],
        "citations": result["citations"],
        "grounding_score": result["grounding_score"],
        "confidence": result["confidence"],
        "pii_detected": list(result["detected_pii"].keys())
    }


@app.post("/api/v1/ocr/scan")
def ocr_document_endpoint(file: UploadFile = File(...)):
    """Ingests scanned invoice/document image and returns OCR structured payload."""
    contents = file.file.read()
    result = ocr_tool.parse_document(file.filename)
    return result


@app.post("/api/v1/search")
def semantic_search_endpoint(req: QueryRequest):
    """Secure Orthogonal Vector Search (FAISS + BM25 RRF)."""
    clean_query, _ = guard.filter_and_tokenize(req.query)
    q_vec = embedder.embed_texts([clean_query])[0]
    hits = retriever.retrieve(clean_query, q_vec, top_k=req.top_k)
    return {
        "query": req.query,
        "sanitized_query": clean_query,
        "results": hits
    }


@app.post("/api/v1/ask")
def ask_rag_endpoint(req: QueryRequest):
    """Grounded RAG synthesis with verifiable citations."""
    clean_query, _ = guard.filter_and_tokenize(req.query)
    q_vec = embedder.embed_texts([clean_query])[0]
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
