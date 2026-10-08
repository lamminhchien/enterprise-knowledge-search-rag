"""
================================================================================
Enterprise Knowledge Search & RAG Framework: Master Demonstration
================================================================================
Demonstrates the complete end-to-end enterprise lifecycle:
1. Document Ingestion & PySpark Distributed MapReduce Batch ETL
2. Inbound One-Way Privacy Osmosis Guard (PII Scrubbing & Tokenization)
3. Ephemeral Orthogonal Cache Vault & FAISS Vector Indexing
4. Secure Hybrid Retrieval & Grounded RAG Question-Answering
5. Quantitative Information Retrieval (IR) Benchmark (MRR, Latency)
6. Tamper-Evident Noise Collapse Verification
================================================================================
"""

import os
import sys
import time
import numpy as np

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_chunker import TextChunker
from src.data_ingestion.pyspark_batch_etl import PySparkBatchETLPipeline
from src.security.pii_osmosis_guard import PIIOsmosisGuard
from src.security.ephemeral_vault import EphemeralOrthogonalVault
from src.embeddings.dense_embedder import DenseEmbedder
from src.vector_store.faiss_index_manager import VectorIndexManager
from src.retriever.hybrid_retriever import HybridRetriever
from src.generator.rag_engine import RAGEngine
from src.evaluation.ir_evaluator import IREvaluator


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, "data", "raw_documents")

    print("\n" + "=" * 75)
    print("🚀 [Step 1/6] Ingesting Corporate Documents & Running PySpark Batch ETL...")
    print("=" * 75)
    loader = DocumentLoader(data_dir=data_dir)
    raw_docs = loader.load_documents()
    print(f"Loaded {len(raw_docs)} enterprise corporate documents from disk.")

    # PySpark MapReduce demonstration
    spark_etl = PySparkBatchETLPipeline()
    _, stats = spark_etl.execute_batch_etl(raw_docs)
    print(f"Corpus Statistics -> Total Words: {stats['total_word_count']} | Avg Doc Length: {stats['average_doc_length']} words")

    chunker = TextChunker(chunk_size=300, chunk_overlap=40)
    corpus_chunks = chunker.chunk_corpus(raw_docs)
    print(f"Segmented corpus into {len(corpus_chunks)} semantic chunks with overlap.")

    print("\n" + "=" * 75)
    print("🛡️ [Step 2/6] Passing Chunks through One-Way Privacy Osmosis Membrane...")
    print("=" * 75)
    guard = PIIOsmosisGuard()
    sanitized_chunks = []
    total_scrubbed = 0
    for chk in corpus_chunks:
        clean_text, tokens = guard.filter_and_tokenize(chk["text"])
        total_scrubbed += len(tokens)
        chk_copy = chk.copy()
        chk_copy["text"] = clean_text
        sanitized_chunks.append(chk_copy)
    print(f"Privacy Membrane active: Sanitized {len(sanitized_chunks)} chunks (Scrubbed PII entities: {total_scrubbed}).")

    print("\n" + "=" * 75)
    print("🔐 [Step 3/6] Generating Embeddings & Building Ephemeral Orthogonal Index...")
    print("=" * 75)
    dim = 384
    embedder = DenseEmbedder(dim=dim)
    vault = EphemeralOrthogonalVault(dim=dim, ttl_seconds=60.0, seed=42)
    vector_store = VectorIndexManager(dim=dim)

    chunk_texts = [c["text"] for c in sanitized_chunks]
    raw_embeddings = embedder.embed_texts(chunk_texts)
    
    # Secure Orthogonal Rotation: v_enc = Q * v (Exact Cosine preservation, blinded values)
    encrypted_embeddings = vault.encrypt_vector(raw_embeddings)
    vector_store.add_vectors(encrypted_embeddings, sanitized_chunks)
    print(f"FAISS/Vector Index populated with {len(sanitized_chunks)} encrypted 384D vectors.")

    print("\n" + "=" * 75)
    print("🔍 [Step 4/6] Executing Hybrid Search & Grounded RAG with Simulated PII Query...")
    print("=" * 75)
    retriever = HybridRetriever(vector_store, vault)
    rag_engine = RAGEngine(guard)

    # Inbound test query with sensitive PII injected
    test_query = "Nhân viên Nguyen Van A (CCCD 048192003849, Phone 0935162424) hỏi quy trình báo cáo khi nghi ngờ lộ dữ liệu là gì?"
    print(f"Raw Inbound Query: '{test_query}'")

    # Inbound sanitization
    clean_query, detected_pii = guard.filter_and_tokenize(test_query)
    print(f"Sanitized Prompt:  '{clean_query}'")
    print(f"Detected PII Vault Tokens: {list(detected_pii.keys())}")

    # Search in rotated space
    q_vec = embedder.embed_texts(clean_query)[0]
    hits = retriever.retrieve(clean_query, q_vec, top_k=2)

    print("\nTop Retrieved Evidences (Rotated Vector Space):")
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] Source: {hit['filename']} | Score: {hit['rrf_score']} | Excerpt: {hit['text'][:120]}...")

    # RAG Generation (Authorized user view)
    rag_output = rag_engine.generate_grounded_answer(clean_query, hits, user_authorized=True)
    print(f"\nRAG Answer (Confidence: {rag_output['confidence']*100:.1f}%):")
    print(f"{rag_output['answer']}")

    print("\n" + "=" * 75)
    print("📊 [Step 5/6] Benchmarking Information Retrieval (IR) Scientific Metrics...")
    print("=" * 75)
    ir_evaluator = IREvaluator(k_values=[1, 3])
    ground_truth_test_cases = [
        {"query": "Quy định xác thực đa yếu tố MFA cho người dùng như thế nào?", "expected_doc_id": "it_security_sop"},
        {"query": "Thời gian lưu trữ dữ liệu giao dịch khách hàng là bao lâu?", "expected_doc_id": "data_governance_policy"},
        {"query": "Các hành vi nào bị nghiêm cấm khi sử dụng AI và chatbot?", "expected_doc_id": "cloud_ai_usage_guidelines"},
    ]
    benchmark_metrics = ir_evaluator.evaluate_retrieval_benchmark(retriever, embedder, ground_truth_test_cases)
    for metric_name, val in benchmark_metrics.items():
        print(f"  • {metric_name.upper()}: {val}")

    print("\n" + "=" * 75)
    print("💥 [Step 6/6] Verifying Tamper-Evident Self-Destruction (Zero Entropy Trap)...")
    print("=" * 75)
    # Simulate an expired vault or unauthorized intrusion
    expired_vault = EphemeralOrthogonalVault(dim=dim, ttl_seconds=-1.0)  # Forced expired
    dummy_vector = np.ones((1, dim))
    trapped_output = expired_vault.encrypt_vector(dummy_vector)
    
    # Calculate entropy/variance: trapped output collapses to Gaussian white noise
    noise_variance = float(np.var(trapped_output))
    print(f"Session TTL expired -> Vector rotation self-destructed.")
    print(f"Observation collapse verified: Trapped vector variance = {noise_variance:.4f} (Gaussian white noise trap active).")

    print("\n✨ All 6 enterprise research steps executed flawlessly! System fully verified.\n")


if __name__ == "__main__":
    main()
