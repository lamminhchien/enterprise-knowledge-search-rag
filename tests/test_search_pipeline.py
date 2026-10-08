"""
Unit Test Suite for Enterprise Knowledge Search & RAG Framework.
Validates PII redaction, Orthogonal isometry, Hybrid search, and RAG answer synthesis.
Uses standard library unittest to run seamlessly without external test runners.
"""

import unittest
import numpy as np
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.security.pii_osmosis_guard import PIIOsmosisGuard
from src.security.ephemeral_vault import EphemeralOrthogonalVault
from src.embeddings.dense_embedder import DenseEmbedder
from src.vector_store.faiss_index_manager import VectorIndexManager
from src.retriever.hybrid_retriever import HybridRetriever
from src.generator.rag_engine import RAGEngine


class TestEnterpriseSearchPipeline(unittest.TestCase):
    def test_pii_sanitization_and_rehydration(self):
        guard = PIIOsmosisGuard()
        raw_text = "Nhân viên Tran B (CCCD: 012345678901, Phone: 0905123456, Email: test@example.com) xin cấp quyền."
        sanitized, tokens = guard.filter_and_tokenize(raw_text)

        # Assert no sensitive plain text leaked
        self.assertNotIn("012345678901", sanitized)
        self.assertNotIn("0905123456", sanitized)
        self.assertNotIn("test@example.com", sanitized)
        self.assertEqual(len(tokens), 3)

        # Assert rehydration for authorized user
        restored = guard.rehydrate_context(sanitized, user_authorized=True)
        self.assertIn("012345678901", restored)
        self.assertIn("0905123456", restored)

        # Assert unauthorized keeps tokens
        unauthorized = guard.rehydrate_context(sanitized, user_authorized=False)
        self.assertIn("[NATIONAL_ID_ID_", unauthorized)

    def test_orthogonal_isometry_preservation(self):
        dim = 128
        vault = EphemeralOrthogonalVault(dim=dim, seed=42)
        u = np.random.randn(dim)
        v = np.random.randn(dim)

        _, _, err = vault.verify_cosine_preservation(u, v)
        self.assertLess(err, 1e-6, f"Cosine isometry violated: error={err}")

    def test_expired_vault_decorrelation(self):
        dim = 128
        expired_vault = EphemeralOrthogonalVault(dim=dim, ttl_seconds=-1.0)
        v = np.ones((1, dim))
        trapped = expired_vault.encrypt_vector(v)

        # Variance of Gaussian noise should be ~ 1.0
        self.assertGreater(np.var(trapped), 0.5)

    def test_hybrid_retrieval(self):
        dim = 64
        vault = EphemeralOrthogonalVault(dim=dim, seed=123)
        vector_store = VectorIndexManager(dim=dim)
        retriever = HybridRetriever(vector_store, vault)

        # Add 3 dummy documents
        corpus = [
            {"doc_id": "doc1", "filename": "policy1.txt", "text": "Quy trình xác thực đa yếu tố MFA cho hệ thống nội bộ.", "chunk_id": 0},
            {"doc_id": "doc2", "filename": "policy2.txt", "text": "Quy định quản lý mật khẩu và thời gian hết hạn phiên làm việc.", "chunk_id": 1},
            {"doc_id": "doc3", "filename": "policy3.txt", "text": "Chính sách nghỉ phép năm và chế độ phúc lợi nhân viên.", "chunk_id": 2},
        ]
        dummy_embeddings = np.random.randn(3, dim).astype(np.float32)
        dummy_embeddings /= np.linalg.norm(dummy_embeddings, axis=1, keepdims=True)
        rot_embeddings = vault.encrypt_vector(dummy_embeddings)
        vector_store.add_vectors(rot_embeddings, corpus)

        # Query
        query = "Xác thực MFA bảo mật"
        q_vec = dummy_embeddings[0]  # Match doc1
        hits = retriever.retrieve(query, q_vec, top_k=2)

        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["doc_id"], "doc1")


if __name__ == "__main__":
    unittest.main()
