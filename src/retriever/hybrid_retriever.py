"""
Hybrid Retriever with Ephemeral Orthogonal Security Integration.
Merges Dense Semantic Cosine Search (in rotated space) with Sparse Keyword Matching using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Tuple
import re
import numpy as np

from src.vector_store.faiss_index_manager import VectorIndexManager
from src.security.ephemeral_vault import EphemeralOrthogonalVault


class HybridRetriever:
    """
    Dual-Channel Retriever:
    Channel 1: Secure Dense Cosine Search via Ephemeral Orthogonal Vault.
    Channel 2: Lexical Keyword Matching (BM25 term overlap proxy).
    Fusion: Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_manager: VectorIndexManager,
        vault: EphemeralOrthogonalVault,
        rrf_constant: int = 60
    ):
        self.vector_manager = vector_manager
        self.vault = vault
        self.rrf_constant = rrf_constant

    def _lexical_keyword_score(self, query: str, chunk_text: str) -> float:
        """Computes normalized lexical token overlap score."""
        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        chunk_words = set(re.findall(r"\b\w+\b", chunk_text.lower()))
        if not query_words or not chunk_words:
            return 0.0
        intersection = query_words.intersection(chunk_words)
        return len(intersection) / len(query_words)

    def retrieve(
        self,
        query: str,
        raw_query_vector: np.ndarray,
        top_k: int = 3,
        use_encrypted_vault: bool = True
    ) -> List[Dict[str, any]]:
        """
        Executes hybrid retrieval.
        query: Text query string
        raw_query_vector: 384D dense vector
        """
        # Step 1: Secure Dense Semantic Search
        if use_encrypted_vault:
            # Vector is rotated into blinded orthogonal space
            search_vector = self.vault.encrypt_vector(raw_query_vector)
        else:
            search_vector = raw_query_vector

        dense_candidates = self.vector_manager.search(search_vector, top_k=top_k * 2)

        # Step 2: Lexical Scoring across retrieved candidate pool
        rrf_scores: Dict[str, float] = {}
        candidate_meta: Dict[str, Dict[str, any]] = {}

        # Dense ranking contribution
        for rank, (meta, dense_sim) in enumerate(dense_candidates, start=1):
            chk_id = meta["chunk_id"]
            candidate_meta[chk_id] = meta
            rrf_scores[chk_id] = rrf_scores.get(chk_id, 0.0) + (1.0 / (self.rrf_constant + rank))

        # Lexical ranking contribution
        lexical_sorted = sorted(
            candidate_meta.items(),
            key=lambda item: self._lexical_keyword_score(query, item[1]["text"]),
            reverse=True
        )
        for rank, (chk_id, meta) in enumerate(lexical_sorted, start=1):
            rrf_scores[chk_id] = rrf_scores.get(chk_id, 0.0) + (1.0 / (self.rrf_constant + rank))

        # Sort by final fused RRF score
        fused_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        final_results = []
        for chk_id, score in fused_ranking:
            meta = candidate_meta[chk_id]
            final_results.append({
                "chunk_id": chk_id,
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "text": meta["text"],
                "rrf_score": round(score, 6)
            })

        return final_results
