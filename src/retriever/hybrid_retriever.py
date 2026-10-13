"""
Hybrid Retriever with Ephemeral Orthogonal Security Integration.
Merges Dense Semantic Cosine Search (in rotated space) with True Inverted Index Okapi BM25
using Reciprocal Rank Fusion (RRF) across the entire corpus.
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np
from rank_bm25 import BM25Okapi

from src.vector_store.faiss_index_manager import VectorIndexManager
from src.security.ephemeral_vault import EphemeralOrthogonalVault


class HybridRetriever:
    """
    Genuine Dual-Channel Hybrid Retriever:
    Channel 1: Secure Dense Cosine Search via Ephemeral Orthogonal Vault (FAISS / PyTorch).
    Channel 2: Lexical Inverted Index Okapi BM25 over the complete document corpus.
    Fusion: Reciprocal Rank Fusion (RRF k=60) combining independent global channel rankings.
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
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus_map: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text into lowercase alphanumeric terms for lexical indexing."""
        return re.findall(r"\b\w+\b", text.lower())

    def update_bm25_index(self):
        """Builds or refreshes the global BM25Okapi inverted index from the vector store metadata."""
        all_chunks = self.vector_manager._metadata_store
        if not all_chunks:
            return

        self._bm25_corpus_map = list(all_chunks)
        tokenized_corpus = [self._tokenize(chk["text"]) for chk in self._bm25_corpus_map]
        self._bm25_index = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        raw_query_vector: np.ndarray,
        top_k: int = 3,
        use_encrypted_vault: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes genuine dual-channel hybrid retrieval across the entire corpus.
        """
        all_chunks = self.vector_manager._metadata_store
        total_docs = len(all_chunks)
        if total_docs == 0:
            return []

        # Refresh BM25 index if corpus size changed
        if self._bm25_index is None or len(self._bm25_corpus_map) != total_docs:
            self.update_bm25_index()

        rrf_scores: Dict[str, float] = {}
        candidate_meta: Dict[str, Dict[str, Any]] = {}

        for chk in all_chunks:
            candidate_meta[chk["chunk_id"]] = chk

        # -------------------------------------------------------------
        # Channel 1: Secure Dense Semantic Search (Rotated Space)
        # -------------------------------------------------------------
        if use_encrypted_vault:
            search_vector = self.vault.encrypt_vector(raw_query_vector)
        else:
            search_vector = raw_query_vector

        # Retrieve dense ranking over the full corpus
        dense_candidates = self.vector_manager.search(search_vector, top_k=total_docs)
        for rank, (meta, dense_sim) in enumerate(dense_candidates, start=1):
            chk_id = meta["chunk_id"]
            rrf_scores[chk_id] = rrf_scores.get(chk_id, 0.0) + (1.0 / (self.rrf_constant + rank))

        # -------------------------------------------------------------
        # Channel 2: Genuine Inverted Index Okapi BM25 Ranking
        # -------------------------------------------------------------
        tokenized_query = self._tokenize(query)
        if self._bm25_index and tokenized_query:
            bm25_scores = self._bm25_index.get_scores(tokenized_query)
            # Rank all documents by BM25 score descending
            bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
            for rank, idx in enumerate(bm25_ranked_indices, start=1):
                # Only attribute score if document has non-zero BM25 match
                if bm25_scores[idx] > 0.0:
                    chk_id = self._bm25_corpus_map[idx]["chunk_id"]
                    rrf_scores[chk_id] = rrf_scores.get(chk_id, 0.0) + (1.0 / (self.rrf_constant + rank))

        # -------------------------------------------------------------
        # Rank Aggregation via Reciprocal Rank Fusion (RRF)
        # -------------------------------------------------------------
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
