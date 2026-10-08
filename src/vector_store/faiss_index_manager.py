"""
High-Performance Vector Index Manager.
Supports FAISS (IndexFlatIP) with pure PyTorch/NumPy accelerated matrix fallback.
Stores vector representations alongside document metadata.
"""

from typing import List, Dict, Tuple
import os
import json
import numpy as np
import torch


class VectorIndexManager:
    """
    Manages vector index storage and Top-K cosine similarity searches.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._faiss_index = None
        self._vector_store: List[np.ndarray] = []
        self._metadata_store: List[Dict[str, any]] = []

        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._has_faiss = True
            print("[VectorStore] Powered by native FAISS (IndexFlatIP).")
        except ImportError:
            self._has_faiss = False
            print("[VectorStore] Operating on PyTorch/NumPy Vectorized Cosine Engine.")

    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict[str, any]]) -> None:
        """
        Adds normalized vectors and metadata into the index.
        vectors: shape [N, dim]
        """
        assert len(vectors) == len(metadata_list), "Vector and metadata counts must match"
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self._has_faiss and self._faiss_index is not None:
            self._faiss_index.add(vectors)
        else:
            self._vector_store.append(vectors)

        self._metadata_store.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, any], float]]:
        """
        Searches top_k most similar items for a query vector.
        Returns: List of (metadata_dict, cosine_similarity_score)
        """
        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        total_items = len(self._metadata_store)
        if total_items == 0:
            return []

        top_k = min(top_k, total_items)

        if self._has_faiss and self._faiss_index is not None:
            distances, indices = self._faiss_index.search(query_vector, top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < total_items:
                    results.append((self._metadata_store[idx], float(dist)))
            return results

        # PyTorch / NumPy Vectorized Cosine Multiplication
        all_vecs = np.vstack(self._vector_store)  # Shape [Total, dim]
        scores = np.dot(all_vecs, query_vector.T).squeeze()  # [Total]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self._metadata_store[idx], float(scores[idx])))
        return results

    def save_index(self, directory: str) -> None:
        """Saves vector arrays and JSON metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata_store, f, indent=2, ensure_ascii=False)

        all_vecs = np.vstack(self._vector_store) if not self._has_faiss else None
        if all_vecs is not None:
            np.save(os.path.join(directory, "vectors.npy"), all_vecs)

    def load_index(self, directory: str) -> bool:
        """Loads index and metadata from disk."""
        meta_path = os.path.join(directory, "metadata.json")
        vec_path = os.path.join(directory, "vectors.npy")
        if not os.path.exists(meta_path):
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata_store = json.load(f)

        if os.path.exists(vec_path):
            loaded_vecs = np.load(vec_path)
            self._vector_store = [loaded_vecs]
        return True
