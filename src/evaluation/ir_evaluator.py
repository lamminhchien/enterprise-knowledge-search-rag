"""
Information Retrieval (IR) Evaluation Suite.
Computes quantitative benchmarks: Mean Reciprocal Rank (MRR@K), Precision@K, Recall@K,
and end-to-end Query Latency.
"""

from typing import List, Dict, Tuple
import time
import numpy as np


class IREvaluator:
    """
    Rigorously benchmarks retrieval accuracy and latency under strict ground truth test cases.
    """

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5]

    def evaluate_retrieval_benchmark(
        self,
        retriever,
        embedder,
        test_queries: List[Dict[str, any]]
    ) -> Dict[str, float]:
        """
        Runs evaluation over test benchmark queries.
        test_queries format:
        [
            {
                "query": "What are the rules for multi-factor authentication?",
                "expected_doc_id": "it_security_sop"
            },
            ...
        ]
        """
        reciprocal_ranks = []
        precision_at_k = {k: [] for k in self.k_values}
        latencies_ms = []

        for item in test_queries:
            q_text = item["query"]
            target_doc = item["expected_doc_id"]

            t0 = time.perf_counter()
            q_vec = embedder.embed_texts(q_text)[0]
            retrieved = retriever.retrieve(q_text, q_vec, top_k=max(self.k_values))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            # Measure Rank of first relevant document
            hit_rank = None
            retrieved_doc_ids = [r["doc_id"] for r in retrieved]
            for rank_idx, doc_id in enumerate(retrieved_doc_ids, start=1):
                if doc_id == target_doc:
                    hit_rank = rank_idx
                    break

            # MRR
            rr = 1.0 / hit_rank if hit_rank is not None else 0.0
            reciprocal_ranks.append(rr)

            # Precision@K
            for k in self.k_values:
                top_k_ids = retrieved_doc_ids[:k]
                p_k = 1.0 if target_doc in top_k_ids else 0.0
                precision_at_k[k].append(p_k)

        mrr_score = float(np.mean(reciprocal_ranks))
        avg_latency = float(np.mean(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))

        results = {
            "mrr": round(mrr_score, 4),
            "mean_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "total_benchmark_queries": len(test_queries),
        }
        for k in self.k_values:
            results[f"precision@{k}"] = round(float(np.mean(precision_at_k[k])), 4)

        return results
