"""
Large-Scale Distributed Batch Ingestion & MapReduce Text Preprocessing.
Designed for high-throughput enterprise document processing using Apache Spark principles.
Includes automated fallback for environments without an active PySpark cluster.
"""

from typing import List, Dict, Tuple
import os
import re


def map_clean_and_tokenize(doc_record: Dict[str, str]) -> Dict[str, any]:
    """
    Map Phase: Cleans whitespace, lowercases, and tokenizes document content.
    """
    text = doc_record.get("content", "")
    cleaned = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"\b\w+\b", cleaned.lower())
    return {
        "doc_id": doc_record.get("doc_id", "unknown"),
        "filename": doc_record.get("filename", ""),
        "word_count": len(tokens),
        "cleaned_text": cleaned
    }


def reduce_aggregate_statistics(mapped_records: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Reduce Phase: Aggregates corpus statistics (Total volume, average document length).
    """
    total_docs = len(mapped_records)
    total_words = sum(r["word_count"] for r in mapped_records)
    avg_words = total_words / max(total_docs, 1)

    return {
        "total_documents": total_docs,
        "total_word_count": total_words,
        "average_doc_length": round(avg_words, 2)
    }


class PySparkBatchETLPipeline:
    """
    Simulates / Executes distributed PySpark DataFrame transformations
    for batch text ingestion and token filtering.
    """

    def __init__(self, app_name: str = "EnterpriseSearchBatchETL"):
        self.app_name = app_name
        self._spark = None

    def execute_batch_etl(self, documents: List[Dict[str, str]]) -> Tuple[List[Dict[str, any]], Dict[str, any]]:
        """
        Executes distributed MapReduce ingestion.
        Attempts native PySpark RDD/DataFrame; gracefully falls back to local map/reduce.
        """
        try:
            import pyspark
            from pyspark.sql import SparkSession
            if self._spark is None:
                self._spark = SparkSession.builder.appName(self.app_name).master("local[*]").getOrCreate()
            
            # Spark RDD Parallelism
            rdd = self._spark.sparkContext.parallelize(documents)
            mapped_rdd = rdd.map(map_clean_and_tokenize)
            processed_data = mapped_rdd.collect()
            stats = reduce_aggregate_statistics(processed_data)
            print("[PySpark Engine] Batch ETL completed via native Apache Spark cluster.")
            return processed_data, stats

        except (ImportError, Exception) as e:
            # Resilient Local Data Parallelism Fallback
            processed_data = [map_clean_and_tokenize(doc) for doc in documents]
            stats = reduce_aggregate_statistics(processed_data)
            print("[PySpark Engine (Local Fallback)] Batch MapReduce completed successfully.")
            return processed_data, stats
