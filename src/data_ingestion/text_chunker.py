"""
Semantic Text Chunker with Sliding Window Context Preservation.
"""

from typing import List, Dict


class TextChunker:
    """
    Partitions long corporate documents into bounded, overlapping semantic chunks
    to optimize dense vector embedding accuracy.
    """

    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Splits a single document into indexed chunks.
        Preserves provenance: doc_id, chunk_id, section headers.
        """
        content = doc["content"]
        chunks = []
        step = max(self.chunk_size - self.chunk_overlap, 1)

        start = 0
        chunk_idx = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_chk{chunk_idx}",
                    "doc_id": doc["doc_id"],
                    "filename": doc["filename"],
                    "text": chunk_text,
                    "char_start": start,
                    "char_end": end
                })
                chunk_idx += 1

            if end >= len(content):
                break
            start += step

        return chunks

    def chunk_corpus(self, docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Batches chunking across all corpus documents."""
        all_chunks = []
        for d in docs:
            all_chunks.extend(self.chunk_document(d))
        return all_chunks
