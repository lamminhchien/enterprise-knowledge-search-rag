"""
Document Ingestion & File Parsing Utility.
Loads and cleans corporate knowledge files (.md, .txt) with metadata tracking.
"""

from typing import List, Dict
import os
import glob


class DocumentLoader:
    """Loads documents from raw storage directory with file-level provenance."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_documents(self) -> List[Dict[str, str]]:
        """
        Scans data_dir for markdown and text files.
        Returns list of dicts: {'doc_id': str, 'filename': str, 'content': str}
        """
        documents = []
        filepaths = glob.glob(os.path.join(self.data_dir, "**", "*.md"), recursive=True) + \
                    glob.glob(os.path.join(self.data_dir, "**", "*.txt"), recursive=True)

        for path in filepaths:
            filename = os.path.basename(path)
            doc_id = os.path.splitext(filename)[0]
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    documents.append({
                        "doc_id": doc_id,
                        "filename": filename,
                        "filepath": path,
                        "content": content
                    })
            except Exception as e:
                print(f"Warning: Failed reading {path}: {e}")

        return documents
