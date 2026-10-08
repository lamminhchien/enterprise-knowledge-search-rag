# Enterprise Knowledge Search & RAG Framework
### Secure Ephemeral-Vaulted Semantic Information Retrieval & AI Governance

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS%20Cosine-blue.svg?style=flat)](https://github.com/facebookresearch/faiss)
[![PySpark](https://img.shields.io/badge/BigData-PySpark%20ETL-E25A1C.svg?style=flat&logo=apachespark)](https://spark.apache.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Privacy-Guard](https://img.shields.io/badge/Security-One--Way%20Osmosis-success.svg?style=flat)]()
[![License-MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

An end-to-end, enterprise-grade **Semantic Search and Retrieval-Augmented Generation (RAG)** system engineered for corporate knowledge governance (M365/SaaS documents). Integrates an **Inbound One-Way Privacy Osmosis Guard** for PII sanitization, an **Ephemeral Orthogonal Cache Vault** for blinded cosine search, and a **PySpark-inspired distributed MapReduce batch pipeline**.

---

## 📌 Problem Statement

In enterprise environments (such as SharePoint, Confluence, and internal wikis), deploying Generative AI and search engines faces two existential hurdles:
1. **PII & Confidential Data Leakage:** Prompts containing employee/customer sensitive records (Citizen IDs, phone numbers, salary contracts) entering untrusted vector stores or external LLM context caches.
2. **Context Memory Vulnerability:** Storing unencrypted plaintext representations on cloud servers exposes enterprises to memory scraping, prompt injection, and unauthorized data extraction.

This framework implements **Security-by-Design Information Retrieval**, allowing high-speed semantic search while preserving strict mathematical confidentiality.

---

## 🏛️ System Architecture

```text
       [ Enterprise Raw Documents (.md, .txt, .pdf) ]
                             │
            (Distributed PySpark MapReduce ETL)
            - Distributed Word Count & Tokenization
            - Sliding-Window Semantic Chunking
                             │
                             ▼
            [ Inbound Privacy Osmosis Membrane ]
            - Regex & NER PII Scanning (IDs, Emails, Phones)
            - Irreversible Salted HMAC-SHA256 Tokenization
                             │
                             ▼
               [ Scrubbed Semantic Chunks ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ PyTorch Dense Embedder ]       [ Keyword Lexical Index ]
   - 384D Unit-Normalized Vectors   - BM25 Token Frequency Map
            │                                 │
            ▼                                 │
   [ Ephemeral Orthogonal Vault ]             │
   - Secret Rotation: v' = Q * v              │
   - Exact Cosine Isometry: <Qu, Qv> = <u, v> │
   - Session Half-Life (TTL Expiry)           │
            │                                 │
            ▼                                 │
   [ Blinded FAISS Vector Store ]             │
   - IndexFlatIP / Inner Product              │
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                 [ Hybrid Retriever (RRF) ]
                 - Reciprocal Rank Fusion
                 - Sub-millisecond Latency
                             │
                             ▼
                 [ Grounded RAG Generator ]
                 - Factual Synthesis
                 - Source Citations & Provenance
                 - Outbound RBAC Rehydration
```

---

## 🔬 Core Innovations

### 1. Inbound One-Way Privacy Osmosis Membrane
* Automatically scrubs sensitive PII (Citizen IDs, Emails, Phone numbers, API keys) prior to vectorization.
* Replaces entities with deterministic cryptographic tokens (`[NATIONAL_ID_ID_A1D615D1]`).
* Reversible lookup table is preserved exclusively in client-side ephemeral memory, ensuring the cloud vector index never stores plaintext PII.

### 2. Ephemeral Orthogonal Cache Vault (Rotation-Based Embedding Obfuscation)
* Derives a secret orthogonal transformation matrix $Q \in \mathbb{R}^{d \times d}$ via QR decomposition ($Q^T Q = I$).
* Transforms embeddings: $\vec{v}' = Q \vec{v}$.
* **Mathematical Isometry Guarantee:**
  $$\langle Q\vec{u}, Q\vec{v} \rangle = \vec{u}^T Q^T Q \vec{v} = \vec{u}^T I \vec{v} = \langle \vec{u}, \vec{v} \rangle$$
* The server conducts exact nearest-neighbor searches in rotated coordinate space without direct exposure of canonical coordinates.
* **Tamper-Evident Expiry & Decorrelation:** If session TTL expires, vectors collapse to uncorrelated Gaussian white noise $\mathcal{N}(0, I)$ destroying semantic retrieval.

### 3. Distributed PySpark Ingestion Pipeline
* Implements batch MapReduce text preprocessing and token profiling across document repositories, with local resilient fallback.

### 4. Hybrid Lexical-Semantic Retrieval with RRF
* Fuses dense vector similarity with lexical keyword overlap using Reciprocal Rank Fusion:
  $$\mathrm{RRF}(d) = \sum_{m \in \{\mathrm{Dense},\, \mathrm{Lexical}\}} \frac{1}{60 + \mathrm{rank}_m(d)}$$

---

## 🛡️ Threat Model & Limitations

Understanding the operational boundary of rotation-based obfuscation is critical:

* **What it Protects Against (Passive Server Dumps):**
  * Prevents direct linguistic/semantic coordinate reconstruction if the cloud vector database memory is scraped without the rotation key $Q$.
  * Removes cleartext PII (IDs, Phones, Emails) from all cloud embeddings and prompt histories via client-side HMAC tokenization.
* **What it Does NOT Protect Against (Cryptographic Boundaries):**
  * **Preserved Geometric Geometry:** Because all inner products are mathematically identical, the geometric cluster topology of the embeddings remains intact.
  * **Known-Plaintext / Procrustes Attacks:** An adversary with access to $\ge d$ known plaintext-embedding pairs $(v_i, Q v_i)$ can reconstruct $Q$ via Orthogonal Procrustes analysis.
  * **Scope:** This architecture serves as an **in-memory obfuscation and defense-in-depth layer**, not semantic-preserving Homomorphic Encryption.

---

## 📊 Quantitative Benchmarks (Information Retrieval)

Measured over multi-query compliance benchmark evaluation:

| Metric | Measured Score | Specification Target |
| :--- | :---: | :---: |
| **Mean Reciprocal Rank (MRR@5)** | **0.8889** | $> 0.80$ |
| **Precision@1 (Top-1 Accuracy)** | **85.0%** | $> 80.0\%$ |
| **Mean Query Latency** | **< 1.0 ms** | $< 35.0\text{ ms}$ |
| **P95 Query Latency** | **< 2.5 ms** | $< 50.0\text{ ms}$ |
| **Cosine Preservation Error ($| \Delta |$)** | **$< 10^{-7}$** | Strictly Zero |
| **Post-Expiry Decorrelation ($| \text{Sim} |$)** | **$< 0.05$** | Orthogonal Noise |

---

## 🚀 Quickstart & Master Demonstration

### 1. Clone & Setup
```bash
git clone https://github.com/lamminhchien/enterprise-knowledge-search-rag.git
cd enterprise-knowledge-search-rag
pip install -r requirements.txt
```

### 2. Run 1-Click Master Demonstration
Executes document loading, PySpark ETL, PII scrubbing, encrypted indexing, hybrid search, RAG synthesis, and tamper verification:
```bash
python demo_search.py
```

### 3. Launch FastAPI REST Service
```bash
uvicorn api.app_fastapi:app --reload --port 8000
```
* **API Documentation:** `http://localhost:8000/docs`
* **Search Endpoint:** `POST /api/v1/search`
* **RAG Endpoint:** `POST /api/v1/ask`

---

## 📁 Repository Structure

```text
enterprise-knowledge-search-rag/
├── data/
│   └── raw_documents/                 # Sample corporate policies (IT SOP, Data Governance, AI Policy)
├── src/
│   ├── data_ingestion/
│   │   ├── document_loader.py         # File loader with provenance
│   │   ├── text_chunker.py            # Sliding-window semantic chunker
│   │   └── pyspark_batch_etl.py       # Distributed MapReduce batch pipeline
│   ├── security/
│   │   ├── pii_osmosis_guard.py       # One-way privacy osmosis membrane (HMAC-SHA256)
│   │   └── ephemeral_vault.py         # Ephemeral orthogonal vault with TTL & noise trap
│   ├── embeddings/
│   │   └── dense_embedder.py          # PyTorch unit-normalized 384D embedder
│   ├── vector_store/
│   │   └── faiss_index_manager.py     # FAISS / PyTorch inner product index
│   ├── retriever/
│   │   └── hybrid_retriever.py        # Dense + Lexical Reciprocal Rank Fusion
│   ├── generator/
│   │   └── rag_engine.py              # Grounded RAG with source citations
│   └── evaluation/
│       └── ir_evaluator.py            # IR benchmark (MRR, Precision@K, Latency)
├── api/
│   └── app_fastapi.py                 # FastAPI service endpoints
├── demo_search.py                     # Master end-to-end runnable demonstration
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚖️ License
This project is licensed under the MIT License.
