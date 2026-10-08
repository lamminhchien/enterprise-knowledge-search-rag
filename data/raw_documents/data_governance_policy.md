# Global Data Governance and Privacy Policy (POL-DAT-2026)

## 1. Scope and Classification
This policy governs all customer personal data (Personally Identifiable Information - PII) and corporate confidential records handled across internal search engines, databases, and multi-tenant cloud storage.
- Tier 1 (Public): Public marketing materials and released whitepapers.
- Tier 2 (Internal): Internal technical documentation, non-sensitive tickets, and engineering roadmaps.
- Tier 3 (Strictly Confidential / PII): Customer names, National ID numbers (CCCD/SSN), passport numbers, financial bank accounts, customer contract pricing, and private encryption keys.

## 2. Ingestion & Privacy Redaction Mandate
Before any Tier 3 dataset is indexed into enterprise search engines or retrieval vector stores:
- Automated de-identification must replace real customer names and National IDs with salted cryptographic pseudonyms.
- Plaintext PII must never be stored in persistent vector databases or third-party Large Language Model context caches.
- Violation of data retention policies incurs severe regulatory penalties under GDPR and local privacy frameworks.

## 3. Data Retention and Shredding
- Customer transaction history is retained for a statutory period of 5 years.
- Upon contract termination, automated cryptographic erasure (cryptographic shredding) must be completed within 30 business days.
