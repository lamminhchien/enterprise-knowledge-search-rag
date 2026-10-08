# Enterprise Artificial Intelligence & LLM Governance Guidelines (AI-GOV-2026)

## 1. Permitted Use of AI & Agents
Enterprise AI assistants, Copilot integrations, and automated RAG systems are permitted strictly for accelerating productivity, semantic document search, code refactoring, and draft summarization.
- All AI-generated responses must cite verifiable source documents.
- Unverifiable hallucinations must be flagged via automated confidence scoring.

## 2. Prohibition of Sensitive Data Ingestion
Employees and automated bots are strictly prohibited from submitting the following items into public or non-governed LLM prompts:
- Customer PII (Names, Citizen IDs, Payment credentials).
- Production encryption keys, JWT signing secrets, or database connection strings.
- Non-public financial results prior to formal earnings disclosure.

## 3. Defense Against Adversarial Prompts
All Retrieval-Augmented Generation (RAG) pipelines must implement:
- Input sanitization to prevent prompt injection and jailbreak payloads.
- Ephemeral context caching with automated session destruction to prevent cross-tenant memory leakage.
- Role-Based Access Control (RBAC) ensuring users only retrieve documents matching their security clearance.
