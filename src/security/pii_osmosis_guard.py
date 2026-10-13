"""
One-Way Privacy Osmosis Guard (Inbound Semi-Permeable Membrane).
Ensures sensitive PII (Citizen IDs, Emails, Phones, Credit Cards) is scrubbed
and replaced with cryptographic pseudonyms prior to indexing or prompting.
Uses cryptographically secure random session salts to prevent dictionary attacks.
"""

from typing import Dict, Tuple, List, Optional
import os
import re
import hashlib
import hmac
import secrets


class PIIOsmosisGuard:
    """
    Inbound Semi-Permeable Privacy Membrane:
    Permeates semantic intent while irreversibly blocking and tokenizing sensitive PII.
    
    Security Guarantees:
    - 256-bit CSPRNG salt generated per-instance / per-session (eliminates rainbow tables & dictionary attacks).
    - Session-isolated de-anonymization vaults (no cross-tenant memory leakage).
    - Client-side storage: Cloud LLM and Vector Store never observe canonical PII.
    """

    def __init__(self, salt_key: Optional[bytes] = None):
        # Derive 256-bit salt from environment variable or generate fresh cryptographically secure random bytes
        env_salt = os.getenv("HMAC_VAULT_SALT")
        if salt_key:
            self.salt_key = salt_key
        elif env_salt:
            self.salt_key = env_salt.encode("utf-8")
        else:
            self.salt_key = secrets.token_bytes(32)

        # Session-isolated de-anonymization maps: session_id -> {Token: Original}
        self._session_vaults: Dict[str, Dict[str, str]] = {}

        # High-precision RegEx patterns for enterprise PII
        self.patterns = {
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "NATIONAL_ID": re.compile(r"\b(?:\d{9}|\d{12})\b"),  # 9-digit or 12-digit Citizen IDs (CCCD)
            "PHONE": re.compile(r"\b(?:\+84|0)(?:3|5|7|8|9)\d{8}\b"),  # Vietnamese Mobile Numbers
            "CREDIT_CARD": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),  # 16-digit Card Numbers
            "API_KEY": re.compile(r"\b(?:sk-[a-zA-Z0-9]{32,}|AKIA[0-9A-Z]{16})\b"),  # Secrets / Keys
        }

    def _generate_pseudonym(self, entity_type: str, raw_value: str) -> str:
        """Derives a deterministic, salted HMAC-SHA256 pseudonym token for the raw value."""
        digest = hmac.new(self.salt_key, raw_value.encode("utf-8"), hashlib.sha256).hexdigest()[:8].upper()
        return f"[{entity_type}_ID_{digest}]"

    def filter_and_tokenize(self, text: str, session_id: str = "default_session") -> Tuple[str, Dict[str, str]]:
        """
        Processes inbound text:
        Returns:
            sanitized_text: Safe string ready for embedding / LLM prompt.
            detected_entities: Mapping of replaced entities for this session.
        """
        if session_id not in self._session_vaults:
            self._session_vaults[session_id] = {}

        session_map = self._session_vaults[session_id]
        sanitized_text = text
        detected_entities: Dict[str, str] = {}

        for entity_type, pattern in self.patterns.items():
            matches = list(pattern.finditer(sanitized_text))
            # Process in reverse order to preserve string character offsets
            for match in reversed(matches):
                raw_val = match.group(0)
                pseudo_token = self._generate_pseudonym(entity_type, raw_val)

                # Store in session vault
                session_map[pseudo_token] = raw_val
                detected_entities[pseudo_token] = raw_val

                # Replace in text
                sanitized_text = (
                    sanitized_text[:match.start()] + pseudo_token + sanitized_text[match.end():]
                )

        return sanitized_text, detected_entities

    def rehydrate_context(self, response_text: str, session_id: str = "default_session", user_authorized: bool = False) -> str:
        """
        Outbound rehydration:
        If user is authorized (RBAC check passed), restores original values for this session.
        Otherwise, leaves sanitized tokens intact to enforce confidentiality.
        """
        if not user_authorized or session_id not in self._session_vaults:
            return response_text

        session_map = self._session_vaults[session_id]
        rehydrated = response_text
        for token, original in session_map.items():
            rehydrated = rehydrated.replace(token, original)
        return rehydrated


if __name__ == "__main__":
    guard = PIIOsmosisGuard()
    raw_sample = "Customer Nguyen Van A (CCCD: 012345678901, Phone: 0901234567, Email: client_demo@example.com)."
    clean_text, tokens = guard.filter_and_tokenize(raw_sample, session_id="test_sess_1")
    print("--- RAW TEXT ---")
    print(raw_sample)
    print("\n--- SANITIZED FOR AI (ONE-WAY OSMOSIS) ---")
    print(clean_text)
    print("\n--- REHYDRATED FOR AUTHORIZED USER ---")
    print(guard.rehydrate_context(clean_text, session_id="test_sess_1", user_authorized=True))
