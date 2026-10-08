"""
One-Way Privacy Osmosis Guard (Inbound Semi-Permeable Membrane).
Ensures sensitive PII (Citizen IDs, Emails, Phones, Credit Cards) is scrubbed
and replaced with irreversible cryptographic pseudonyms prior to indexing or prompting.
"""

from typing import Dict, Tuple, List, Optional
import re
import hashlib
import hmac


class PIIOsmosisGuard:
    """
    Acts as an inbound semi-permeable membrane:
    Permeates semantic intent while irreversibly blocking and tokenizing sensitive PII.
    """

    def __init__(self, salt_key: Optional[bytes] = None):
        # Ephemeral salt key for HMAC pseudonym generation (kept exclusively in memory)
        self.salt_key = salt_key or b"osmosis_ephemeral_client_salt_2026"
        self._local_vault_map: Dict[str, str] = {}  # Token -> Original (Kept local, never sent to cloud)

        # High-precision RegEx patterns for enterprise PII
        self.patterns = {
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "NATIONAL_ID": re.compile(r"\b(?:\d{9}|\d{12})\b"),  # 9-digit or 12-digit Citizen IDs (CCCD)
            "PHONE": re.compile(r"\b(?:\+84|0)(?:3|5|7|8|9)\d{8}\b"),  # Vietnamese Mobile Numbers
            "CREDIT_CARD": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),  # 16-digit Card Numbers
            "API_KEY": re.compile(r"\b(?:sk-[a-zA-Z0-9]{32,}|AKIA[0-9A-Z]{16})\b"),  # Secrets / Keys
        }

    def _generate_pseudonym(self, entity_type: str, raw_value: str) -> str:
        """Derives a deterministic, salted one-way token for the raw value."""
        digest = hmac.new(self.salt_key, raw_value.encode("utf-8"), hashlib.sha256).hexdigest()[:8].upper()
        return f"[{entity_type}_ID_{digest}]"

    def filter_and_tokenize(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Processes inbound text:
        Returns:
            sanitized_text: Safe string ready for embedding / LLM prompt.
            session_tokens: Local mapping of replaced entities.
        """
        sanitized_text = text
        detected_entities: Dict[str, str] = {}

        for entity_type, pattern in self.patterns.items():
            matches = list(pattern.finditer(sanitized_text))
            # Process in reverse order to preserve string indices
            for match in reversed(matches):
                raw_val = match.group(0)
                pseudo_token = self._generate_pseudonym(entity_type, raw_val)
                
                # Store in local vault
                self._local_vault_map[pseudo_token] = raw_val
                detected_entities[pseudo_token] = raw_val

                # Replace in text
                sanitized_text = (
                    sanitized_text[:match.start()] + pseudo_token + sanitized_text[match.end():]
                )

        return sanitized_text, detected_entities

    def rehydrate_context(self, response_text: str, user_authorized: bool = False) -> str:
        """
        Outbound rehydration:
        If user is authorized (RBAC check passed), restores original values.
        Otherwise, leaves sanitized tokens intact.
        """
        if not user_authorized:
            return response_text  # Maintain privacy membrane

        rehydrated = response_text
        for token, original in self._local_vault_map.items():
            rehydrated = rehydrated.replace(token, original)
        return rehydrated


if __name__ == "__main__":
    guard = PIIOsmosisGuard()
    raw_sample = "Customer Nguyen Van A (CCCD: 012345678901, Phone: 0905123456, Email: client@example.com) uploaded file."
    clean_text, tokens = guard.filter_and_tokenize(raw_sample)
    print("--- RAW TEXT ---")
    print(raw_sample)
    print("\n--- SANITIZED FOR AI (ONE-WAY OSMOSIS) ---")
    print(clean_text)
    print("\n--- REHYDRATED FOR AUTHORIZED USER ---")
    print(guard.rehydrate_context(clean_text, user_authorized=True))
