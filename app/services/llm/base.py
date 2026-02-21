"""
app/services/llm/base.py

Abstract base class for all LLM providers.
"""
from abc import ABC, abstractmethod
from app.utils.logger import logger


class BaseLLMProvider(ABC):
    """
    Abstract interface every LLM provider must implement.

    Three send modes:
      - chunks : PDF → one image per page → N separate LLM calls → results merged
      - images : PDF → all images → ONE LLM call
      - pdf    : raw PDF bytes → ONE LLM call (Gemini only)
    """

    @abstractmethod
    async def call_chunks(
        self,
        pages_b64: list[str],
        prompt: str,
        system_prompt: str,
        bank_name: str,
    ) -> list[dict]:
        """
        Sends each page image in a separate LLM request.
        Returns a flat list of transaction dicts (already parsed JSON objects).
        """
        ...

    @abstractmethod
    async def call_images(
        self,
        pages_b64: list[str],
        prompt: str,
        system_prompt: str,
        bank_name: str,
    ) -> list[dict]:
        """
        Sends all page images in a single LLM request.
        Returns a flat list of transaction dicts (already parsed JSON objects).
        """
        ...

    async def call_pdf(
        self,
        pdf_bytes: bytes,
        prompt: str,
        system_prompt: str,
        bank_name: str,
    ) -> list[dict]:
        """
        Sends the raw PDF in a single LLM request.
        Default implementation raises NotImplementedError.
        Override in providers that support native PDF input.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support 'pdf' send mode. "
            "Use send_mode='images' or 'chunks' instead."
        )

    # ------------------------------------------------------------------ helpers

    def _parse_transactions(self, raw_json: str, bank_name: str) -> list[dict]:
        """Parse JSON string returned by LLM into a list of transaction dicts."""
        import json

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            data = self._repair_and_parse(raw_json, bank_name)
            if data is None:
                return []

        # Tolerate various top-level shapes
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            txs = data.get("transactions")
            if txs is not None and isinstance(txs, list):
                return txs
            # Fall back: return the first list value found
            for val in data.values():
                if isinstance(val, list):
                    return val

        logger.warning(f"[{bank_name}] LLM response had unexpected shape; returning []")
        return []

    def _repair_and_parse(self, broken_json: str, bank_name: str):
        """Best-effort JSON repair for truncated responses."""
        import json

        logger.debug(f"[{bank_name}] Attempting JSON repair on broken response")
        last_comma = broken_json.rfind("},")
        if last_comma != -1:
            candidate = broken_json[: last_comma + 1] + "]}"
            try:
                d = json.loads(candidate)
                logger.warning(
                    f"[{bank_name}] ⚠️ REPAIR: Closed list after last complete object. "
                    "Extraction is PARTIAL – some transactions may be missing."
                )
                return d
            except Exception:
                pass

        for suffix in ("]}", "}", " ] } "):
            try:
                d = json.loads(broken_json + suffix)
                logger.warning(
                    f"[{bank_name}] ⚠️ REPAIR: Appended '{suffix}'. "
                    "Extraction is PARTIAL – some transactions may be missing."
                )
                return d
            except Exception:
                continue

        logger.error(f"[{bank_name}] ❌ Could not repair JSON; dropping response.")
        return None
