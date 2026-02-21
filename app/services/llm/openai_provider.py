"""
app/services/llm/openai_provider.py

OpenAI (GPT) provider — supports 'chunks' and 'images' send modes.
"""
import json
from typing import Any

from openai import AsyncOpenAI

from app.services.llm.base import BaseLLMProvider
from app.utils.config import settings
from app.utils.logger import logger


class OpenAIProvider(BaseLLMProvider):
    """
    Calls the OpenAI Chat Completions API with vision (image_url) support.

    Supported modes: chunks, images
    Not supported:   pdf  (raises NotImplementedError via base class)
    """

    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=600.0,  # 10-min ceiling for large vision tasks
        )

    # ------------------------------------------------------------------ public

    async def call_chunks(
        self,
        pages_b64: list[str],
        prompt: str,
        system_prompt: str,
        bank_name: str,
    ) -> list[dict]:
        """Send each page as a separate request; merge results."""
        all_transactions: list[dict] = []
        for page_idx, b64 in enumerate(pages_b64):
            logger.info(
                f"[{bank_name}] OpenAI chunks: page {page_idx + 1}/{len(pages_b64)}"
            )
            content = self._build_content(prompt, [b64])
            raw = await self._call(content, system_prompt, bank_name)
            txs = self._parse_transactions(raw, bank_name)
            logger.info(
                f"[{bank_name}] Page {page_idx + 1}: extracted {len(txs)} transactions"
            )
            all_transactions.extend(txs)
        return all_transactions

    async def call_images(
        self,
        pages_b64: list[str],
        prompt: str,
        system_prompt: str,
        bank_name: str,
    ) -> list[dict]:
        """Send all pages in a single request."""
        logger.info(
            f"[{bank_name}] OpenAI images: sending {len(pages_b64)} pages in one request"
        )
        content = self._build_content(prompt, pages_b64)
        raw = await self._call(content, system_prompt, bank_name)
        return self._parse_transactions(raw, bank_name)

    # ----------------------------------------------------------------- private

    def _build_content(self, prompt: str, pages_b64: list[str]) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in pages_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                }
            )
        return content

    async def _call(
        self, user_content: Any, system_prompt: str, bank_name: str
    ) -> str:
        self._log_payload(user_content, system_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=16384,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        logger.info(f"[{bank_name}] OpenAI response received | usage={response.usage}")
        if response.choices[0].finish_reason == "length":
            raise RuntimeError(
                f"[{bank_name}] OpenAI response truncated (max_tokens reached). "
                "Try chunks mode or a larger model."
            )
        logger.debug(f"OPENAI RESPONSE BODY: {result}")
        return result

    def _log_payload(self, user_content: Any, system_prompt: str):
        try:
            log_parts = []
            if isinstance(user_content, list):
                for part in user_content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        log_parts.append(
                            {"type": "image_url", "url": f"{url[:50]}…[{len(url)} chars]"}
                        )
                    else:
                        log_parts.append(part)
            else:
                log_parts = [{"type": "text", "text": str(user_content)}]

            logger.debug(
                f"LLM REQUEST (OPENAI / {self.model}):\n"
                + json.dumps(
                    {"model": self.model, "system": system_prompt, "user": log_parts},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to log OpenAI payload: {exc}")
