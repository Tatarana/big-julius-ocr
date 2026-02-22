"""
app/services/llm/google_provider.py

Google Gemini provider — supports 'chunks', 'images', and 'pdf' send modes.
Using the new unified google-genai SDK with native async methods (client.aio.*).
"""
import tempfile
import os
from typing import Any

import httpx

from google import genai
from google.genai import types

from app.services.llm.base import BaseLLMProvider
from app.utils.config import settings
from app.utils.logger import logger


class GeminiProvider(BaseLLMProvider):
    """
    Calls the Google Gemini API.

    Supported modes: chunks, images, pdf
    """

    def __init__(self, model: str):
        self.model = model
        # Use native async methods (client.aio.*) to guarantee the httpx async
        # transport is used. The asyncio.to_thread approach called the sync
        # client whose timeout path is separate and unreliable.
        #
        # We pass an explicit httpx_async_client so the SDK never falls back
        # to aiohttp (which has its own 300-second default read timeout and
        # ignores HttpOptions.timeout entirely).
        self.client = genai.Client(
            api_key=settings.GOOGLE_API_KEY,
            http_options=types.HttpOptions(
                timeout=600000,  # ms — passed to httpx per-request
                httpx_async_client=httpx.AsyncClient(timeout=600.0),
            ),
        )

    # ----------------------------------------------------------------- helpers

    def _get_config(self, system_prompt: str) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            max_output_tokens=50000,
            response_mime_type="application/json",
        )

    def _check_truncation(self, response, bank_name: str):
        """Raise if Gemini hit MAX_TOKENS."""
        if not response.candidates:
            return

        candidate = response.candidates[0]
        if candidate.finish_reason == "MAX_TOKENS":
            raise RuntimeError(
                f"[{bank_name}] Gemini response truncated (MAX_TOKENS). "
                "Try chunks mode or a model with higher output token limit."
            )

    # ------------------------------------------------------------------ public

    async def call_chunks(
        self,
        pages_b64: list[str],
        prompt: str,
        system_prompt: str,
        bank_name: str,
    ) -> list[dict]:
        """Send each page image in a separate request; merge results."""
        all_transactions: list[dict] = []
        config = self._get_config(system_prompt)

        for page_idx, b64 in enumerate(pages_b64):
            logger.info(
                f"[{bank_name}] Gemini chunks: page {page_idx + 1}/{len(pages_b64)}"
            )
            content = [
                types.Part.from_bytes(data=b64, mime_type="image/jpeg"),
                prompt
            ]
            self._log_payload(content, system_prompt)

            # Use native async API (client.aio.models) — respects httpx timeout
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=content,
                config=config,
            )
            self._check_truncation(response, bank_name)
            raw = response.text
            logger.debug(f"GEMINI CHUNK RESPONSE (page {page_idx+1}): {raw}")
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
            f"[{bank_name}] Gemini images: sending {len(pages_b64)} pages in one request"
        )
        config = self._get_config(system_prompt)

        content: list[Any] = [prompt]
        for b64 in pages_b64:
            content.append(types.Part.from_bytes(data=b64, mime_type="image/jpeg"))

        self._log_payload(content, system_prompt)

        # Use native async API (client.aio.models) — respects httpx timeout
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=content,
            config=config,
        )
        self._check_truncation(response, bank_name)
        raw = response.text
        logger.debug(f"GEMINI IMAGES RESPONSE: {raw}")
        return self._parse_transactions(raw, bank_name)

    async def call_pdf(
        self,
        pdf_bytes: bytes,
        prompt: str,
        system_prompt: str,
        bank_name: str,
    ) -> list[dict]:
        """
        Upload PDF via Gemini File API, then call generate_content with the file URI.
        The uploaded file is deleted after the call to avoid accumulation.
        """
        logger.info(f"[{bank_name}] Gemini pdf: uploading PDF ({len(pdf_bytes)} bytes)")

        # Write to a temp file so the File API can upload it
        tmp_path = None
        uploaded_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # Use native async API (client.aio.files) — respects httpx timeout
            uploaded_file = await self.client.aio.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(
                    mime_type="application/pdf",
                    display_name=f"{bank_name}_statement.pdf",
                )
            )
            logger.info(
                f"[{bank_name}] Gemini file uploaded: {uploaded_file.name} ({uploaded_file.uri})"
            )

            config = self._get_config(system_prompt)
            content = [uploaded_file, prompt]

            self._log_payload(
                [{"type": "file", "name": uploaded_file.name, "uri": uploaded_file.uri}, {"type": "text", "text": prompt}],
                system_prompt,
            )

            # Use native async API (client.aio.models) — respects httpx timeout
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=content,
                config=config,
            )
            self._check_truncation(response, bank_name)
            raw = response.text
            logger.debug(f"GEMINI PDF RESPONSE: {raw}")
            return self._parse_transactions(raw, bank_name)

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            # Delete from Gemini File API to avoid quota build-up
            if uploaded_file:
                try:
                    await self.client.aio.files.delete(name=uploaded_file.name)
                    logger.info(f"[{bank_name}] Deleted uploaded Gemini file: {uploaded_file.name}")
                except Exception as del_err:
                    logger.warning(f"[{bank_name}] Could not delete Gemini file {uploaded_file.name}: {del_err}")

    # ----------------------------------------------------------------- private

    def _log_payload(self, content: Any, system_prompt: str):
        try:
            log_parts = []
            if isinstance(content, list):
                for part in content:
                    if hasattr(part, "data") and part.data:
                        log_parts.append({"type": "image", "data": "[HIDDEN BYTES]"})
                    elif hasattr(part, "text") and part.text:
                        log_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, str):
                        log_parts.append({"type": "text", "text": part})
                    else:
                        log_parts.append(str(part))
            else:
                log_parts = [str(content)]

            import json
            logger.debug(
                f"LLM REQUEST (GEMINI / {self.model}):\n"
                + json.dumps(
                    {"model": self.model, "system": system_prompt, "user": log_parts},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to log Gemini payload: {exc}")
