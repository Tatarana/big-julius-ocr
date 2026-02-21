"""
app/services/llm/google_provider.py

Google Gemini provider — supports 'chunks', 'images', and 'pdf' send modes.
"""
import asyncio
import json
import tempfile
import os
from typing import Any

import google.generativeai as genai

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
        genai.configure(api_key=settings.GOOGLE_API_KEY)

    # ----------------------------------------------------------------- helpers

    def _make_model(self, system_prompt: str) -> genai.GenerativeModel:
        return genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
        )

    @property
    def _gen_config(self) -> genai.types.GenerationConfig:
        return genai.types.GenerationConfig(
            temperature=0,
            max_output_tokens=50000,
            response_mime_type="application/json",
        )

    def _check_truncation(self, response, bank_name: str):
        """Raise if Gemini hit MAX_TOKENS."""
        candidate = response.candidates[0]
        if hasattr(candidate, "finish_reason") and (
            candidate.finish_reason == 2
            or getattr(candidate.finish_reason, "name", "") == "MAX_TOKENS"
        ):
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
        gemini_model = self._make_model(system_prompt)

        for page_idx, b64 in enumerate(pages_b64):
            logger.info(
                f"[{bank_name}] Gemini chunks: page {page_idx + 1}/{len(pages_b64)}"
            )
            content = [prompt, {"mime_type": "image/jpeg", "data": b64}]
            self._log_payload(content, system_prompt)

            response = await asyncio.to_thread(
                gemini_model.generate_content,
                content,
                generation_config=self._gen_config,
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
        gemini_model = self._make_model(system_prompt)

        content: list[Any] = [prompt]
        for b64 in pages_b64:
            content.append({"mime_type": "image/jpeg", "data": b64})

        self._log_payload(content, system_prompt)

        response = await asyncio.to_thread(
            gemini_model.generate_content,
            content,
            generation_config=self._gen_config,
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

            # Upload (blocking; run in thread to keep async)
            uploaded_file = await asyncio.to_thread(
                genai.upload_file,
                path=tmp_path,
                mime_type="application/pdf",
                display_name=f"{bank_name}_statement.pdf",
            )
            logger.info(
                f"[{bank_name}] Gemini file uploaded: {uploaded_file.name} ({uploaded_file.uri})"
            )

            gemini_model = self._make_model(system_prompt)
            content = [uploaded_file, prompt]

            self._log_payload(
                [{"type": "file", "name": uploaded_file.name, "uri": uploaded_file.uri}, {"type": "text", "text": prompt}],
                system_prompt,
            )

            response = await asyncio.to_thread(
                gemini_model.generate_content,
                content,
                generation_config=self._gen_config,
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
                    await asyncio.to_thread(genai.delete_file, uploaded_file.name)
                    logger.info(f"[{bank_name}] Deleted uploaded Gemini file: {uploaded_file.name}")
                except Exception as del_err:
                    logger.warning(f"[{bank_name}] Could not delete Gemini file {uploaded_file.name}: {del_err}")

    # ----------------------------------------------------------------- private

    def _log_payload(self, content: Any, system_prompt: str):
        try:
            log_parts = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "data" in part:
                        log_parts.append({"type": "image", "data": "[HIDDEN BASE64]"})
                    elif isinstance(part, str):
                        log_parts.append({"type": "text", "text": part})
                    else:
                        log_parts.append(str(part))
            else:
                log_parts = [str(content)]

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
