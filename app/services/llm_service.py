"""
app/services/llm_service.py

Compatibility shim retained for:
  - /check-all-connections endpoint (check_connection)
  - Any external code that still imports llm_service

All actual LLM calls now go through app.services.llm (provider package).
"""
import google.generativeai as genai
from openai import AsyncOpenAI
from app.utils.config import settings
from app.utils.logger import logger


class LLMService:
    def __init__(self):
        self._openai_client = None
        if settings.OPENAI_API_KEY:
            self._openai_client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=600.0,
            )
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)

    async def check_connection(self) -> bool:
        """Health-check: returns True if at least one configured provider is reachable."""
        openai_ok = False
        if self._openai_client:
            try:
                await self._openai_client.models.list()
                openai_ok = True
            except Exception as e:
                logger.warning(f"OpenAI connection check failed: {e}")

        google_ok = False
        if settings.GOOGLE_API_KEY:
            try:
                genai.list_models()
                google_ok = True
            except Exception as e:
                logger.warning(f"Google connection check failed: {e}")

        return openai_ok or google_ok

    def categorize_transaction(self, description: str) -> str:
        return "Uncategorized"


llm_service = LLMService()
