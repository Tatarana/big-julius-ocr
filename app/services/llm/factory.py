"""
app/services/llm/factory.py

Creates and caches provider instances by (provider_key, model) pairs.
"""
from functools import lru_cache
from app.utils.logger import logger


@lru_cache(maxsize=None)
def get_provider(provider_key: str, model: str):
    """
    Return a cached BaseLLMProvider instance for the given provider + model.

    Args:
        provider_key: "openai" | "google"
        model:        model name string (e.g. "gpt-4o", "gemini-2.0-flash")

    Raises:
        ValueError: if provider_key is unknown
    """
    provider_key = provider_key.lower()

    if provider_key == "openai":
        from app.services.llm.openai_provider import OpenAIProvider  # noqa: PLC0415
        logger.info(f"Creating OpenAIProvider with model={model}")
        return OpenAIProvider(model=model)

    if provider_key == "google":
        from app.services.llm.google_provider import GeminiProvider  # noqa: PLC0415
        logger.info(f"Creating GeminiProvider with model={model}")
        return GeminiProvider(model=model)

    raise ValueError(
        f"Unknown LLM provider '{provider_key}'. Supported: 'openai', 'google'."
    )
