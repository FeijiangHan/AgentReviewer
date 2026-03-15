from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider


def create_provider(provider_name: str, api_key: str, model_name: str):
    normalized = provider_name.strip().lower()
    if normalized == 'gemini':
        return GeminiProvider(api_key=api_key, model_name=model_name)
    if normalized == 'openai':
        return OpenAIProvider(api_key=api_key, model_name=model_name)
    raise ValueError(f"Unsupported provider: {provider_name}")
