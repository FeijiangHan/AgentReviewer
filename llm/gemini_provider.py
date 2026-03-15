import json
from typing import Any, Dict, Optional

from .base import LLMProvider
from .json_utils import load_json_with_repair


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-flash'):
        from google import genai

        self._genai = genai
        self._model_name = model_name
        self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return 'gemini'

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_json(self, system_prompt: str, user_prompt: str, response_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config_kwargs: Dict[str, Any] = {
            'system_instruction': system_prompt,
            'response_mime_type': 'application/json',
        }
        if response_schema is not None:
            config_kwargs['response_schema'] = response_schema

        generation_config = self._genai.types.GenerateContentConfig(**config_kwargs)
        last_error = None
        for _ in range(2):
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[user_prompt],
                config=generation_config,
            )
            try:
                return load_json_with_repair(response.text)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to parse JSON response from Gemini: {last_error}")

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        generation_config = self._genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[user_prompt],
            config=generation_config,
        )
        return response.text.strip()
