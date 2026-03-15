import json
from typing import Any, Dict, Optional

from .base import LLMProvider
from .json_utils import load_json_with_repair


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = 'gpt-4o-mini'):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required for OpenAIProvider") from exc

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return 'openai'

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_json(self, system_prompt: str, user_prompt: str, response_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Keep compatibility broad by using chat.completions json_object mode.
        last_error = None
        for _ in range(2):
            response = self._client.chat.completions.create(
                model=self._model_name,
                response_format={'type': 'json_object'},
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
            )
            content = response.choices[0].message.content
            try:
                return load_json_with_repair(content)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Failed to parse JSON response from OpenAI: {last_error}")

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
