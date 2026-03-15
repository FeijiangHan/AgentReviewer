from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str, response_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError
