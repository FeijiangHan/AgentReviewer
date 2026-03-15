import json
import re
from typing import Dict, Any, Optional

from llm.base import LLMProvider
from .utils import REVIEW_SCHEMA


class LLMAgentReviewer:
    """
    Individual reviewer agent with a fixed persona.
    """

    def __init__(self, reviewer_id: int, persona: str, provider: LLMProvider, initial_elo: float = 1500.0, persona_prompt: str | None = None):
        self.reviewer_id: int = reviewer_id
        self.persona: str = persona
        _ = initial_elo  # compatibility

        self._base_system_prompt: str = persona_prompt if persona_prompt else self._load_persona_prompt()
        self.review_policy_prompt: str = self._load_review_policy_prompt()
        self.provider: LLMProvider = provider
        self.llm_model: str = provider.model_name
        self.REVIEW_SCHEMA = REVIEW_SCHEMA

    def _load_persona_prompt(self) -> str:
        file_path = f"prompt/persona/{self.persona}.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_review_policy_prompt(self) -> str:
        file_path = "prompt/reviewer.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_current_system_prompt(self) -> str:
        return self._base_system_prompt

    def _parse_rating(self, review_text: str) -> Optional[int]:
        match = re.search(r"\bscore[:\s]+([0246810])\b", review_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _prepare_paper_content(self, paper_content: str, max_chars: int = 12000) -> str:
        """
        Minimal chunking strategy for long papers: keep head+tail slices.
        """
        if len(paper_content) <= max_chars:
            return paper_content
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        return (
            paper_content[:head_chars]
            + "\n\n[...CONTENT TRUNCATED FOR CONTEXT WINDOW...]\n\n"
            + paper_content[-tail_chars:]
        )


    def generate_review_stage_1(self, paper_content: str, paper_label: str = "") -> Dict[str, Any]:
        full_system_prompt = self._get_current_system_prompt()
        prepared_content = self._prepare_paper_content(paper_content)
        user_prompt = (
            f"Paper Source: {paper_label or 'N/A'}\n\n"
            f"Paper Content:\n{prepared_content}\n\n"
            f"You are conducting the first stage of the review process. "
            f"Generate your initial review following the required policy.\n\n"
            f"Review Policy:\n{self.review_policy_prompt}"
        )

        review_data = self.provider.generate_json(
            system_prompt=full_system_prompt,
            user_prompt=user_prompt,
            response_schema=self.REVIEW_SCHEMA,
        )
        initial_rating = review_data.get("score", None)

        return {
            "initial_rating": initial_rating,
            "initial_review_text": json.dumps(review_data, ensure_ascii=False),
            "system_prompt_used": full_system_prompt
        }

    def generate_review_stage_2(
        self,
        paper_content: str,
        initial_review: Dict[str, Any],
        other_reviews: Dict[int, Dict[str, Any]],
        paper_label: str = "",
    ) -> Dict[str, Any]:
        formatted_other_reviews = "\n\n--- Other Reviewers' Feedback ---\n"
        for r_id, review_data in other_reviews.items():
            formatted_other_reviews += (
                f"\nReviewer {r_id} (Score: {review_data['initial_rating']}):\n"
                f"Review Details: {review_data['initial_review_text']}\n"
            )
        formatted_other_reviews += "\n----------------------------------------\n"

        full_system_prompt = self._get_current_system_prompt()
        prepared_content = self._prepare_paper_content(paper_content)
        user_prompt = (
            f"Paper Source: {paper_label or 'N/A'}\n\n"
            f"Paper Content:\n{prepared_content}\n\n"
            f"You are conducting the second stage of the review process (Discussion/Rebuttal). "
            f"Your initial review and rating were: Score {initial_review['initial_rating']}.\n"
            f"Your Initial Review (JSON):\n{initial_review['initial_review_text']}\n\n"
            f"Review the paper in light of the other reviews and adjust if needed while staying aligned with your persona.\n"
            f"{formatted_other_reviews}\n"
            f"Review Policy:\n{self.review_policy_prompt}"
        )

        review_data = self.provider.generate_json(
            system_prompt=full_system_prompt,
            user_prompt=user_prompt,
            response_schema=self.REVIEW_SCHEMA,
        )
        final_rating = review_data.get("score", None)

        return {
            "final_rating": final_rating,
            "final_review_text": json.dumps(review_data, ensure_ascii=False),
            "system_prompt_used": full_system_prompt
        }
