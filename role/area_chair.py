import json
from typing import Dict, Any

from llm.base import LLMProvider
from .utils import AC_RESPONSE_SCHEMA


class AreaChairAgent:
    """
    Area Chair agent responsible for final Accept/Reject and review-quality scoring.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.llm_model = provider.model_name
        self.system_prompt = self._load_ac_prompt()
        self.review_schema = AC_RESPONSE_SCHEMA

    def _load_ac_prompt(self) -> str:
        file_path = "prompt/ac.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return (
                "You are an Area Chair (AC) responsible for making final decisions on papers. "
                "Evaluate all provided reviewer submissions and generate JSON output."
            )

    def _prepare_reviewer_data(self, final_reviews: Dict[int, Dict[str, Any]]) -> str:
        data_lines = []
        for r_id, review_data in final_reviews.items():
            short_id = f"R{r_id:02d}"
            review_block = f"--- Reviewer {short_id} ---\n"
            review_block += f"Final Rating: {review_data.get('final_rating', 'N/A')}\n"
            review_block += "Review Text (JSON):\n"
            review_block += f"{review_data.get('final_review_text', 'No text provided.')}\n"
            review_block += "-------------------------"
            data_lines.append(review_block)
        return "\n".join(data_lines)

    def make_final_decision(self, paper_title: str, final_reviews: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        reviewer_data_string = self._prepare_reviewer_data(final_reviews)
        user_prompt = (
            f"Paper Title: {paper_title}\n\n"
            f"Task: Make a final decision (Accept/Reject) on this paper.\n"
            f"You must also evaluate the quality of each individual review on a scale of 0 to 10 (even numbers only).\n\n"
            f"--- All Review Data ---\n"
            f"{reviewer_data_string}\n"
            f"----------------------------------------"
        )

        decision_data = self.provider.generate_json(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_schema=self.review_schema,
        )

        for eval_item in decision_data.get('review_evaluation', []):
            eval_item['score'] = int(eval_item['score']) if not isinstance(eval_item['score'], int) else eval_item['score']

        if decision_data.get('final_decision') not in ["Accept", "Reject"]:
            raise ValueError(f"AC agent returned invalid final_decision: {decision_data.get('final_decision')}")

        return decision_data
