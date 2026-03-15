import re
import httpx
import io
import json
from typing import Dict, Any, Optional

from google import genai
from google.genai.errors import APIError

from .utils import REVIEW_SCHEMA


class LLMAgentReviewer:
    """
    Represents an individual LLM reviewer agent for the simulation.
    Handles two-stage review generation under a fixed persona.
    """

    def __init__(self, reviewer_id: int, persona: str, client: genai.Client, initial_elo: float = 1500.0):
        self.reviewer_id: int = reviewer_id
        self.persona: str = persona
        # `initial_elo` is intentionally ignored in Milestone A to keep constructor compatibility.
        _ = initial_elo

        self._base_system_prompt: str = self._load_persona_prompt()
        self.review_policy_prompt: str = self._load_review_policy_prompt()
        self.llm_client: genai.Client = client
        self.llm_model: str = 'gemini-2.5-flash'
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
        """
        Returns the current system prompt for the reviewer.
        """
        return self._base_system_prompt

    def _parse_rating(self, review_text: str) -> Optional[int]:
        """
        Extracts the numerical rating (0, 2, 4, 6, 8, 10) from review text.
        """
        match = re.search(r"\bscore[:\s]+([0246810])\b", review_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _upload_pdf_from_url(self, paper_pdf_url: str):
        """
        Fetches the PDF content from URL and uploads it to Gemini.
        """
        print(f"-> Reviewer {self.reviewer_id} fetching and uploading PDF from: {paper_pdf_url}")
        try:
            response = httpx.get(paper_pdf_url)
            response.raise_for_status()

            doc_io = io.BytesIO(response.content)

            uploaded_file = self.llm_client.files.upload(
                file=doc_io,
                config=dict(mime_type='application/pdf')
            )
            print(f"-> PDF uploaded successfully. File Name: {uploaded_file.name}")
            return uploaded_file
        except httpx.HTTPError as e:
            print(f"Error fetching PDF from URL: {e}")
            return None
        except APIError as e:
            print(f"Error uploading file to Gemini API: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during PDF upload: {e}")
            return None

    def _cleanup_file(self, uploaded_file):
        """
        Deletes an uploaded file from Gemini API service.
        """
        if uploaded_file:
            try:
                self.llm_client.files.delete(name=uploaded_file.name)
                print(f"-> Cleaned up uploaded file: {uploaded_file.name}")
            except APIError as e:
                print(f"Warning: Failed to delete uploaded file {uploaded_file.name}: {e}")

    def generate_review_stage_1(self, paper_pdf_url: str) -> Dict[str, Any]:
        """
        Stage 1: Generates an initial review and rating without seeing other reviews.
        """
        uploaded_file = self._upload_pdf_from_url(paper_pdf_url)
        if not uploaded_file:
            return {"initial_rating": None, "initial_review_text": "Error: Failed to process PDF.", "system_prompt_used": ""}

        full_system_prompt = self._get_current_system_prompt()

        user_prompt = (
            "You are conducting the **first stage** of the review process. "
            "Review the paper provided as context. "
            "Generate your initial review following the required policy.\n\n"
            f"Review Policy:\n{self.review_policy_prompt}"
        )

        print(f"-> Reviewer {self.reviewer_id} calling Gemini API for Stage 1...")

        generation_config = genai.types.GenerateContentConfig(
            system_instruction=full_system_prompt,
            response_mime_type="application/json",
            response_schema=self.REVIEW_SCHEMA
        )

        response = self.llm_client.models.generate_content(
            model=self.llm_model,
            contents=[uploaded_file, user_prompt],
            config=generation_config
        )

        review_data = json.loads(response.text)
        initial_rating = review_data.get("score", None)

        return {
            "initial_rating": initial_rating,
            "initial_review_text": response.text,
            "system_prompt_used": full_system_prompt
        }

    def generate_review_stage_2(self, paper_pdf_url: str, initial_review: Dict[str, Any], other_reviews: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stage 2: Generates an adjusted review and final rating after seeing others.
        """
        uploaded_file = self._upload_pdf_from_url(paper_pdf_url)
        if not uploaded_file:
            return {"final_rating": None, "final_review_text": "Error: Failed to process PDF.", "system_prompt_used": ""}

        formatted_other_reviews = "\n\n--- Other Reviewers' Feedback ---\n"
        for r_id, review_data in other_reviews.items():
            formatted_other_reviews += (
                f"\nReviewer {r_id} (Score: {review_data['initial_rating']}):\n"
                f"Review Details: {review_data['initial_review_text']}\n"
            )
        formatted_other_reviews += "\n----------------------------------------\n"

        full_system_prompt = self._get_current_system_prompt()

        user_prompt = (
            "You are conducting the **second stage** of the review process (Discussion/Rebuttal Phase). "
            f"Your initial review and rating were: Score {initial_review['initial_rating']}.\n"
            f"Your Initial Review (JSON):\n{initial_review['initial_review_text']}\n\n"
            "Review the paper provided as context, in light of the other reviews.\n"
            "You have the **option to adjust your rating**, but you must ensure your second stage review remains "
            "**ALIGNED with your core persona** (defined in the system prompt).\n"
            "Critically re-evaluate your rating and adjust your review text if necessary. "
            "Generate your final review following the required JSON policy.\n"
            f"{formatted_other_reviews}\n\n"
            f"Review Policy:\n{self.review_policy_prompt}"
        )

        print(f"-> Reviewer {self.reviewer_id} calling Gemini API for Stage 2 (Adjustment)...")

        generation_config = genai.types.GenerateContentConfig(
            system_instruction=full_system_prompt,
            response_mime_type="application/json",
            response_schema=self.REVIEW_SCHEMA
        )

        response = self.llm_client.models.generate_content(
            model=self.llm_model,
            contents=[uploaded_file, user_prompt],
            config=generation_config
        )

        review_data = json.loads(response.text)
        final_rating = review_data.get("score", None)

        return {
            "final_rating": final_rating,
            "final_review_text": response.text,
            "system_prompt_used": full_system_prompt
        }
