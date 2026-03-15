import json
import random
from typing import Dict, List, Any, Optional

try:
    import numpy as np
except ImportError:  # optional dependency
    class _NP:
        class random:
            @staticmethod
            def seed(_seed):
                return None
    np = _NP()

from llm import create_provider
from llm.base import LLMProvider
from .reviewer import LLMAgentReviewer
from .area_chair import AreaChairAgent


class SimulationManager:
    """
    Core multi-round review pipeline manager (Elo-free) with pluggable LLM provider.
    """

    def __init__(
        self,
        provider: LLMProvider,
        paper_list: List[Dict[str, Any]],
        num_rounds: int = 5,
        seed: Optional[int] = 42,
    ):
        self.provider = provider
        self.num_rounds: int = num_rounds
        self.persona_names: List[str] = ['bluffer', 'critic', 'expert', 'harmonizer', 'optimist', 'skimmer']
        self.num_reviewers_per_paper: int = 3
        self.num_papers_per_round: int = 2

        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.reviewers: Dict[int, LLMAgentReviewer] = {}
        self.ac_agent = AreaChairAgent(provider=self.provider)
        self.papers: List[Dict[str, Any]] = paper_list
        self.results: List[Dict[str, Any]] = []

    @classmethod
    def from_config(
        cls,
        provider_name: str,
        api_key: str,
        model_name: str,
        paper_list: List[Dict[str, Any]],
        num_rounds: int = 5,
        seed: Optional[int] = 42,
    ):
        provider = create_provider(provider_name=provider_name, api_key=api_key, model_name=model_name)
        return cls(provider=provider, paper_list=paper_list, num_rounds=num_rounds, seed=seed)

    def _initialize_reviewers(self):
        print("--- Initializing Reviewer Agents ---")
        self.reviewers = {}
        for i, persona_name in enumerate(self.persona_names):
            reviewer_id = i + 1
            self.reviewers[reviewer_id] = LLMAgentReviewer(
                reviewer_id=reviewer_id,
                persona=persona_name,
                provider=self.provider,
            )
            print(f"Initialized Reviewer {reviewer_id}: {persona_name}")

    def _get_paper_content(self, paper: Dict[str, Any]) -> tuple[str, str]:
        if 'content' in paper and paper['content']:
            return paper['content'], paper.get('source', paper.get('id', 'paper'))
        paper_id = paper['id']
        paper_url = paper['url'].format(paper_ID=paper_id) if '{paper_ID}' in paper.get('url', '') else paper.get('url', '')
        placeholder = f"Paper content not pre-parsed. Source URL: {paper_url}"
        return placeholder, paper_url

    def _run_single_round(self, round_num: int, available_papers: List[Dict[str, Any]], all_reviewer_ids: List[int]):
        print(f"\n======== Running Round {round_num} ========")
        round_results: List[Dict[str, Any]] = []

        papers_to_review_count = min(self.num_papers_per_round, len(available_papers))
        papers_to_review_indices = random.sample(range(len(available_papers)), papers_to_review_count)

        paper_assignment: Dict[int, List[int]] = {}
        for paper_index in papers_to_review_indices:
            assigned_reviewers = random.sample(all_reviewer_ids, self.num_reviewers_per_paper)
            paper_assignment[paper_index] = assigned_reviewers

        for paper_index in paper_assignment:
            reviewer_ids = paper_assignment[paper_index]
            paper = available_papers[paper_index]
            paper_id = paper['id']
            paper_actual_rating = paper.get('actual_rating')
            paper_content, paper_label = self._get_paper_content(paper)

            print(f"\n--- Paper {paper_id} assigned to Reviewers {reviewer_ids} ---")

            stage1_reviews: Dict[int, Dict[str, Any]] = {}
            for r_id in reviewer_ids:
                stage1_reviews[r_id] = self.reviewers[r_id].generate_review_stage_1(paper_content, paper_label=paper_label)

            stage2_reviews: Dict[int, Dict[str, Any]] = {}
            for r_id in reviewer_ids:
                reviewer = self.reviewers[r_id]
                other_reviews = {k: v for k, v in stage1_reviews.items() if k != r_id}
                stage2_reviews[r_id] = reviewer.generate_review_stage_2(
                    paper_content,
                    stage1_reviews[r_id],
                    other_reviews,
                    paper_label=paper_label,
                )

            ac_decision = self.ac_agent.make_final_decision(paper_id, stage2_reviews)
            print(f"-> AC Decision: {ac_decision.get('final_decision')}")

            reviews_data_for_paper: Dict[int, Dict[str, Any]] = {}
            for r_id in reviewer_ids:
                review_details = {**stage1_reviews[r_id], **stage2_reviews[r_id]}
                reviews_data_for_paper[r_id] = review_details

            round_results.append({
                'round': round_num,
                'paper_id': paper_id,
                'actual_rating': paper_actual_rating,
                'ac_decision': ac_decision,
                'reviews': reviews_data_for_paper,
                'runtime_metadata': {
                    'llm_provider': self.provider.provider_name,
                    'review_model': self.reviewers[reviewer_ids[0]].llm_model if reviewer_ids else None,
                    'ac_model': self.ac_agent.llm_model,
                    'reviewer_ids': reviewer_ids,
                }
            })

        self.results.extend(round_results)
        print(f"======== Round {round_num} Completed ========")

    def run_experiment(self, experiment_mode: Optional[int] = None):
        _ = experiment_mode
        print("\n\n==============================================")
        print("🚀 STARTING CORE REVIEW PIPELINE (Elo-free)")
        print("==============================================")

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            print(f"Random seed reset to {self.seed}")

        self._initialize_reviewers()
        all_reviewer_ids = list(self.reviewers.keys())

        for round_num in range(1, self.num_rounds + 1):
            self._run_single_round(round_num, self.papers, all_reviewer_ids)

        print("\n==============================================")
        print("✅ CORE REVIEW PIPELINE COMPLETED")
        print("==============================================")

    def run_all_experiments(self, output_path: str = 'simulation_results.json'):
        self.run_experiment()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nPipeline complete. Results saved to '{output_path}'.")
