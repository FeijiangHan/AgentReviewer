import os
import json
import random
import numpy as np
from google import genai
from typing import Dict, List, Any, Optional

from .reviewer import LLMAgentReviewer
from .area_chair import AreaChairAgent


class SimulationManager:
    """
    Manages multi-round paper review using LLM reviewer agents and an Area Chair.

    Milestone A (Elo removal): this manager keeps the core review/evaluation flow
    and removes Elo-based ranking/reward/memory update behavior.
    """

    def __init__(self, api_key: str, paper_list: List[Dict[str, Any]], num_rounds: int = 5, seed: Optional[int] = 42):
        """
        Initializes the Simulation Manager.

        Args:
            api_key (str): The Google AI API key.
            paper_list (List[Dict]): List of papers with ID and actual_rating.
            num_rounds (int): The number of rounds (review cycles) to run.
            seed (Optional[int]): Random seed for reproducibility. Default is 42.
        """
        os.environ['GEMINI_API_KEY'] = api_key
        self.client = genai.Client(api_key=api_key)

        self.num_rounds: int = num_rounds
        self.persona_names: List[str] = ['bluffer', 'critic', 'expert', 'harmonizer', 'optimist', 'skimmer']
        self.num_reviewers_per_paper: int = 3
        self.num_papers_per_round: int = 2

        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.reviewers: Dict[int, 'LLMAgentReviewer'] = {}
        self.ac_agent: 'AreaChairAgent' = AreaChairAgent(client=self.client)
        self.papers: List[Dict[str, Any]] = paper_list
        self.results: List[Dict[str, Any]] = []

    def _initialize_reviewers(self, initial_elo: float = 1500.0):
        """
        Initializes the 6 reviewer agents with unique IDs and personas.

        Note: `initial_elo` is kept only for backward constructor compatibility of
        `LLMAgentReviewer` and is no longer used in pipeline logic.
        """
        print("--- Initializing Reviewer Agents ---")
        self.reviewers = {}
        for i, persona_name in enumerate(self.persona_names):
            reviewer_id = i + 1
            self.reviewers[reviewer_id] = LLMAgentReviewer(
                reviewer_id=reviewer_id,
                persona=persona_name,
                client=self.client,
                initial_elo=initial_elo
            )
            print(f"Initialized Reviewer {reviewer_id}: {persona_name}")

    def _run_single_round(self, round_num: int, available_papers: List[Dict[str, Any]], all_reviewer_ids: List[int]):
        """
        Executes one full round of core review workflow:
        random assignment -> stage-1 reviews -> stage-2 revisions -> AC decision.
        """
        print(f"\n======== Running Round {round_num} ========")
        round_results: List[Dict[str, Any]] = []

        papers_to_review_indices = random.sample(range(len(available_papers)), self.num_papers_per_round)

        paper_assignment: Dict[int, List[int]] = {}
        for paper_index in papers_to_review_indices:
            assigned_reviewers = random.sample(all_reviewer_ids, self.num_reviewers_per_paper)
            paper_assignment[paper_index] = assigned_reviewers

        for paper_index in paper_assignment:
            reviewer_ids = paper_assignment[paper_index]
            paper = available_papers[paper_index]
            paper_id = paper['id']
            paper_url = paper['url'].format(paper_ID=paper_id) if '{paper_ID}' in paper['url'] else paper['url']
            paper_actual_rating = paper['actual_rating']

            print(f"\n--- Paper {paper_id} assigned to Reviewers {reviewer_ids} ---")

            # Stage 1: Initial Review
            stage1_reviews: Dict[int, Dict[str, Any]] = {}
            for r_id in reviewer_ids:
                stage1_reviews[r_id] = self.reviewers[r_id].generate_review_stage_1(paper_url)

            # Stage 2: Adjusted Review
            stage2_reviews: Dict[int, Dict[str, Any]] = {}
            for r_id in reviewer_ids:
                reviewer = self.reviewers[r_id]
                other_reviews = {k: v for k, v in stage1_reviews.items() if k != r_id}
                stage2_reviews[r_id] = reviewer.generate_review_stage_2(
                    paper_url,
                    stage1_reviews[r_id],
                    other_reviews
                )

            # AC Decision (Elo disclosure disabled in Milestone A)
            ac_decision = self.ac_agent.make_final_decision(
                paper_id,
                stage2_reviews,
                reviewer_elos={},
                disclose_elo=False
            )
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
                    'llm_provider': 'gemini',
                    'review_model': self.reviewers[reviewer_ids[0]].llm_model if reviewer_ids else None,
                    'ac_model': self.ac_agent.llm_model,
                    'reviewer_ids': reviewer_ids
                }
            })

        self.results.extend(round_results)
        print(f"======== Round {round_num} Completed ========")

    def run_experiment(self, experiment_mode: Optional[int] = None):
        """
        Runs the Elo-free multi-round review pipeline once.

        `experiment_mode` is accepted for backward compatibility and ignored.
        """
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

    def run_all_experiments(self):
        """
        Backward-compatible entrypoint.
        In Milestone A it runs one Elo-free pipeline execution.
        """
        self.run_experiment()

        with open('simulation_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        print("\nPipeline complete. Results saved to 'simulation_results.json'.")
