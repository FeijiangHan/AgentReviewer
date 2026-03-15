import json
import unittest

from llm.base import LLMProvider
from role.simulator import SimulationManager


class DummyProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return 'dummy'

    @property
    def model_name(self) -> str:
        return 'dummy-model'

    def generate_json(self, system_prompt, user_prompt, response_schema=None):
        if 'Task: Make a final decision' in user_prompt:
            return {
                'final_justification': 'stub',
                'final_decision': 'Accept',
                'review_evaluation': [
                    {'id': 'R01', 'justification': 'ok', 'score': 6},
                    {'id': 'R02', 'justification': 'ok', 'score': 6},
                    {'id': 'R03', 'justification': 'ok', 'score': 6},
                ],
            }
        return {
            'strengths': ['s1', 's2'],
            'weaknesses': ['w1', 'w2'],
            'justification': 'stub',
            'score': '6',
        }

    def generate_text(self, system_prompt, user_prompt):
        return 'stub'


class MilestoneATests(unittest.TestCase):
    def _run_pipeline(self):
        papers = [
            {'id': 'paper_a', 'url': 'https://example.com/{paper_ID}.pdf', 'actual_rating': 5.0},
            {'id': 'paper_b', 'url': 'https://example.com/{paper_ID}.pdf', 'actual_rating': 6.0},
        ]
        manager = SimulationManager(provider=DummyProvider(), paper_list=papers, num_rounds=1, seed=7)
        manager.run_all_experiments()
        with open('simulation_results.json', 'r', encoding='utf-8') as f:
            output = json.load(f)
        return manager.results, output

    def test_round_execution_without_elo_dependencies(self):
        in_memory_results, output_results = self._run_pipeline()

        self.assertTrue(in_memory_results)
        self.assertTrue(output_results)
        for record in in_memory_results:
            self.assertNotIn('elos_after_round', record)
            self.assertNotIn('experiment', record)

    def test_output_schema_snapshot_minimal(self):
        _in_memory_results, output_results = self._run_pipeline()

        sample = output_results[0]
        expected_keys = {'round', 'paper_id', 'actual_rating', 'ac_decision', 'reviews', 'runtime_metadata'}
        self.assertEqual(set(sample.keys()), expected_keys)
        self.assertEqual(sample['runtime_metadata']['llm_provider'], 'dummy')


if __name__ == '__main__':
    unittest.main()
