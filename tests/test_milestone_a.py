import json
import sys
import types
import unittest
import importlib


class MilestoneATests(unittest.TestCase):
    def setUp(self):
        # Stub numpy used by simulator seed path
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.random = types.SimpleNamespace(seed=lambda _seed: None)
        sys.modules["numpy"] = numpy_stub

        # Stub google.genai minimal surface
        google_mod = types.ModuleType("google")
        genai_mod = types.ModuleType("google.genai")
        errors_mod = types.ModuleType("google.genai.errors")

        class APIError(Exception):
            pass

        class DummyClient:
            def __init__(self, api_key=None):
                self.api_key = api_key

        class DummyGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        genai_mod.Client = DummyClient
        genai_mod.types = types.SimpleNamespace(GenerateContentConfig=DummyGenerateContentConfig)
        errors_mod.APIError = APIError

        google_mod.genai = genai_mod
        sys.modules["google"] = google_mod
        sys.modules["google.genai"] = genai_mod
        sys.modules["google.genai.errors"] = errors_mod

        # Stub httpx to satisfy reviewer import in test env
        httpx_mod = types.ModuleType("httpx")
        class HTTPError(Exception):
            pass
        def _dummy_get(_url):
            raise HTTPError("stub")
        httpx_mod.HTTPError = HTTPError
        httpx_mod.get = _dummy_get
        sys.modules["httpx"] = httpx_mod

        # Re-import modules so they bind to stubs
        self.sim_mod = importlib.reload(importlib.import_module("role.simulator"))

        # Monkeypatch networked methods to deterministic local behavior
        self.sim_mod.LLMAgentReviewer.generate_review_stage_1 = lambda self, _url: {
            "initial_rating": "6",
            "initial_review_text": '{"score":"6"}',
            "system_prompt_used": "stub_system"
        }
        self.sim_mod.LLMAgentReviewer.generate_review_stage_2 = lambda self, _url, _initial, _others: {
            "final_rating": "6",
            "final_review_text": '{"score":"6"}',
            "system_prompt_used": "stub_system"
        }
        self.sim_mod.AreaChairAgent.make_final_decision = (
            lambda self, _paper_title, final_reviews, reviewer_elos=None, disclose_elo=False, **_kwargs: {
                "final_justification": "stub",
                "final_decision": "Accept",
                "review_evaluation": [
                    {"id": f"R{rid:02d}", "justification": "ok", "score": 6}
                    for rid in sorted(final_reviews.keys())
                ]
            }
        )

    def _run_pipeline(self):
        papers = [
            {"id": "paper_a", "url": "https://example.com/{paper_ID}.pdf", "actual_rating": 5.0},
            {"id": "paper_b", "url": "https://example.com/{paper_ID}.pdf", "actual_rating": 6.0},
        ]
        manager = self.sim_mod.SimulationManager(api_key="dummy", paper_list=papers, num_rounds=1, seed=7)

        manager.run_all_experiments()
        with open("simulation_results.json", "r", encoding="utf-8") as f:
            output = json.load(f)

        return manager.results, output

    def test_round_execution_without_elo_dependencies(self):
        in_memory_results, output_results = self._run_pipeline()

        self.assertTrue(in_memory_results)
        self.assertTrue(output_results)
        for record in in_memory_results:
            self.assertNotIn("elos_after_round", record)
            self.assertNotIn("experiment", record)

    def test_output_schema_snapshot_minimal(self):
        _in_memory_results, output_results = self._run_pipeline()

        sample = output_results[0]
        expected_keys = {
            "round",
            "paper_id",
            "actual_rating",
            "ac_decision",
            "reviews",
            "runtime_metadata",
        }
        self.assertEqual(set(sample.keys()), expected_keys)
        self.assertEqual(sample["runtime_metadata"]["llm_provider"], "gemini")


if __name__ == "__main__":
    unittest.main()
