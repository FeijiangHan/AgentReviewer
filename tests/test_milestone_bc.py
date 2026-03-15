import os
import sys
import types
import unittest

from llm import create_provider
from pipeline.ingest import extract_references, detect_title


class MilestoneBCTests(unittest.TestCase):
    def test_provider_factory_supports_gemini_and_openai(self):
        # Stub google.genai
        google_mod = types.ModuleType('google')
        genai_mod = types.ModuleType('google.genai')

        class DummyGeminiClient:
            def __init__(self, api_key=None):
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_content=lambda **kwargs: types.SimpleNamespace(text='{"ok": true}'))

        class DummyConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        genai_mod.Client = DummyGeminiClient
        genai_mod.types = types.SimpleNamespace(GenerateContentConfig=DummyConfig)
        google_mod.genai = genai_mod
        sys.modules['google'] = google_mod
        sys.modules['google.genai'] = genai_mod

        # Stub openai
        openai_mod = types.ModuleType('openai')

        class DummyOpenAI:
            def __init__(self, api_key=None):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=lambda **kwargs: types.SimpleNamespace(
                            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"ok": true}'))]
                        )
                    )
                )

        openai_mod.OpenAI = DummyOpenAI
        sys.modules['openai'] = openai_mod

        gemini_provider = create_provider('gemini', 'k1', 'gemini-test')
        openai_provider = create_provider('openai', 'k2', 'gpt-test')

        self.assertEqual(gemini_provider.provider_name, 'gemini')
        self.assertEqual(openai_provider.provider_name, 'openai')

    def test_local_pdf_pipeline_writes_outputs(self):
        import tempfile
        from pipeline import local_review_pipeline as lrp
        from pipeline.ingest import PaperContext

        original_ingest = lrp.ingest_local_pdf
        original_from_config = lrp.SimulationManager.from_config

        class DummyManager:
            def __init__(self):
                self.results = [{'ac_decision': {'final_decision': 'Accept', 'final_justification': 'ok'}}]
                self.num_papers_per_round = 0

            def run_all_experiments(self, output_path='simulation_results.json'):
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f)

        try:
            lrp.ingest_local_pdf = lambda _path: PaperContext(
                paper_id='demo',
                source='demo.pdf',
                full_text='sample text',
                references=['r1', 'r2'],
            )
            lrp.SimulationManager.from_config = lambda **_kwargs: DummyManager()

            with tempfile.TemporaryDirectory() as tmp:
                cwd = os.getcwd()
                os.chdir(tmp)
                try:
                    out = lrp.run_local_pdf_review(
                        pdf_path='demo.pdf',
                        provider_name='gemini',
                        api_key='k',
                        model_name='m',
                        output_dir=tmp,
                    )
                    self.assertTrue(out['reviews_json'].endswith('reviews.json'))
                    self.assertTrue(out['report_md'].endswith('report.md'))
                    self.assertEqual(out['references_count'], 2)
                finally:
                    os.chdir(cwd)
        finally:
            lrp.ingest_local_pdf = original_ingest
            lrp.SimulationManager.from_config = original_from_config

    def test_detect_title_heuristic(self):
        text = "A Great Paper Title\n\nAbstract\nSomething..."
        self.assertEqual(detect_title(text), 'A Great Paper Title')

    def test_reference_extraction_from_text(self):
        sample_text = (
            'Intro section\nMethod section\nREFERENCES\n'
            '[1] Alice A. Paper One.\n'
            '[2] Bob B. Paper Two.\n'
        )
        refs = extract_references(sample_text)
        self.assertGreaterEqual(len(refs), 2)
        self.assertIn('Alice', refs[0])


if __name__ == '__main__':
    unittest.main()
