import os
import tempfile
import unittest

from persona.dynamic_builder import DynamicPersonaBuilder
from persona.dynamic_pipeline import DynamicPersonaPipeline
from persona.models import CandidateReviewer
from persona.reference_miner import extract_candidate_reviewers_from_references
from persona.reviewer_matcher import ReviewerMatcher


class MilestoneDTests(unittest.TestCase):
    def test_reference_miner_extracts_candidates(self):
        refs = [
            "Alice Smith, Bob Lee, and Carol Tan. 2023. Strong Baseline for Vision.",
            "[2] David Kim, Alice Smith. 2022. Efficient Pretraining.",
        ]
        candidates = extract_candidate_reviewers_from_references(refs)
        names = [c.name for c in candidates]
        self.assertTrue(any("Alice" in n for n in names))
        self.assertTrue(len(candidates) >= 3)

    def test_matcher_ranks_by_confidence_and_overlap(self):
        matcher = ReviewerMatcher(top_k=2)
        c1 = CandidateReviewer(name="Alice Smith", research_areas=["vision transformer"], confidence=0.9)
        c2 = CandidateReviewer(name="Bob Lee", research_areas=["reinforcement learning"], confidence=0.1)
        c3 = CandidateReviewer(name="Carol Tan", research_areas=["vision", "diffusion"], confidence=0.6)

        top = matcher.match("We propose a vision transformer with diffusion guidance.", [c1, c2, c3])
        self.assertEqual(len(top), 2)
        self.assertIn(top[0].name, {"Alice Smith", "Carol Tan"})

    def test_dynamic_builder_generates_persona_cards(self):
        builder = DynamicPersonaBuilder()
        candidates = [
            CandidateReviewer(
                name="Alice Smith",
                affiliation="Example University",
                research_areas=["vision", "multimodal learning"],
                confidence=0.8,
                evidence_references=["ref1", "ref2"],
                source_signals={"openalex": "A123"},
            )
        ]
        cards = builder.build(candidates)
        self.assertEqual(len(cards), 1)
        self.assertIn("Alice Smith", cards[0].to_prompt())

    def test_local_pipeline_dynamic_mode_writes_persona_cards(self):
        from pipeline import local_review_pipeline as lrp

        original_ingest = lrp.ingest_local_pdf
        original_dynamic_runner = lrp._run_dynamic_persona_review

        class DummyPaper:
            paper_id = 'demo'
            source = 'demo.pdf'
            full_text = 'paper body'
            references = ['Alice Smith, Bob Lee. 2023. Test.']

        try:
            lrp.ingest_local_pdf = lambda _path: DummyPaper()
            lrp._run_dynamic_persona_review = lambda **kwargs: (
                [{
                    'round': 1,
                    'paper_id': 'demo',
                    'actual_rating': None,
                    'ac_decision': {'final_decision': 'Accept', 'final_justification': 'ok', 'review_evaluation': []},
                    'reviews': {},
                    'runtime_metadata': {'persona_mode': 'dynamic'}
                }],
                [{'name': 'Alice Smith', 'confidence': 0.8}]
            )

            with tempfile.TemporaryDirectory() as tmp:
                out = lrp.run_local_pdf_review(
                    pdf_path='demo.pdf',
                    provider_name='gemini',
                    api_key='k',
                    model_name='m',
                    output_dir=tmp,
                    persona_mode='dynamic',
                )
                self.assertTrue(os.path.exists(out['persona_cards_json']))
                self.assertTrue(os.path.exists(out['reviews_json']))
                self.assertTrue(os.path.exists(out['report_md']))
        finally:
            lrp.ingest_local_pdf = original_ingest
            lrp._run_dynamic_persona_review = original_dynamic_runner

    def test_dynamic_pipeline_filters_low_confidence(self):
        pipeline = DynamicPersonaPipeline(top_k=3, min_confidence=0.9)

        class StubResolver:
            def resolve(self, _cands):
                return [CandidateReviewer(name='Low A', confidence=0.2), CandidateReviewer(name='Low B', confidence=0.1)]

        pipeline.identity_resolver = StubResolver()
        cards = pipeline.run('paper', ['Alice Smith. 2023. X.'])
        self.assertEqual(cards, [])

    def test_identity_resolver_uses_crossref_signal(self):
        from persona.identity_resolver import IdentityResolver

        class OA:
            def search_author_by_name(self, _name):
                return []

        class S2:
            def search_author(self, _name):
                return []

        class CR:
            def search_work_by_title(self, _title):
                return {'DOI': '10.1000/test', 'title': ['Canonical Title']}

        resolver = IdentityResolver(openalex_client=OA(), semanticscholar_client=S2(), crossref_client=CR())
        cand = CandidateReviewer(name='Alice Smith', evidence_references=['Alice Smith 2023 A Good Paper'])
        out = resolver.resolve([cand])[0]
        self.assertIn('crossref_doi', out.source_signals)
        self.assertIn('Canonical Title', out.publications)



if __name__ == '__main__':
    unittest.main()
