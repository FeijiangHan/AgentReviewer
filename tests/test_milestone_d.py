import os
import json
import tempfile
import unittest

from persona.dynamic_builder import DynamicPersonaBuilder
from persona.dynamic_pipeline import DynamicPersonaPipeline
from persona.models import CandidateReviewer
from persona.reference_miner import extract_candidate_reviewers_from_references
from persona.reviewer_matcher import ReviewerMatcher
from persona.models import PersonaCard
from persona.llm_search_enricher import LLMSearchEnricher
from persona.persona_validator import validate_persona_card
from role.simulator import SimulationManager
from llm.base import LLMProvider



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
                [{'name': 'Alice Smith', 'confidence': 0.8}],
                {'references': {'count': 1, 'extraction_success': True, 'items': ['r1']}, 'candidate_stats': {'final_selected_count': 1}}
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
                self.assertTrue(os.path.exists(out['dynamic_trace_json']))
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

    def test_dataset_mode_dynamic_persona_runs(self):
        import role.simulator as sim_mod

        original_pipeline_cls = sim_mod.DynamicPersonaPipeline

        class StubPipeline:
            def __init__(self, top_k=3, min_confidence=0.35, provider=None, use_llm_search=False):
                self.top_k = top_k
                self.min_confidence = min_confidence
                self.provider = provider
                self.use_llm_search = use_llm_search

            def run_with_trace(self, paper_text, references):
                cards = [
                    PersonaCard(
                        name='Dyn A', affiliation='Inst A', research_areas=['vision'],
                        methodological_preferences=['empirical rigor'],
                        common_concerns=['novelty'], style_signature='technical',
                        potential_biases=['vision preference'], confidence=0.8,
                        evidence_sources=['ref1'],
                    ),
                    PersonaCard(
                        name='Dyn B', affiliation='Inst B', research_areas=['nlp'],
                        methodological_preferences=['ablation'],
                        common_concerns=['clarity'], style_signature='balanced',
                        potential_biases=['nlp preference'], confidence=0.7,
                        evidence_sources=['ref2'],
                    ),
                ]
                trace = {
                    'references': {'count': len(references), 'extraction_success': True, 'items': references},
                    'candidate_stats': {'raw_count': 2, 'resolved_count': 2, 'filtered_count': 2, 'selected_count': 2, 'requested_top_k': self.top_k},
                    'all_candidate_authors': [],
                    'selected_reviewers': [],
                }
                return cards, trace

        try:
            sim_mod.DynamicPersonaPipeline = StubPipeline
            papers = [{
                'id': 'p1',
                'content': 'A paper about vision and language.',
                'references': ['Alice Smith. 2023. X.'],
                'actual_rating': 6.0,
            }]
            manager = SimulationManager(
                provider=DummyProvider(),
                paper_list=papers,
                num_rounds=1,
                seed=7,
                persona_mode='dynamic',
                top_k_reviewers=2,
            )
            manager.num_papers_per_round = 1
            manager.run_all_experiments(output_path='simulation_results.json')

            self.assertTrue(manager.results)
            md = manager.results[0]['runtime_metadata']
            self.assertEqual(md['persona_mode'], 'dynamic')
            self.assertEqual(len(md['reviewer_ids']), 2)
            self.assertTrue(os.path.exists(os.path.join('outputs', 'p1', 'dynamic_persona_trace.json')))
        finally:
            sim_mod.DynamicPersonaPipeline = original_pipeline_cls

    def test_dynamic_pipeline_trace_contains_verification_fields(self):
        pipeline = DynamicPersonaPipeline(top_k=2, min_confidence=0.0)

        class StubResolver:
            def resolve(self, _cands):
                return [
                    CandidateReviewer(
                        name='Alice Smith',
                        affiliation='Uni A',
                        confidence=0.8,
                        source_signals={'openalex': 'A1'},
                        evidence_references=['ref1'],
                    )
                ]

        pipeline.identity_resolver = StubResolver()
        cards, trace = pipeline.run_with_trace('vision paper', ['Alice Smith. 2024. Test.'])
        self.assertEqual(len(cards), 1)
        self.assertTrue(trace['references']['extraction_success'])
        self.assertIn('all_candidate_authors', trace)
        self.assertTrue(trace['all_candidate_authors'][0]['name_verified'])
        self.assertTrue(trace['all_candidate_authors'][0]['affiliation_verified'])

    def test_persona_validator_rejects_incomplete_card(self):
        card = PersonaCard(
            name='A',
            affiliation='',
            research_areas=['vision'],
            methodological_preferences=['ablation'],
            common_concerns=['novelty'],
            style_signature='technical',
            potential_biases=['none'],
            confidence=0.3,
            evidence_sources=['ref1'],
        )
        result = validate_persona_card(card, min_completeness=0.75)
        self.assertFalse(result.accepted)
        self.assertIn('affiliation', result.missing_dimensions)


    def test_dynamic_pipeline_does_not_filter_by_confidence_threshold(self):
        pipeline = DynamicPersonaPipeline(top_k=2, min_confidence=0.95, min_persona_completeness=0.0)

        class StubResolver:
            def resolve(self, _cands):
                return [
                    CandidateReviewer(name='Low A', affiliation='Inst A', confidence=0.1, research_areas=['vision', 'ml'], evidence_references=['r1']),
                    CandidateReviewer(name='Low B', affiliation='Inst B', confidence=0.2, research_areas=['nlp', 'ml'], evidence_references=['r2']),
                ]

        pipeline.identity_resolver = StubResolver()
        cards, trace = pipeline.run_with_trace('vision and nlp paper', ['A. 2024. X'])
        self.assertEqual(len(cards), 2)
        self.assertFalse(trace['candidate_stats']['confidence_filtering_enabled'])
        self.assertEqual(trace['candidate_stats']['filtered_count'], 2)

    def test_llm_search_enricher_merges_profile_fields(self):
        class StubProvider(LLMProvider):
            @property
            def provider_name(self):
                return 'stub'

            @property
            def model_name(self):
                return 'stub-model'

            def generate_json(self, system_prompt, user_prompt, response_schema=None):
                return {
                    'affiliation': 'Example University',
                    'research_areas': ['vision', 'multimodal'],
                    'methodological_preferences': ['ablation'],
                    'common_concerns': ['rigor'],
                    'style_signature': 'constructive',
                    'evidence_sources': ['https://example.edu/~alice'],
                    'confidence': 0.7,
                }

            def generate_text(self, system_prompt, user_prompt):
                return 'stub'

        enricher = LLMSearchEnricher(StubProvider())
        cand = CandidateReviewer(name='Alice Smith', confidence=0.2, evidence_references=['ref1'])
        out = enricher.enrich('paper text', [cand])[0]
        self.assertEqual(out.affiliation, 'Example University')
        self.assertIn('vision', out.research_areas)
        self.assertIn('llm_search', out.source_signals)
        self.assertGreaterEqual(out.confidence, 0.7)

    def test_dataset_dynamic_fallback_fills_top_k(self):
        import role.simulator as sim_mod

        original_pipeline_cls = sim_mod.DynamicPersonaPipeline

        class StubPipeline:
            def __init__(self, top_k=3, min_confidence=0.35, provider=None, use_llm_search=False):
                self.top_k = top_k
                self.min_confidence = min_confidence
                self.provider = provider
                self.use_llm_search = use_llm_search

            def run_with_trace(self, paper_text, references):
                trace = {
                    'references': {'count': len(references), 'extraction_success': False, 'items': references},
                    'candidate_stats': {'raw_count': 0, 'resolved_count': 0, 'filtered_count': 0, 'selected_count': 0, 'requested_top_k': self.top_k},
                    'all_candidate_authors': [],
                    'selected_reviewers': [],
                }
                return [], trace

        try:
            sim_mod.DynamicPersonaPipeline = StubPipeline
            manager = SimulationManager(
                provider=DummyProvider(),
                paper_list=[{'id': 'p2', 'content': 'x', 'references': []}],
                num_rounds=1,
                seed=1,
                persona_mode='dynamic',
                top_k_reviewers=3,
            )
            manager.num_papers_per_round = 1
            manager.run_all_experiments(output_path='simulation_results.json')
            md = manager.results[0]['runtime_metadata']
            self.assertEqual(len(md['reviewer_ids']), 3)
            trace_path = os.path.join('outputs', 'p2', 'dynamic_persona_trace.json')
            self.assertTrue(os.path.exists(trace_path))
            with open(trace_path, 'r', encoding='utf-8') as f:
                trace = json.load(f)
            self.assertTrue(trace['fallback_used'])
            self.assertEqual(trace['candidate_stats']['final_selected_count'], 3)
        finally:
            sim_mod.DynamicPersonaPipeline = original_pipeline_cls



if __name__ == '__main__':
    unittest.main()
