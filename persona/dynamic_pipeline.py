from typing import Any, Dict, List, Optional, Tuple

from llm.base import LLMProvider
from .dynamic_builder import DynamicPersonaBuilder
from .identity_resolver import IdentityResolver
from .llm_search_enricher import LLMSearchEnricher
from .models import PersonaCard
from .persona_validator import validate_persona_card
from .reference_miner import extract_candidate_reviewers_from_references
from .reviewer_matcher import ReviewerMatcher


class DynamicPersonaPipeline:
    def __init__(
        self,
        top_k: int = 3,
        min_confidence: float = 0.35,
        min_persona_completeness: float = 0.75,
        provider: Optional[LLMProvider] = None,
        use_llm_search: bool = False,
    ):
        self.identity_resolver = IdentityResolver()
        self.matcher = ReviewerMatcher(top_k=top_k)
        self.builder = DynamicPersonaBuilder()
        self.min_confidence = min_confidence
        self.min_persona_completeness = min_persona_completeness
        self.provider = provider
        self.use_llm_search = use_llm_search
        self.llm_enricher = LLMSearchEnricher(provider) if (use_llm_search and provider is not None) else None

    def run(self, paper_text: str, references: List[str]) -> List[PersonaCard]:
        cards, _trace = self.run_with_trace(paper_text, references)
        return cards

    def run_with_trace(self, paper_text: str, references: List[str]) -> Tuple[List[PersonaCard], Dict[str, Any]]:
        candidates = extract_candidate_reviewers_from_references(references)
        resolved = self.identity_resolver.resolve(candidates)
        enriched = self.llm_enricher.enrich(paper_text, resolved) if self.llm_enricher else resolved
        ranked = self.matcher.rank(paper_text, enriched)

        accepted_cards: List[PersonaCard] = []
        backup_pool: List[Dict[str, Any]] = []
        validation_trace: List[Dict[str, Any]] = []
        for cand in ranked:
            card = self.builder.build([cand])[0]
            validation = validate_persona_card(card, min_completeness=self.min_persona_completeness)
            validation_trace.append(
                {
                    'name': card.name,
                    'accepted': validation.accepted,
                    'completeness_score': validation.completeness_score,
                    'missing_dimensions': validation.missing_dimensions,
                }
            )
            if validation.accepted and len(accepted_cards) < self.matcher.top_k:
                accepted_cards.append(card)
            else:
                backup_pool.append(
                    {
                        'name': card.name,
                        'confidence': card.confidence,
                        'completeness_score': validation.completeness_score,
                        'missing_dimensions': validation.missing_dimensions,
                    }
                )

        cards = accepted_cards

        trace: Dict[str, Any] = {
            'references': {
                'count': len(references),
                'extraction_success': bool(references),
                'items': references,
            },
            'candidate_stats': {
                'raw_count': len(candidates),
                'resolved_count': len(resolved),
                'filtered_count': len(enriched),
                'selected_count': min(len(ranked), self.matcher.top_k),
                'validated_count': len(validation_trace),
                'accepted_persona_count': len(accepted_cards),
                'requested_top_k': self.matcher.top_k,
                'min_confidence': self.min_confidence,
                'confidence_filtering_enabled': False,
                'llm_search_enrichment_enabled': bool(self.llm_enricher),
                'min_persona_completeness': self.min_persona_completeness,
            },
            'all_candidate_authors': [
                {
                    'name': c.name,
                    'affiliation': c.affiliation,
                    'research_areas': c.research_areas,
                    'confidence': c.confidence,
                    'evidence_references': c.evidence_references,
                    'source_signals': c.source_signals,
                    'name_verified': bool(c.source_signals),
                    'affiliation_verified': bool(c.affiliation),
                    'verified_name_affiliation_pair': bool(c.source_signals) and bool(c.affiliation),
                }
                for c in resolved
            ],
            'selected_reviewers': [
                {
                    'name': c.name,
                    'affiliation': c.affiliation,
                    'research_areas': c.research_areas,
                    'confidence': c.confidence,
                    'source_signals': c.source_signals,
                }
                for c in ranked[: self.matcher.top_k]
            ],
            'persona_validation': validation_trace,
            'backup_pool': backup_pool,
        }

        return cards, trace
