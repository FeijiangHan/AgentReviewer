from typing import Any, Dict, List, Tuple

from .dynamic_builder import DynamicPersonaBuilder
from .identity_resolver import IdentityResolver
from .models import PersonaCard
from .reference_miner import extract_candidate_reviewers_from_references
from .reviewer_matcher import ReviewerMatcher


class DynamicPersonaPipeline:
    def __init__(self, top_k: int = 3, min_confidence: float = 0.35):
        self.identity_resolver = IdentityResolver()
        self.matcher = ReviewerMatcher(top_k=top_k)
        self.builder = DynamicPersonaBuilder()
        self.min_confidence = min_confidence

    def run(self, paper_text: str, references: List[str]) -> List[PersonaCard]:
        cards, _trace = self.run_with_trace(paper_text, references)
        return cards

    def run_with_trace(self, paper_text: str, references: List[str]) -> Tuple[List[PersonaCard], Dict[str, Any]]:
        candidates = extract_candidate_reviewers_from_references(references)
        resolved = self.identity_resolver.resolve(candidates)
        filtered = [c for c in resolved if c.confidence >= self.min_confidence]
        matched = self.matcher.match(paper_text, filtered)
        cards = self.builder.build(matched)

        trace: Dict[str, Any] = {
            'references': {
                'count': len(references),
                'extraction_success': bool(references),
                'items': references,
            },
            'candidate_stats': {
                'raw_count': len(candidates),
                'resolved_count': len(resolved),
                'filtered_count': len(filtered),
                'selected_count': len(matched),
                'requested_top_k': self.matcher.top_k,
                'min_confidence': self.min_confidence,
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
                for c in matched
            ],
        }

        return cards, trace
