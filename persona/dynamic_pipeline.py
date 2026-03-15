from typing import List

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
        candidates = extract_candidate_reviewers_from_references(references)
        resolved = self.identity_resolver.resolve(candidates)
        filtered = [c for c in resolved if c.confidence >= self.min_confidence]
        matched = self.matcher.match(paper_text, filtered)
        return self.builder.build(matched)
