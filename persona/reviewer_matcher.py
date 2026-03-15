import re
from typing import List

from .models import CandidateReviewer


class ReviewerMatcher:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z]{3,}", text.lower())}

    def match(self, paper_text: str, candidates: List[CandidateReviewer]) -> List[CandidateReviewer]:
        paper_tokens = self._tokenize(paper_text)

        scored = []
        for cand in candidates:
            profile_text = " ".join(cand.research_areas + cand.publications + cand.evidence_references)
            profile_tokens = self._tokenize(profile_text)
            overlap = len(paper_tokens.intersection(profile_tokens))
            score = overlap + (cand.confidence * 10.0)
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cand for _, cand in scored[: self.top_k]]
