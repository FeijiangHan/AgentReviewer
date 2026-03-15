import re
from typing import List

from retrieval.crossref_client import CrossrefClient
from retrieval.openalex_client import OpenAlexClient
from retrieval.semanticscholar_client import SemanticScholarClient
from .models import CandidateReviewer


class IdentityResolver:
    def __init__(
        self,
        openalex_client: OpenAlexClient | None = None,
        semanticscholar_client: SemanticScholarClient | None = None,
        crossref_client: CrossrefClient | None = None,
    ):
        self.openalex_client = openalex_client or OpenAlexClient()
        self.semanticscholar_client = semanticscholar_client or SemanticScholarClient()
        self.crossref_client = crossref_client or CrossrefClient()

    @staticmethod
    def _extract_title_from_reference(reference: str) -> str:
        # heuristic: remove leading numbering/authors/year, keep probable title span
        line = re.sub(r"^\s*\[?\d+\]?\.??\s*", "", reference).strip()
        # split after year token and take the next segment as title candidate
        parts = re.split(r"\b(?:19|20)\d{2}\b", line, maxsplit=1)
        if len(parts) == 2:
            candidate = parts[1].strip(" .:-")
            if candidate:
                return candidate[:200]
        return line[:200]

    def resolve(self, candidates: List[CandidateReviewer]) -> List[CandidateReviewer]:
        resolved: List[CandidateReviewer] = []
        for cand in candidates:
            oa_hits = self.openalex_client.search_author_by_name(cand.name)
            s2_hits = self.semanticscholar_client.search_author(cand.name)

            score = 0.2
            if oa_hits:
                top = oa_hits[0]
                cand.affiliation = (top.get('last_known_institution') or {}).get('display_name', cand.affiliation)
                topics = [c.get('display_name', '') for c in top.get('x_concepts', [])[:5]]
                cand.research_areas = [t for t in topics if t]
                cand.source_signals['openalex'] = top.get('id', '')
                score += 0.4

            if s2_hits:
                top_s2 = s2_hits[0]
                affs = top_s2.get('affiliations') or []
                if affs and not cand.affiliation:
                    cand.affiliation = affs[0]
                cand.source_signals['semanticscholar'] = str(top_s2.get('authorId', ''))
                score += 0.3

            # Crossref enrichment from evidence references (title/DOI hints)
            if cand.evidence_references:
                title_hint = self._extract_title_from_reference(cand.evidence_references[0])
                cr_work = self.crossref_client.search_work_by_title(title_hint)
                if cr_work:
                    doi = cr_work.get('DOI', '')
                    if doi:
                        cand.source_signals['crossref_doi'] = doi
                    cr_title = (cr_work.get('title') or [""])[0]
                    if cr_title:
                        cand.publications.append(cr_title)
                    score += 0.1

            repeat_bonus = min(0.1, 0.02 * len(cand.evidence_references))
            cand.confidence = min(1.0, score + repeat_bonus)
            resolved.append(cand)

        return resolved
