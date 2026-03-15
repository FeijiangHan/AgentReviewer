from typing import Any, Dict, List

from llm.base import LLMProvider
from .models import CandidateReviewer


class LLMSearchEnricher:
    """
    Optional enrichment adapter that asks an LLM (preferably with web-search capability)
    to gather professional, public information for reviewer personas.

    Safety rule: candidates must come from references only; this class only enriches fields.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def enrich(self, paper_text: str, candidates: List[CandidateReviewer]) -> List[CandidateReviewer]:
        if not candidates:
            return candidates

        for cand in candidates:
            payload = self._query_candidate_profile(paper_text=paper_text, candidate=cand)
            self._merge_payload(cand, payload)
        return candidates

    def _query_candidate_profile(self, paper_text: str, candidate: CandidateReviewer) -> Dict[str, Any]:
        system_prompt = (
            "You are a scholarly profile retrieval assistant. "
            "Only use public professional information. "
            "If browsing/search is unavailable, return best-effort unknown fields."
        )
        user_prompt = (
            "Task: Build a professional reviewer profile candidate from publicly available info.\n"
            "Constraints:\n"
            "1) Candidate identity comes from the paper references and cannot be replaced.\n"
            "2) Focus on professional fields only (affiliation, topics, methods, concerns).\n"
            "3) Prefer evidence from homepage, lab page, scholar profile, talks, publications.\n"
            "4) Return strict JSON with keys: affiliation (string), research_areas (string[]), "
            "methodological_preferences (string[]), common_concerns (string[]), "
            "style_signature (string), evidence_sources (string[]), confidence (number 0-1).\n\n"
            f"Candidate name: {candidate.name}\n"
            f"Reference evidence: {candidate.evidence_references[:3]}\n"
            f"Paper abstract/content snippet: {paper_text[:1200]}"
        )
        try:
            result = self.provider.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _merge_payload(candidate: CandidateReviewer, payload: Dict[str, Any]):
        affiliation = payload.get('affiliation', '')
        if isinstance(affiliation, str) and affiliation and not candidate.affiliation:
            candidate.affiliation = affiliation

        areas = payload.get('research_areas') or []
        if isinstance(areas, list):
            candidate.research_areas = _merge_unique(candidate.research_areas, [str(x) for x in areas if str(x).strip()])

        methods = payload.get('methodological_preferences') or []
        if isinstance(methods, list):
            candidate.source_signals['llm_method_prefs'] = '; '.join(str(x) for x in methods if str(x).strip())[:500]

        concerns = payload.get('common_concerns') or []
        if isinstance(concerns, list):
            candidate.source_signals['llm_common_concerns'] = '; '.join(str(x) for x in concerns if str(x).strip())[:500]

        style = payload.get('style_signature', '')
        if isinstance(style, str) and style.strip():
            candidate.source_signals['llm_style_signature'] = style.strip()[:200]

        evidence_sources = payload.get('evidence_sources') or []
        if isinstance(evidence_sources, list):
            for src in evidence_sources[:5]:
                src_text = str(src).strip()
                if src_text:
                    candidate.evidence_references.append(src_text)

        confidence = payload.get('confidence')
        if isinstance(confidence, (int, float)):
            # confidence is used as ranking signal only (not hard filter)
            candidate.confidence = max(candidate.confidence, min(1.0, float(confidence)))

        candidate.source_signals['llm_search'] = 'enabled'


def _merge_unique(existing: List[str], incoming: List[str]) -> List[str]:
    merged = list(existing)
    seen = {x.lower().strip() for x in merged}
    for item in incoming:
        key = item.lower().strip()
        if key and key not in seen:
            merged.append(item)
            seen.add(key)
    return merged
