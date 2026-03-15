from typing import List

from .models import CandidateReviewer, PersonaCard


class DynamicPersonaBuilder:
    def build(self, candidates: List[CandidateReviewer]) -> List[PersonaCard]:
        persona_cards: List[PersonaCard] = []
        for cand in candidates:
            areas = cand.research_areas or ["machine learning", "empirical evaluation"]
            methods = [
                "strong ablation and controlled comparisons",
                "reproducibility and clear experimental setup",
            ]
            concerns = [
                "novel contribution over prior work",
                "methodological rigor",
                "clarity and evidence-backed claims",
            ]
            style = "balanced, technical, evidence-focused"
            biases = [
                f"may prefer papers related to {areas[0]}",
                "may be stricter on evaluation quality",
            ]

            evidence = []
            if cand.source_signals.get('openalex'):
                evidence.append(f"OpenAlex:{cand.source_signals['openalex']}")
            if cand.source_signals.get('semanticscholar'):
                evidence.append(f"SemanticScholar:{cand.source_signals['semanticscholar']}")
            evidence.extend(cand.evidence_references[:3])

            persona_cards.append(
                PersonaCard(
                    name=cand.name,
                    affiliation=cand.affiliation,
                    research_areas=areas,
                    methodological_preferences=methods,
                    common_concerns=concerns,
                    style_signature=style,
                    potential_biases=biases,
                    confidence=cand.confidence,
                    evidence_sources=evidence,
                )
            )
        return persona_cards
