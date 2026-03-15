from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class CandidateReviewer:
    name: str
    evidence_references: List[str] = field(default_factory=list)
    affiliation: str = ""
    research_areas: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source_signals: Dict[str, str] = field(default_factory=dict)


@dataclass
class PersonaCard:
    name: str
    affiliation: str
    research_areas: List[str]
    methodological_preferences: List[str]
    common_concerns: List[str]
    style_signature: str
    potential_biases: List[str]
    confidence: float
    evidence_sources: List[str]

    def to_prompt(self) -> str:
        areas = ", ".join(self.research_areas) if self.research_areas else "general machine learning"
        prefs = "; ".join(self.methodological_preferences) if self.methodological_preferences else "clear empirical validation"
        concerns = "; ".join(self.common_concerns) if self.common_concerns else "novelty, rigor, and clarity"
        biases = "; ".join(self.potential_biases) if self.potential_biases else "none explicitly inferred"
        evidence = " | ".join(self.evidence_sources[:5])
        return (
            f"You are simulating reviewer persona '{self.name}'.\n"
            f"Professional Profile:\n"
            f"- Affiliation: {self.affiliation or 'Unknown'}\n"
            f"- Research Areas: {areas}\n"
            f"- Methodological Preferences: {prefs}\n"
            f"- Common Concerns: {concerns}\n"
            f"- Review Style: {self.style_signature}\n"
            f"- Potential Biases (probabilistic): {biases}\n"
            f"- Confidence: {self.confidence:.2f}\n"
            f"- Evidence Sources: {evidence or 'Reference-derived heuristic profile'}\n\n"
            "Use this persona consistently, but do not fabricate unverifiable personal facts."
        )
