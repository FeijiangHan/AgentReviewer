from dataclasses import dataclass
from typing import Dict, List

from .models import PersonaCard


@dataclass
class PersonaValidationResult:
    accepted: bool
    completeness_score: float
    missing_dimensions: List[str]


def validate_persona_card(card: PersonaCard, min_completeness: float = 0.75) -> PersonaValidationResult:
    """
    Validate whether a persona card is complete enough for reviewer simulation.

    Dimensions are intentionally professional-only and evidence-linked.
    """
    checks: Dict[str, bool] = {
        'name': bool(card.name.strip()),
        'affiliation': bool(card.affiliation.strip()),
        'research_areas': len(card.research_areas) >= 2,
        'methodological_preferences': len(card.methodological_preferences) >= 2,
        'common_concerns': len(card.common_concerns) >= 2,
        'style_signature': bool(card.style_signature.strip()),
        'evidence_sources': len(card.evidence_sources) >= 2,
        'confidence': card.confidence >= 0.5,
    }
    passed = sum(1 for v in checks.values() if v)
    completeness = passed / float(len(checks))
    missing = [k for k, ok in checks.items() if not ok]
    return PersonaValidationResult(
        accepted=completeness >= min_completeness,
        completeness_score=completeness,
        missing_dimensions=missing,
    )
