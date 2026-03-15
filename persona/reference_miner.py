import re
from typing import List

from .models import CandidateReviewer


def extract_candidate_reviewers_from_references(references: List[str]) -> List[CandidateReviewer]:
    """
    Build an initial reviewer candidate pool from reference author strings.
    Heuristic parser designed to be robust for noisy bibliographic text.
    """
    candidates = {}
    for ref in references:
        # Capture author segment before year/title separators when possible
        segment = re.sub(r"^\s*\[?\d+\]?\.?\s*", "", ref)
        segment = re.split(r"\(\d{4}\)|\b\d{4}\b", segment, maxsplit=1)[0]
        segment = re.sub(r"\bet\s+al\.?", "", segment, flags=re.IGNORECASE)
        segment = segment.replace(";", ",")
        segment = re.sub(r"\s+(and|&)\s+", ",", segment, flags=re.IGNORECASE)
        raw_names = [p.strip() for p in segment.split(",") if p.strip()]

        for raw in raw_names:
            cleaned = re.sub(r"[^A-Za-z\-\s\.]", "", raw).strip(" .")
            tokens = cleaned.split()
            if len(tokens) < 2:
                continue
            # Keep likely person names only
            if any(t.lower() in {"et", "al", "proceedings", "conference", "journal"} for t in tokens):
                continue

            name = " ".join(tokens[:4])
            key = re.sub(r"\s+", " ", name.lower())
            if key not in candidates:
                candidates[key] = CandidateReviewer(name=name, evidence_references=[ref])
            else:
                candidates[key].evidence_references.append(ref)

    return list(candidates.values())
