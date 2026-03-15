import mimetypes
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from .pdf_parser import extract_text_from_pdf


MAX_PDF_SIZE_BYTES = 30 * 1024 * 1024  # 30MB


@dataclass
class PaperContext:
    paper_id: str
    source: str
    full_text: str
    references: List[str]
    title: Optional[str] = None


def detect_title(full_text: str) -> Optional[str]:
    if not full_text:
        return None
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    if not lines:
        return None
    # Heuristic: first non-trivial line before abstract-like section
    for line in lines[:10]:
        lowered = line.lower()
        if lowered in {'abstract', 'introduction'}:
            break
        if len(line) > 10:
            return line[:200]
    return lines[0][:200]


def extract_references(full_text: str) -> List[str]:
    if not full_text:
        return []
    lower_text = full_text.lower()
    idx = lower_text.rfind("references")
    if idx == -1:
        return []

    ref_block = full_text[idx:]
    lines = [line.strip() for line in ref_block.splitlines() if line.strip()]
    references: List[str] = []
    for line in lines[1:]:
        if re.match(r"^(\[?\d+\]?\.?\s+)", line) or len(references) == 0:
            references.append(line)
        elif references:
            references[-1] = f"{references[-1]} {line}".strip()
    return references


def _validate_pdf_input(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not os.path.isfile(pdf_path):
        raise ValueError(f"Path is not a file: {pdf_path}")
    if not os.access(pdf_path, os.R_OK):
        raise PermissionError(f"PDF is not readable: {pdf_path}")

    file_size = os.path.getsize(pdf_path)
    if file_size <= 0:
        raise ValueError(f"PDF is empty: {pdf_path}")
    if file_size > MAX_PDF_SIZE_BYTES:
        raise ValueError(f"PDF is too large (>30MB): {pdf_path}")

    mime_guess, _ = mimetypes.guess_type(pdf_path)
    if mime_guess not in (None, 'application/pdf') and not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"Input does not look like a PDF file: {pdf_path}")


def ingest_local_pdf(pdf_path: str) -> PaperContext:
    _validate_pdf_input(pdf_path)

    paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
    full_text = extract_text_from_pdf(pdf_path)
    references = extract_references(full_text)
    title = detect_title(full_text)
    return PaperContext(
        paper_id=paper_id,
        source=pdf_path,
        full_text=full_text,
        references=references,
        title=title,
    )
