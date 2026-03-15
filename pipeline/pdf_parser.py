from typing import List



def _extract_with_pypdf(pdf_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    page_texts: List[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return "\n".join(page_texts).strip()



def _extract_with_pdfplumber(pdf_path: str) -> str:
    import pdfplumber

    page_texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_texts.append(page.extract_text() or "")
    return "\n".join(page_texts).strip()



def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Layered local PDF text extraction:
    1) pypdf
    2) pdfplumber fallback
    """
    errors = []

    try:
        return _extract_with_pypdf(pdf_path)
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    try:
        return _extract_with_pdfplumber(pdf_path)
    except Exception as exc:
        errors.append(f"pdfplumber: {exc}")

    raise RuntimeError(f"Failed to extract PDF text. Tried pypdf and pdfplumber. Details: {' | '.join(errors)}")
