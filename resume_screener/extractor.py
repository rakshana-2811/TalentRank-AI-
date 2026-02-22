import os
from typing import Dict
from pypdf import PdfReader


def extract_text_from_pdf(path: str) -> str:
    """Extract and return text from a single PDF file."""
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception:
        return ""
    return "\n".join(text_parts)


def extract_texts_from_dir(directory: str) -> Dict[str, str]:
    """Extract text from all PDF files in `directory`.

    Returns a dict mapping filename -> extracted text.
    """
    results = {}
    for entry in os.listdir(directory):
        if entry.lower().endswith(".pdf"):
            path = os.path.join(directory, entry)
            text = extract_text_from_pdf(path)
            results[entry] = text
    return results
