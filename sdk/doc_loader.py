from pathlib import Path
from typing import Optional
from pypdf import PdfReader


def load_pdf_text(path: str | Path, max_chars: Optional[int] = 20000) -> str:
    """
    This fxn extracts the pdf text.
    max_chars trims very long docs to avoid blowing up the context window
    """
    path = Path(path)
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)

    full = ('\n\n'.join(pages))
    if max_chars is not None and len(full) > max_chars:
        return full[:max_chars] + '\n\n[TRUNCATED]'

    return full
