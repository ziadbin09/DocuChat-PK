"""Turn uploaded files into a list of (page_number, text) pairs."""
import io
import re

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


class UnsupportedFile(ValueError):
    pass


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load(filename: str, data: bytes) -> list[tuple[int, str]]:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED:
        raise UnsupportedFile(f"Unsupported file type '{ext or filename}'. Use PDF, TXT or MD.")

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [(i, _clean(p.extract_text() or "")) for i, p in enumerate(reader.pages, start=1)]
    else:
        pages = [(1, _clean(data.decode("utf-8", errors="replace")))]

    pages = [(n, t) for n, t in pages if t]
    if not pages:
        raise UnsupportedFile(
            "No text could be extracted. If this is a scanned PDF it needs OCR first."
        )
    return pages
