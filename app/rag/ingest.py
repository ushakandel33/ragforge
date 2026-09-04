"""
Document ingestion: load raw text (txt/pdf) and split it into overlapping
chunks suitable for embedding + retrieval.
"""
from pathlib import Path
from pypdf import PdfReader
from app.config import settings


def load_text_from_file(path: str) -> str:
    """Extract raw text from a .txt, .md, or .pdf file."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> list[str]:
    """
    Simple, dependency-free sliding-window chunker operating on whitespace
    tokens. Overlap preserves context continuity across chunk boundaries,
    which improves retrieval recall for facts that straddle a split point.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    words = text.split()

    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def ingest_file(path: str) -> list[dict]:
    """
    Full ingestion pipeline for a single file: extract -> chunk -> attach
    metadata. Returns a list of {"text":..., "metadata":...} dicts ready
    for embedding.
    """
    text = load_text_from_file(path)
    chunks = chunk_text(text)
    filename = Path(path).name
    return [
        {"text": chunk, "metadata": {"source": filename, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]
