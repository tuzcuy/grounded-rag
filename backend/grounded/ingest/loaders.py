"""File loaders. Every loader returns plain text plus the original line count,
so chunk line ranges always refer to the file the user actually opened."""

from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type: {path.name}")
    if suffix == ".pdf":
        return _load_pdf(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if _looks_like_rfc(text):
        text = _strip_rfc_page_furniture(text)
    return text


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _looks_like_rfc(text: str) -> bool:
    head = text[:2000]
    return "Request for Comments:" in head


def _strip_rfc_page_furniture(text: str) -> str:
    """Remove form feeds and the repeated page header/footer lines of RFC text
    renderings, keeping line positions meaningful for citation ranges."""
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = line.strip("\x0c").rstrip()
        if stripped.endswith("]") and "[Page" in stripped:
            cleaned.append("")  # keep line numbering stable
            continue
        if stripped.startswith("RFC ") and stripped.count("  ") >= 2:
            cleaned.append("")
            continue
        cleaned.append(line.replace("\x0c", ""))
    return "\n".join(cleaned)
