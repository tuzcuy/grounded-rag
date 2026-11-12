"""Heading-aware chunking.

Two heading styles are recognized:
- Markdown headings (``#``, ``##``, ...)
- Numbered section titles as used by RFCs and many specs (``4.1.2.  Title``)

The heading hierarchy travels with every chunk (``heading_path``), so a hit
deep inside a document can still be cited as
``rfc6749.txt > 4. Obtaining Authorization > 4.1. Authorization Code Grant``.
"""

import re
from dataclasses import dataclass, field

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s{1,3}(\S.{0,120})$")

MAX_CHUNK_CHARS = 1400
MIN_CHUNK_CHARS = 200


@dataclass
class Chunk:
    doc: str
    heading_path: list[str]
    text: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    id: int = field(default=-1)


def _heading_of(line: str) -> tuple[int, str] | None:
    """Return (level, title) if the line is a heading, else None."""
    m = _MD_HEADING.match(line)
    if m:
        return len(m.group(1)), m.group(2)
    m = _NUMBERED_HEADING.match(line.rstrip())
    if m:
        number, title = m.groups()
        # Guard against prose that merely starts with a number ("2119 keywords are...").
        if title and not title[0].islower() and len(title.split()) <= 12:
            level = number.count(".") + 1
            return level, f"{number}. {title.strip()}"
    return None


def chunk_document(doc_name: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 1

    def flush(end_line: int) -> None:
        nonlocal buf, buf_start
        body = "\n".join(buf).strip()
        if body:
            path = [title for _, title in heading_stack]
            chunks.extend(_split_oversized(doc_name, path, body, buf_start, end_line))
        buf = []
        buf_start = end_line + 1

    for lineno, line in enumerate(lines, start=1):
        heading = _heading_of(line)
        if heading is not None:
            flush(lineno - 1)
            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            buf_start = lineno
            continue
        buf.append(line)
    flush(len(lines))

    for i, chunk in enumerate(chunks):
        chunk.id = i
    return chunks


def _split_oversized(
    doc: str, heading_path: list[str], body: str, start_line: int, end_line: int
) -> list[Chunk]:
    """Split a section body that exceeds MAX_CHUNK_CHARS on paragraph
    boundaries, distributing line ranges proportionally per paragraph."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [Chunk(doc, list(heading_path), body, start_line, end_line)]

    paragraphs = re.split(r"\n\s*\n", body)
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = para
        else:
            current = candidate
    if current:
        if pieces and len(current) < MIN_CHUNK_CHARS:
            pieces[-1] = f"{pieces[-1]}\n\n{current}"
        else:
            pieces.append(current)

    total = sum(len(p) for p in pieces) or 1
    span = end_line - start_line + 1
    out: list[Chunk] = []
    cursor = start_line
    for i, piece in enumerate(pieces):
        share = max(1, round(span * len(piece) / total))
        last = end_line if i == len(pieces) - 1 else min(end_line, cursor + share - 1)
        out.append(Chunk(doc, list(heading_path), piece.strip(), cursor, last))
        cursor = last + 1
    return out
