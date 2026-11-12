"""Sentence splitting and [n] citation handling."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_CITATION = re.compile(r"\[(\d{1,2})\]")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip())]
    return [p for p in parts if p]


def extract_citations(sentence: str, context_size: int) -> list[int]:
    """Return the 1-based citation markers in a sentence, dropping anything
    that does not point into the provided context."""
    seen: list[int] = []
    for raw in _CITATION.findall(sentence):
        n = int(raw)
        if 1 <= n <= context_size and n not in seen:
            seen.append(n)
    return seen


def strip_citations(sentence: str) -> str:
    return _CITATION.sub("", sentence).strip()


def resolve_citations(text: str, context_size: int) -> list[tuple[str, list[int]]]:
    """Split an answer into sentences and attach the validated citations of
    each sentence. A trailing bare citation block ("[1][3]") belongs to the
    sentence before it."""
    out: list[tuple[str, list[int]]] = []
    for sentence in split_sentences(text):
        cites = extract_citations(sentence, context_size)
        if not strip_citations(sentence) and out and cites:
            prev_text, prev_cites = out[-1]
            merged = prev_cites + [c for c in cites if c not in prev_cites]
            out[-1] = (prev_text, merged)
            continue
        out.append((sentence, cites))
    return out
