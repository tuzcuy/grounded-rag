"""A deliberately simple, LLM-free groundedness check.

For every answer sentence we compute the best content-token overlap against
the retrieved passages. It cannot judge paraphrase quality the way an
LLM-as-judge could — but it is deterministic, free, and catches the failure
mode that matters most in a citing system: sentences that talk about things
no retrieved passage contains.
"""

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "with", "as",
    "by", "at", "from", "not", "no", "if", "then", "than", "but", "which",
    "must", "may", "can", "shall", "should", "will", "would", "has", "have",
}

SUPPORT_THRESHOLD = 0.5


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def sentence_support(sentence: str, passages: list[str]) -> float:
    """Best overlap ratio of the sentence's content tokens inside any passage."""
    tokens = content_tokens(sentence)
    if not tokens:
        return 1.0  # nothing substantive claimed
    best = 0.0
    for passage in passages:
        overlap = len(tokens & content_tokens(passage)) / len(tokens)
        best = max(best, overlap)
    return best


def answer_groundedness(sentences: list[str], passages: list[str]) -> dict:
    scores = [sentence_support(s, passages) for s in sentences]
    supported = [s >= SUPPORT_THRESHOLD for s in scores]
    return {
        "per_sentence": scores,
        "supported": supported,
        "score": (sum(scores) / len(scores)) if scores else 0.0,
    }
