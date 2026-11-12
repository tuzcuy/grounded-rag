"""A small, dependency-free BM25 (Okapi) implementation.

Written from scratch instead of pulling a library so the ranking math is a
unit-tested, inspectable part of this codebase rather than a black box.
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.term_freqs = [Counter(t) for t in self.doc_tokens]
        df: Counter[str] = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        n = len(self.doc_tokens)
        # BM25+-style idf floor: ln(1 + (N - df + 0.5)/(df + 0.5)) is always > 0
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str, doc_id: int) -> float:
        tf = self.term_freqs[doc_id]
        dl = self.doc_len[doc_id]
        score = 0.0
        for term in tokenize(query):
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1.0))
            score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
        return score

    def rank(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.doc_tokens))]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return scores[:top_k] if top_k else scores
