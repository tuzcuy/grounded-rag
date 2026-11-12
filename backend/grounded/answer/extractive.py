"""Keyless answer provider: selects the sentences of the retrieved passages
that are most similar to the question and quotes them with citations.

It will never phrase anything the documents did not say — which makes it
both a safe default and a useful baseline when judging what an LLM adds.
"""

from ..retrieval import SearchHit
from .base import Answer, AnswerSentence, Provider
from .citations import split_sentences

MAX_SENTENCES = 4
MIN_SENTENCE_CHARS = 40


class ExtractiveProvider(Provider):
    name = "extractive"

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(provider=self.name, text="No relevant passages were found.")

        backend = _backend_of(hits)
        candidates: list[tuple[str, int]] = []  # (sentence, context position 1-based)
        for pos, hit in enumerate(hits, start=1):
            for sentence in split_sentences(hit.chunk.text):
                if len(sentence) >= MIN_SENTENCE_CHARS:
                    candidates.append((sentence, pos))

        if not candidates:
            return Answer(provider=self.name, text="No quotable sentences in the retrieved passages.")

        import numpy as np

        texts = [s for s, _ in candidates]
        sentence_vecs = backend.encode(texts)
        question_vec = backend.encode([question])[0]
        sims = sentence_vecs @ question_vec
        order = np.argsort(-sims)

        picked: list[tuple[str, int]] = []
        seen_positions: set[int] = set()
        for idx in order:
            sentence, pos = candidates[int(idx)]
            # prefer coverage: at most two quotes from the same passage
            if sum(1 for _, p in picked if p == pos) >= 2:
                continue
            picked.append((sentence, pos))
            seen_positions.add(pos)
            if len(picked) >= MAX_SENTENCES:
                break

        picked.sort(key=lambda pair: pair[1])
        sentences = [
            AnswerSentence(text=f"“{s}” [{pos}]", citations=[pos]) for s, pos in picked
        ]
        text = "\n".join(s.text for s in sentences)
        return Answer(provider=self.name, text=text, sentences=sentences)


def _backend_of(hits: list[SearchHit]):
    # The index that produced the hits carries the embedding backend; the API
    # layer passes it through via this attribute to avoid a second model load.
    backend = getattr(hits[0], "_backend", None)
    if backend is None:
        raise RuntimeError("SearchHit missing embedding backend reference")
    return backend
