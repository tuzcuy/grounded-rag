"""Embedding backends behind one tiny protocol.

- SentenceTransformerBackend: the real model (all-MiniLM-L6-v2 by default).
- HashingEmbedder: a deterministic, dependency-light embedder used by the
  test suite (and available offline). It is NOT semantically meaningful —
  it hashes character n-grams — but it is stable, which is exactly what
  tests need.
"""

import hashlib

import numpy as np


class EmbeddingBackend:
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        raise NotImplementedError


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder(EmbeddingBackend):
    def __init__(self, dim: int = 256, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text.lower()}  "
            for i in range(len(padded) - self.ngram + 1):
                gram = padded[i : i + self.ngram]
                digest = hashlib.blake2b(gram.encode(), digest_size=4).digest()
                out[row, int.from_bytes(digest, "little") % self.dim] += 1.0
        return _normalize(out)


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None)
        if get_dim is None:  # older sentence-transformers releases
            get_dim = self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        return _normalize(np.asarray(vectors, dtype=np.float32))


def make_backend(kind: str, st_model: str) -> EmbeddingBackend:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "st":
        return SentenceTransformerBackend(st_model)
    raise ValueError(f"unknown embedding backend: {kind}")
