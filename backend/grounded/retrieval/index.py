"""The hybrid index: BM25 + dense vectors, fused with RRF.

Dense search is a plain normalized dot product over a numpy matrix — for
corpora under ~100k chunks this is faster to operate and reason about than
an approximate-NN service, and it keeps the project fully self-contained.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..ingest import Chunk, chunk_document, load_document
from .bm25 import BM25
from .embeddings import EmbeddingBackend
from .fusion import reciprocal_rank_fusion

CANDIDATES_PER_CHANNEL = 50


@dataclass
class SearchHit:
    chunk: Chunk
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None
    dense_similarity: float | None


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, backend: EmbeddingBackend):
        self.chunks = chunks
        self.vectors = vectors
        self.backend = backend
        self.bm25 = BM25([c.text for c in chunks])

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, corpus_dir: Path, backend: EmbeddingBackend) -> "Index":
        chunks: list[Chunk] = []
        for path in sorted(corpus_dir.iterdir()):
            if path.suffix.lower() not in {".md", ".markdown", ".txt", ".pdf"}:
                continue
            text = load_document(path)
            chunks.extend(chunk_document(path.name, text))
        for i, chunk in enumerate(chunks):
            chunk.id = i
        vectors = backend.encode([c.text for c in chunks])
        return cls(chunks, vectors, backend)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "vectors.npy", self.vectors)
        payload = [asdict(c) for c in self.chunks]
        (out_dir / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, backend: EmbeddingBackend) -> "Index":
        vectors = np.load(index_dir / "vectors.npy")
        if vectors.shape[1] != backend.dim:
            raise ValueError(
                f"index was built with {vectors.shape[1]}-dim embeddings but the "
                f"configured backend produces {backend.dim}-dim — rebuild the index "
                "or fix GROUNDED_EMBEDDINGS/GROUNDED_ST_MODEL"
            )
        raw = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        return cls(chunks, vectors, backend)

    # -- search ------------------------------------------------------------

    def search(self, query: str, top_k: int = 8, mode: str = "hybrid") -> list[SearchHit]:
        bm25_ranking = [i for i, _ in self.bm25.rank(query, CANDIDATES_PER_CHANNEL)]

        query_vec = self.backend.encode([query])[0]
        sims = self.vectors @ query_vec
        dense_order = np.argsort(-sims)[:CANDIDATES_PER_CHANNEL]
        dense_ranking = [int(i) for i in dense_order]

        if mode == "bm25":
            fused = [(i, 0.0) for i in bm25_ranking]
        elif mode == "dense":
            fused = [(i, 0.0) for i in dense_ranking]
        elif mode == "hybrid":
            fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
        else:
            raise ValueError(f"unknown search mode: {mode}")

        bm25_pos = {cid: r for r, cid in enumerate(bm25_ranking)}
        dense_pos = {cid: r for r, cid in enumerate(dense_ranking)}

        hits: list[SearchHit] = []
        for chunk_id, score in fused[:top_k]:
            hits.append(
                SearchHit(
                    chunk=self.chunks[chunk_id],
                    rrf_score=score,
                    bm25_rank=bm25_pos.get(chunk_id),
                    dense_rank=dense_pos.get(chunk_id),
                    dense_similarity=float(sims[chunk_id]),
                )
            )
        return hits
