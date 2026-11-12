"""FastAPI service: /api/query runs retrieval + answering + groundedness."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .answer import make_provider
from .answer.citations import strip_citations
from .config import get_settings
from .groundedness import answer_groundedness
from .retrieval import Index
from .retrieval.embeddings import make_backend


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    backend = make_backend(settings.embeddings, settings.st_model)
    index_dir = Path(settings.index_dir)
    if not (index_dir / "chunks.json").exists():
        raise RuntimeError(
            f"No index at {index_dir}. Build one first: python -m grounded.index_cli ../corpus"
        )
    app.state.index = Index.load(index_dir, backend)
    app.state.settings = settings
    yield


app = FastAPI(title="Grounded", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=8, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|bm25|dense)$")
    provider: str | None = Field(default=None, pattern="^(extractive|openai|anthropic|ollama)$")


class ChunkOut(BaseModel):
    position: int
    doc: str
    heading_path: list[str]
    start_line: int
    end_line: int
    text: str
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None
    dense_similarity: float | None


class SentenceOut(BaseModel):
    text: str
    citations: list[int]
    support: float
    supported: bool


class QueryResponse(BaseModel):
    provider: str
    answer: str
    sentences: list[SentenceOut]
    groundedness: float
    chunks: list[ChunkOut]


@app.get("/api/health")
def health():
    index: Index = app.state.index
    return {"status": "ok", "version": __version__, "chunks": len(index.chunks)}


@app.get("/api/stats")
def stats():
    index: Index = app.state.index
    docs: dict[str, int] = {}
    for chunk in index.chunks:
        docs[chunk.doc] = docs.get(chunk.doc, 0) + 1
    return {"documents": docs, "chunks": len(index.chunks)}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    index: Index = app.state.index
    settings = app.state.settings
    hits = index.search(req.question, top_k=req.k, mode=req.mode)
    for hit in hits:
        hit._backend = index.backend  # extractive provider reuses the loaded model

    provider_name = req.provider or settings.provider
    try:
        answer = make_provider(provider_name).answer(req.question, hits)
    except RuntimeError as exc:  # missing API key etc.
        raise HTTPException(status_code=400, detail=str(exc))

    passages = [h.chunk.text for h in hits]
    plain = [strip_citations(s.text).strip("“”\"") for s in answer.sentences]
    g = answer_groundedness(plain, passages)

    return QueryResponse(
        provider=answer.provider,
        answer=answer.text,
        sentences=[
            SentenceOut(
                text=s.text,
                citations=s.citations,
                support=round(g["per_sentence"][i], 3),
                supported=g["supported"][i],
            )
            for i, s in enumerate(answer.sentences)
        ],
        groundedness=round(g["score"], 3),
        chunks=[
            ChunkOut(
                position=pos,
                doc=h.chunk.doc,
                heading_path=h.chunk.heading_path,
                start_line=h.chunk.start_line,
                end_line=h.chunk.end_line,
                text=h.chunk.text,
                rrf_score=round(h.rrf_score, 5),
                bm25_rank=h.bm25_rank,
                dense_rank=h.dense_rank,
                dense_similarity=round(h.dense_similarity, 4)
                if h.dense_similarity is not None
                else None,
            )
            for pos, h in enumerate(hits, start=1)
        ],
    )


# Serve the built frontend when present (single-container deployments).
# Mounted last so the /api routes above always win.
_static = Path(os.environ.get("GROUNDED_STATIC", "static"))
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
