"""Retrieval + citation evaluation against the golden set.

    python -m eval [--corpus ../corpus] [--k 10]

Builds the index with the configured embedding backend, evaluates bm25-only,
dense-only and hybrid retrieval, then runs the extractive provider end to end
for citation precision and groundedness. Results land in eval/results.json —
the README table is generated from that file, never typed by hand.
"""

import argparse
import json
from pathlib import Path

from grounded.answer import make_provider
from grounded.answer.citations import strip_citations
from grounded.config import get_settings
from grounded.groundedness import answer_groundedness
from grounded.retrieval import Index
from grounded.retrieval.embeddings import make_backend

HERE = Path(__file__).parent


def _flat(text: str) -> str:
    return " ".join(text.split()).lower()


def relevant(hit, item) -> bool:
    # Whitespace-normalized match: RFC text renderings wrap lines mid-phrase.
    return hit.chunk.doc == item["doc"] and _flat(item["must_contain"]) in _flat(hit.chunk.text)


def eval_retrieval(index: Index, golden: list[dict], k: int) -> dict:
    out = {}
    for mode in ("bm25", "dense", "hybrid"):
        hit1 = hit5 = 0
        rr_total = 0.0
        for item in golden:
            hits = index.search(item["question"], top_k=k, mode=mode)
            rank = next((i for i, h in enumerate(hits, 1) if relevant(h, item)), None)
            if rank is not None:
                rr_total += 1.0 / rank
                hit1 += int(rank == 1)
                hit5 += int(rank <= 5)
        n = len(golden)
        out[mode] = {
            "hit@1": round(hit1 / n, 3),
            "hit@5": round(hit5 / n, 3),
            f"mrr@{k}": round(rr_total / n, 3),
        }
    return out


def eval_answers(index: Index, golden: list[dict], k: int) -> dict:
    provider = make_provider("extractive")
    cited = cited_relevant = 0
    groundedness_sum = 0.0
    for item in golden:
        hits = index.search(item["question"], top_k=k, mode="hybrid")
        for hit in hits:
            hit._backend = index.backend
        answer = provider.answer(item["question"], hits)
        for sentence in answer.sentences:
            for pos in sentence.citations:
                cited += 1
                cited_relevant += int(relevant(hits[pos - 1], item))
        plain = [strip_citations(s.text).strip("“”\"") for s in answer.sentences]
        groundedness_sum += answer_groundedness(plain, [h.chunk.text for h in hits])["score"]
    n = len(golden)
    return {
        "provider": "extractive",
        "citations_total": cited,
        "citation_precision": round(cited_relevant / cited, 3) if cited else 0.0,
        "mean_groundedness": round(groundedness_sum / n, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=HERE.parent.parent / "corpus")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    backend = make_backend(settings.embeddings, settings.st_model)
    index = Index.build(args.corpus, backend)
    golden = json.loads((HERE / "golden.json").read_text(encoding="utf-8"))

    results = {
        "n_questions": len(golden),
        "n_chunks": len(index.chunks),
        "embedding_backend": settings.embeddings,
        "retrieval": eval_retrieval(index, golden, args.k),
        "answering": eval_answers(index, golden, args.k),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
