"""Reciprocal Rank Fusion — combines any number of rankings without needing
their scores to be on the same scale (BM25 scores and cosine similarities
are not comparable; ranks are)."""


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """rankings: lists of ids, best first. Returns (id, fused_score), best first.

    k dampens the contribution of lower ranks; 60 is the value from the
    original RRF paper and works well untouched.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered
