from grounded.retrieval.bm25 import BM25, tokenize

DOCS = [
    "the cat sat on the mat",
    "dogs chase cats in the park",
    "quantum entanglement of photon pairs",
    "the park has a quantum playground for cats and dogs",
]


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! 42x") == ["hello", "world", "42x"]


def test_exact_topic_match_ranks_first():
    bm25 = BM25(DOCS)
    ranking = bm25.rank("photon entanglement")
    assert ranking[0][0] == 2


def test_rare_terms_outweigh_common_terms():
    bm25 = BM25(DOCS)
    # "quantum" appears in 2 docs, "the" in 3 — a quantum query must prefer quantum docs.
    ranking = [i for i, _ in bm25.rank("quantum")]
    assert set(ranking[:2]) == {2, 3}


def test_idf_is_positive_even_for_ubiquitous_terms():
    bm25 = BM25(["common word", "common word", "common word"])
    assert bm25.idf["common"] > 0


def test_length_normalization_prefers_focused_documents():
    short = "espresso brewing"
    long = "espresso " + "filler " * 200
    bm25 = BM25([short, long])
    assert bm25.score("espresso", 0) > bm25.score("espresso", 1)


def test_query_with_no_matches_returns_empty():
    bm25 = BM25(DOCS)
    assert bm25.rank("zeppelin") == []


def test_top_k_limits_results():
    bm25 = BM25(DOCS)
    assert len(bm25.rank("the cats", top_k=2)) == 2
