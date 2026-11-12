from grounded.answer import make_provider


def _hits(index, question, k=4):
    hits = index.search(question, top_k=k)
    for hit in hits:
        hit._backend = index.backend
    return hits


def test_extractive_quotes_carry_citations(tiny_index):
    provider = make_provider("extractive")
    answer = provider.answer("how do I restart the widget", _hits(tiny_index, "restart widget"))
    assert answer.sentences
    for sentence in answer.sentences:
        assert sentence.citations, sentence.text
        assert sentence.text.startswith("“")


def test_extractive_is_deterministic(tiny_index):
    provider = make_provider("extractive")
    hits = _hits(tiny_index, "battery charging time")
    first = provider.answer("battery charging time", hits)
    second = provider.answer("battery charging time", hits)
    assert first.text == second.text


def test_extractive_with_no_hits_degrades_gracefully(tiny_index):
    provider = make_provider("extractive")
    answer = provider.answer("anything", [])
    assert "No relevant passages" in answer.text
    assert answer.sentences == []


def test_citation_positions_point_into_hit_list(tiny_index):
    provider = make_provider("extractive")
    hits = _hits(tiny_index, "warranty claims receipt")
    answer = provider.answer("what do warranty claims require", hits)
    for sentence in answer.sentences:
        for pos in sentence.citations:
            assert 1 <= pos <= len(hits)


def test_unknown_provider_rejected():
    import pytest

    with pytest.raises(ValueError):
        make_provider("telepathy")
