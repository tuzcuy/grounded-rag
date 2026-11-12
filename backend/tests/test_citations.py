from grounded.answer.citations import (
    extract_citations,
    resolve_citations,
    split_sentences,
    strip_citations,
)


def test_split_sentences_basic():
    text = "First sentence. Second one! Third? And a fourth."
    assert len(split_sentences(text)) == 4


def test_split_keeps_abbrev_like_numbers_together():
    text = "The value is 3.5 volts. It charges fast."
    assert split_sentences(text) == ["The value is 3.5 volts.", "It charges fast."]


def test_extract_citations_in_order_without_duplicates():
    assert extract_citations("Fact [2][1][2].", context_size=5) == [2, 1]


def test_out_of_range_citations_dropped():
    assert extract_citations("Fact [7].", context_size=3) == []


def test_strip_citations():
    assert strip_citations("The token expires [1][2].") == "The token expires ."


def test_resolve_attaches_trailing_citation_block_to_previous_sentence():
    resolved = resolve_citations("Tokens expire after an hour. [1][3]", context_size=4)
    assert len(resolved) == 1
    assert resolved[0][1] == [1, 3]
