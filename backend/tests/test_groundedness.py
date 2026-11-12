from grounded.groundedness import answer_groundedness, content_tokens, sentence_support

PASSAGES = [
    "The gadget battery lasts for twelve hours of continuous recording and "
    "charges fully in ninety minutes using the bundled charger.",
    "Warranty claims require the original purchase receipt.",
]


def test_copied_sentence_is_fully_supported():
    score = sentence_support("The gadget battery lasts for twelve hours.", PASSAGES)
    assert score >= 0.9


def test_fabricated_sentence_scores_low():
    score = sentence_support("The spaceship engine burns dilithium crystals.", PASSAGES)
    assert score < 0.3


def test_stopwords_do_not_count_as_support():
    assert content_tokens("the and of is") == set()
    assert sentence_support("the and of is", PASSAGES) == 1.0  # nothing claimed


def test_answer_level_aggregation_flags_the_bad_sentence():
    result = answer_groundedness(
        [
            "Warranty claims require the original purchase receipt.",
            "The warranty also covers meteor strikes and alien damage.",
        ],
        PASSAGES,
    )
    assert result["supported"][0] is True
    assert result["supported"][1] is False
    assert 0.0 < result["score"] < 1.0


def test_empty_answer_scores_zero():
    assert answer_groundedness([], PASSAGES)["score"] == 0.0
