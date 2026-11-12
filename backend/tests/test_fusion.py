import pytest

from grounded.retrieval.fusion import reciprocal_rank_fusion


def test_agreement_wins():
    fused = reciprocal_rank_fusion([[1, 2, 3], [1, 3, 2]])
    assert fused[0][0] == 1


def test_known_values():
    fused = dict(reciprocal_rank_fusion([[7], [7]], k=60))
    assert fused[7] == pytest.approx(2 / 61)


def test_disjoint_rankings_interleave():
    fused = [i for i, _ in reciprocal_rank_fusion([[1, 2], [3, 4]])]
    assert set(fused[:2]) == {1, 3}


def test_item_in_one_ranking_still_scores():
    fused = dict(reciprocal_rank_fusion([[5], []]))
    assert 5 in fused


def test_weights_shift_the_winner():
    fused = reciprocal_rank_fusion([[1], [2]], weights=[1.0, 3.0])
    assert fused[0][0] == 2


def test_mismatched_weights_raise():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1]], weights=[1.0, 2.0])


def test_deterministic_tiebreak_by_id():
    fused = [i for i, _ in reciprocal_rank_fusion([[9, 4]], weights=None) ]
    assert fused == [9, 4]
    tied = [i for i, _ in reciprocal_rank_fusion([[4], [9]])]
    assert tied == [4, 9]  # equal scores -> lower id first
