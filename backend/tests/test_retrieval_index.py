from pathlib import Path

from grounded.retrieval import Index


def test_hybrid_search_finds_lexical_match(tiny_index):
    hits = tiny_index.search("how long does the battery last", top_k=3)
    assert any("twelve hours" in h.chunk.text for h in hits)


def test_modes_produce_rankings(tiny_index):
    for mode in ("bm25", "dense", "hybrid"):
        hits = tiny_index.search("warranty receipt", top_k=3, mode=mode)
        assert hits, mode


def test_hit_carries_channel_diagnostics(tiny_index):
    hits = tiny_index.search("firmware update failed", top_k=2)
    top = hits[0]
    assert top.dense_similarity is not None
    assert top.bm25_rank is not None or top.dense_rank is not None


def test_save_and_load_roundtrip(tiny_index, tmp_path, backend):
    out = tmp_path / "index"
    tiny_index.save(out)
    loaded = Index.load(Path(out), backend)
    assert len(loaded.chunks) == len(tiny_index.chunks)
    original = tiny_index.search("recovery mode", top_k=1)[0].chunk.id
    reloaded = loaded.search("recovery mode", top_k=1)[0].chunk.id
    assert original == reloaded
