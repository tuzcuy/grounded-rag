import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Build a real (hashing-embedder) index on disk and boot the app against it.
    from grounded.retrieval import Index
    from grounded.retrieval.embeddings import HashingEmbedder
    from tests.conftest import DOC_A, DOC_B

    corpus = tmp_path_factory.mktemp("api_corpus")
    (corpus / "widgets.md").write_text(DOC_A, encoding="utf-8")
    (corpus / "gadgets.md").write_text(DOC_B, encoding="utf-8")
    # Default dim: must match what the app's lifespan constructs for "hashing".
    index_dir = tmp_path_factory.mktemp("api_index")
    Index.build(corpus, HashingEmbedder()).save(index_dir)

    os.environ["GROUNDED_INDEX_DIR"] = str(index_dir)
    os.environ["GROUNDED_EMBEDDINGS"] = "hashing"
    from grounded.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_chunk_count(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0


def test_stats_lists_documents(client):
    body = client.get("/api/stats").json()
    assert set(body["documents"]) == {"widgets.md", "gadgets.md"}


def test_query_returns_answer_chunks_and_groundedness(client):
    response = client.post(
        "/api/query", json={"question": "how long does the battery last?", "k": 4}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "extractive"
    assert body["chunks"] and body["chunks"][0]["position"] == 1
    assert 0.0 <= body["groundedness"] <= 1.0
    for sentence in body["sentences"]:
        for cite in sentence["citations"]:
            assert 1 <= cite <= len(body["chunks"])


def test_query_validation_rejects_short_question(client):
    assert client.post("/api/query", json={"question": "a"}).status_code == 422


def test_query_validation_rejects_bad_mode(client):
    response = client.post("/api/query", json={"question": "valid question", "mode": "psychic"})
    assert response.status_code == 422


def test_llm_provider_without_key_is_a_clean_400(client):
    os.environ.pop("OPENAI_API_KEY", None)
    response = client.post(
        "/api/query", json={"question": "battery life?", "provider": "openai"}
    )
    assert response.status_code == 400
    assert "OPENAI_API_KEY" in response.json()["detail"]
