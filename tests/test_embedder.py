"""
tests/test_embedder.py

Unit tests for SchemaEmbedder (embeddings/embedder.py), which calls the
HuggingFace Inference API for every embedding -- no local model download.

The actual HTTP call (requests.Session.post) is monkeypatched so this test
suite runs with zero network access and zero HF token, same as the rest of
CI. A separate, manual smoke test against the real API is the right way to
validate actual embedding quality (see demo_phase1.py), since that's an
external-service concern, not a unit-test concern.
"""

from __future__ import annotations

import numpy as np
import pytest

from embeddings.embedder import EmbeddingAPIError, SchemaEmbedder


class _FakeResponse:
    def __init__(self, status_code: int, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Every test gets a clean SchemaEmbedder singleton (no leaked state
    from a previous test's monkeypatched session/token).
    """
    SchemaEmbedder.reset()
    yield
    SchemaEmbedder.reset()


def test_missing_token_raises_clear_error():
    # LLMSettings is frozen/immutable by design (see config/settings.py),
    # so rather than mutate global settings we just pass an explicit empty
    # token straight to the constructor -- exercising the same guard clause.
    with pytest.raises(EmbeddingAPIError, match="HF_API_TOKEN"):
        SchemaEmbedder(api_token="")


def test_embed_calls_inference_api_and_returns_normalized_vectors(monkeypatch):
    calls = []

    def fake_post(self, url, json, timeout):
        calls.append((url, json))
        # Two 4-dim "embeddings" for two input texts, un-normalized on purpose
        # to prove the client normalizes them itself.
        return _FakeResponse(200, json_data=[[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    monkeypatch.setattr("requests.Session.post", fake_post)

    embedder = SchemaEmbedder(model_name="fake/model", api_token="fake-token")
    vectors = embedder.embed(["table one text", "table two text"])

    assert vectors.shape == (2, 4)
    # 3-4-0-0 normalized should have unit norm.
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(vectors[1]), 1.0, atol=1e-5)

    # Confirms the request hit the feature-extraction endpoint for the
    # configured model, with both texts batched into one call.
    url, payload = calls[0]
    assert "fake/model" in url
    assert payload["inputs"] == ["table one text", "table two text"]
    assert payload["options"]["wait_for_model"] is True


def test_embed_one_returns_single_vector(monkeypatch):
    def fake_post(self, url, json, timeout):
        return _FakeResponse(200, json_data=[1.0, 0.0, 0.0])

    monkeypatch.setattr("requests.Session.post", fake_post)

    embedder = SchemaEmbedder(model_name="fake/model", api_token="fake-token")
    vector = embedder.embed_one("a single question")

    assert vector.shape == (3,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_empty_input_returns_empty_array(monkeypatch):
    embedder = SchemaEmbedder(model_name="fake/model", api_token="fake-token")
    vectors = embedder.embed([])
    assert vectors.shape == (0, 384)  # falls back to configured EMBEDDING_DIM


def test_retries_on_failure_then_succeeds(monkeypatch):
    attempts = {"count": 0}

    def flaky_post(self, url, json, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return _FakeResponse(503, text="model loading")
        return _FakeResponse(200, json_data=[[1.0, 1.0]])

    monkeypatch.setattr("requests.Session.post", flaky_post)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)  # skip real backoff delay in tests

    embedder = SchemaEmbedder(model_name="fake/model", api_token="fake-token")
    vectors = embedder.embed(["retry me"])

    assert attempts["count"] == 3
    assert vectors.shape == (1, 2)


def test_raises_after_exhausting_retries(monkeypatch):
    def always_fails(self, url, json, timeout):
        return _FakeResponse(500, text="server error")

    monkeypatch.setattr("requests.Session.post", always_fails)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    embedder = SchemaEmbedder(model_name="fake/model", api_token="fake-token")
    with pytest.raises(EmbeddingAPIError, match="failed after 3 attempts"):
        embedder.embed(["this will fail"])