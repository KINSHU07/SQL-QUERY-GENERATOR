"""
embeddings/embedder.py

HuggingFace Inference API-based embedder. No local model weights are ever
downloaded onto this machine -- every embedding call goes over the network
to HF's hosted Inference API for the configured model (default:
sentence-transformers/all-MiniLM-L6-v2). This matches the project-wide
constraint that ALL HuggingFace models (embeddings and the SQL-generation
LLM alike) are called via API only, never pulled locally.

Trade-off vs local embedding (documented, not hidden): every table-schema
sync and every user question now costs one network round-trip and is
subject to HF's free-tier rate limits and cold-start ("model loading")
latency on first call. build_from_schema() batches all tables into a
single request to minimize call count; query() necessarily makes one call
per question.
"""

from __future__ import annotations

import time

import numpy as np
import requests

from config.settings import settings


class EmbeddingAPIError(Exception):
    """Raised when the HF Inference API embedding call fails after retries."""


class SchemaEmbedder:
    """Thin HTTP client for HF's feature-extraction Inference API.

    Kept as a process-wide singleton so the requests.Session (and its auth
    header) is set up once, not per call.
    """

    _instance: "SchemaEmbedder | None" = None

    API_URL_TEMPLATE = "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"

    def __init__(self, model_name: str | None = None, api_token: str | None = None) -> None:
        self.model_name = model_name or settings.embedding.model_name
        self.api_token = api_token or settings.llm.hf_api_token
        if not self.api_token:
            raise EmbeddingAPIError(
                "HF_API_TOKEN is required to call the HuggingFace Inference API "
                "for embeddings (no local model download is used). Set it in "
                "your .env -- see .env.example. Get a free token at "
                "https://huggingface.co/settings/tokens"
            )
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_token}"})
        self._url = self.API_URL_TEMPLATE.format(model=self.model_name)

    @classmethod
    def get(cls) -> "SchemaEmbedder":
        """Process-wide singleton accessor -- avoids re-creating the HTTP
        session (and re-reading config) on every request.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clears the singleton. Used by tests so each test can inject a
        fresh token/model without leaking state across test cases.
        """
        cls._instance = None

    def _post(self, inputs: list[str], max_retries: int = 3, backoff_seconds: float = 2.0) -> list:
        """POSTs to the Inference API with retry/backoff.

        Free-tier HF Inference endpoints cold-start a model on first use
        and return a 503 while it loads; `wait_for_model=True` tells HF to
        hold the request open until the model is ready instead of failing
        immediately, and the retry loop is a second line of defense for
        transient network errors or rate limiting.
        """
        payload = {"inputs": inputs, "options": {"wait_for_model": True}}
        last_error = "unknown error"

        for attempt in range(max_retries):
            try:
                response = self._session.post(self._url, json=payload, timeout=30)
            except requests.RequestException as exc:
                last_error = str(exc)
            else:
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"

            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (attempt + 1))

        raise EmbeddingAPIError(
            f"HF Inference API call to '{self.model_name}' failed after "
            f"{max_retries} attempts: {last_error}"
        )

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Returns an (n, dim) float32 array of sentence embeddings.

        HF's Inference API for sentence-transformers-tagged models returns
        already mean-pooled sentence embeddings (one vector per input
        text), so no local pooling/tokenization step is needed here --
        the whole point is that nothing runs on this machine.
        """
        if not texts:
            return np.zeros((0, settings.embedding.dimension), dtype="float32")

        raw = self._post(texts)
        vectors = np.array(raw, dtype="float32")

        if vectors.ndim == 1:
            # A single input can come back as one flat vector; normalize shape.
            vectors = vectors.reshape(1, -1)

        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # avoid divide-by-zero on an all-zero vector
            vectors = vectors / norms

        return vectors

    def embed_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.embed([text], normalize=normalize)[0]