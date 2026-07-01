"""
tests/fakes.py

A deterministic, dependency-free stand-in for SchemaEmbedder used in tests
and CI so the test suite never needs network access to download real model
weights. Uses word-hashing (a simplified hashing-trick bag-of-words) into a
fixed-size vector -- crude compared to a real sentence transformer, but
good enough to prove retrieval mechanics: texts sharing distinctive words
(e.g. "customer", "employee") land closer together than unrelated ones.

Real semantic quality is validated separately/manually against the actual
sentence-transformers model; this fake only needs to validate FAISS
plumbing (build/save/load/query), not embedding quality.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np


class FakeEmbedder:
    dim = 384  # matches config.settings.EmbeddingSettings.dimension default

    def _hash_bucket(self, word: str) -> int:
        h = hashlib.md5(word.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dim

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        words = re.findall(r"[a-zA-Z]+", text.lower())
        for w in words:
            vec[self._hash_bucket(w)] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.stack([self._vectorize(t) for t in texts])

    def embed_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self._vectorize(text)