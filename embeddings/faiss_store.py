"""
embeddings/faiss_store.py

FAISS-backed vector store for schema chunks. One index per connected
database (keyed by a `db_id`, e.g. a hash of the connection string), so
switching between databases in the UI doesn't require re-embedding the
whole schema every time -- indexes are persisted to disk and reloaded.

Uses IndexFlatIP (exact inner-product search) over normalized embeddings,
i.e. exact cosine similarity. Schemas are small (tens to low hundreds of
tables), so exact search is fast enough that there's no need for an
approximate index (IVF/HNSW) at this scale -- that's a Phase-6-scale
optimization if ever needed, not a Phase-1 concern.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from config.settings import settings
from database.schema_extractor import TableSchema
from embeddings.embedder import SchemaEmbedder


@dataclass
class SchemaChunk:
    """One retrievable unit. Phase 1 uses one chunk per table (a table's
    full column/FK/index text block, see TableSchema.to_text()). This keeps
    retrieval granularity aligned with what the LLM actually needs to see
    per matched table.
    """

    table_name: str
    text: str


class FAISSSchemaStore:
    def __init__(
        self,
        db_id: str,
        index_dir: Path | None = None,
        embedder: "SchemaEmbedder | None" = None,
    ) -> None:
        self.db_id = db_id
        self.index_dir = index_dir or settings.faiss.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index: faiss.Index | None = None
        self._chunks: list[SchemaChunk] = []
        # Injectable so tests/CI can swap in a lightweight fake embedder
        # instead of downloading real model weights over the network.
        # Production code (schema_indexer.py) leaves this None and gets
        # the real SchemaEmbedder singleton.
        self._embedder = embedder

    def _get_embedder(self) -> "SchemaEmbedder":
        if self._embedder is not None:
            return self._embedder
        return SchemaEmbedder.get()

    # -- paths -----------------------------------------------------------
    @property
    def _index_path(self) -> Path:
        return self.index_dir / f"{self.db_id}.index"

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / f"{self.db_id}.meta.pkl"

    # -- build -------------------------------------------------------------
    def build_from_schema(self, tables: list[TableSchema]) -> None:
        """Embeds every table's text representation and builds a fresh
        FAISS index. Called on initial connect and whenever the schema is
        refreshed (e.g. via a "resync schema" action in the UI).
        """
        embedder = self._get_embedder()
        chunks = [SchemaChunk(table_name=t.name, text=t.to_text()) for t in tables]
        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts, normalize=True)

        dim = vectors.shape[1] if vectors.shape[0] > 0 else settings.embedding.dimension
        index = faiss.IndexFlatIP(dim)
        if vectors.shape[0] > 0:
            index.add(vectors)

        self._index = index
        self._chunks = chunks
        self.save()

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        if self._index is None:
            raise RuntimeError("No index to save -- call build_from_schema() first.")
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self) -> bool:
        """Returns True if a persisted index was found and loaded."""
        if not self._index_path.exists() or not self._meta_path.exists():
            return False
        self._index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            self._chunks = pickle.load(f)
        return True

    def exists(self) -> bool:
        return self._index_path.exists() and self._meta_path.exists()

    # -- query ---------------------------------------------------------
    def query(self, question: str, top_k: int | None = None) -> list[tuple[SchemaChunk, float]]:
        """Returns the top_k most relevant table chunks for a natural
        language question, as (chunk, similarity_score) pairs, sorted by
        descending relevance.
        """
        if self._index is None:
            if not self.load():
                raise RuntimeError(
                    f"No FAISS index found for db_id='{self.db_id}'. "
                    "Call build_from_schema() first."
                )

        if self._index.ntotal == 0:
            return []

        top_k = top_k or settings.faiss.top_k
        top_k = min(top_k, self._index.ntotal)

        embedder = self._get_embedder()
        query_vec = embedder.embed_one(question, normalize=True).reshape(1, -1)

        scores, indices = self._index.search(query_vec, top_k)

        results: list[tuple[SchemaChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def all_tables(self) -> list[str]:
        return [c.table_name for c in self._chunks]


def make_db_id(sqlalchemy_url: str) -> str:
    """Deterministic, filesystem-safe id derived from a connection string
    (password included in the hash input but never stored in the id
    itself, so index filenames leak no credentials).
    """
    import hashlib

    digest = hashlib.sha256(sqlalchemy_url.encode("utf-8")).hexdigest()[:16]
    return digest