"""
embeddings/schema_indexer.py

Orchestration layer that ties a DB connector -> SchemaExtractor ->
FAISSSchemaStore together. This is the single entry point the FastAPI
backend (Phase 3) and the LangGraph "retrieve relevant tables" node
(Phase 2) will both call, so schema extraction + indexing logic lives in
exactly one place.
"""

from __future__ import annotations

from database.connectors.base import BaseConnector
from database.schema_extractor import SchemaExtractor, TableSchema
from embeddings.faiss_store import FAISSSchemaStore, make_db_id


class SchemaIndexer:
    def __init__(self, connector: BaseConnector) -> None:
        self.connector = connector
        self.db_id = make_db_id(connector.sqlalchemy_url)
        self.store = FAISSSchemaStore(db_id=self.db_id)

    def sync(self, force_rebuild: bool = False, estimate_row_counts: bool = False) -> list[TableSchema]:
        """Ensures a FAISS index exists for this database. If one is
        already persisted on disk and `force_rebuild` is False, it's
        loaded instead of re-extracting/re-embedding -- keeps repeat
        connects fast. Returns the extracted TableSchema list either way
        (re-extracted fresh from the live DB, since schema metadata itself
        is cheap to read even when we skip re-embedding).
        """
        extractor = SchemaExtractor(self.connector.engine)
        tables = extractor.extract(include_views=True, estimate_row_counts=estimate_row_counts)

        if force_rebuild or not self.store.exists():
            self.store.build_from_schema(tables)
        else:
            self.store.load()

        return tables

    def retrieve_relevant_tables(self, question: str, top_k: int | None = None):
        """Returns [(table_name, text_block, score), ...] for the tables
        most relevant to a natural-language question. This is what the
        LangGraph "retrieve" node (Phase 2) will call directly.
        """
        results = self.store.query(question, top_k=top_k)
        return [(chunk.table_name, chunk.text, score) for chunk, score in results]