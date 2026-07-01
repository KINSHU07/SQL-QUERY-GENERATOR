"""
database/connectors/sqlite_connector.py

SQLite connector. Included in Phase 1 purely so the schema extractor and
FAISS pipeline can be demoed/tested with zero setup (no MySQL server
required) -- the extractor code is dialect-agnostic (SQLAlchemy Inspector),
so this also doubles as a correctness check that Phase 1 isn't accidentally
MySQL-specific.
"""

from __future__ import annotations

from database.connectors.base import BaseConnector


class SQLiteConnector(BaseConnector):
    def __init__(self, path: str = ":memory:") -> None:
        super().__init__()
        self.path = path

    @property
    def sqlalchemy_url(self) -> str:
        return f"sqlite:///{self.path}"