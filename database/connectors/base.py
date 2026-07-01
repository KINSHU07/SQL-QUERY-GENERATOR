"""
database/connectors/base.py

Common interface for all database connectors (MySQL now; Postgres/SQLite/
MSSQL plug in later phases without touching the schema extractor or agent
code, since everything downstream talks to this interface, not to a
specific driver).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Connection


class DBConnectionError(Exception):
    """Raised when a connector cannot reach or authenticate to the target DB."""


class BaseConnector(ABC):
    """All connectors expose:
    - a SQLAlchemy Engine (used by the schema extractor's Inspector)
    - a safe `run_query` for read-only execution
    - a `test_connection` health check
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None

    @property
    @abstractmethod
    def sqlalchemy_url(self) -> str:
        ...

    @property
    def dialect(self) -> str:
        """SQLAlchemy dialect name, e.g. 'mysql', 'sqlite', 'postgresql'."""
        return self.engine.dialect.name

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                self.sqlalchemy_url,
                pool_pre_ping=True,  # avoids stale-connection errors on free-tier hosts
                pool_recycle=1800,
            )
        return self._engine

    def test_connection(self) -> tuple[bool, str]:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "OK"
        except Exception as exc:  # noqa: BLE001 - we want to surface any driver error
            return False, str(exc)

    @contextmanager
    def _connect(self) -> Iterator[Connection]:
        try:
            with self.engine.connect() as conn:
                yield conn
        except Exception as exc:  # noqa: BLE001
            raise DBConnectionError(str(exc)) from exc

    def run_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        max_rows: int = 1000,
    ) -> list[dict[str, Any]]:
        """Execute a query and return rows as list[dict]. Row-count capped
        to `max_rows` to protect the app from accidentally huge result sets
        on shared free-tier infra.
        """
        with self._connect() as conn:
            result = conn.execute(text(sql), params or {})
            columns = list(result.keys())
            rows = []
            for i, row in enumerate(result):
                if i >= max_rows:
                    break
                rows.append(dict(zip(columns, row)))
            return rows