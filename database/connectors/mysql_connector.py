"""
database/connectors/mysql_connector.py

MySQL connector, built on pymysql (pure-Python, free, no system deps --
important for free-tier deploy targets like HF Spaces / Render where you
can't always install native MySQL client libraries).
"""

from __future__ import annotations

from database.connectors.base import BaseConnector
from config.settings import settings, MySQLSettings


class MySQLConnector(BaseConnector):
    def __init__(self, config: MySQLSettings | None = None) -> None:
        super().__init__()
        self.config = config or settings.mysql

    @property
    def sqlalchemy_url(self) -> str:
        return self.config.sqlalchemy_url

    def run_query(self, sql: str, params: dict | None = None, max_rows: int = 1000):
        """Adds a read-only guard on top of the base implementation: if
        `MYSQL_READ_ONLY` is set (default True), any statement that isn't a
        SELECT / SHOW / EXPLAIN / WITH is rejected before it ever reaches
        the database. This is the last line of defense in case the SQL
        validator upstream lets something slip through.
        """
        if self.config.read_only:
            stripped = sql.strip().lstrip("(").strip().upper()
            allowed_prefixes = ("SELECT", "SHOW", "EXPLAIN", "WITH", "DESCRIBE", "DESC ")
            if not stripped.startswith(allowed_prefixes):
                raise PermissionError(
                    "Read-only mode is enabled: only SELECT/SHOW/EXPLAIN/WITH "
                    "statements are allowed. Set MYSQL_READ_ONLY=false to override."
                )
        return super().run_query(sql, params=params, max_rows=max_rows)