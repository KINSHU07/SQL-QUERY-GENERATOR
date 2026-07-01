"""
database/schema_extractor.py

Reads full schema metadata (tables, columns, types, primary keys, foreign
keys, indexes, views, row-count estimates) from a connected database using
SQLAlchemy's `Inspector`. This is dialect-agnostic: the exact same code
works for MySQL, PostgreSQL, and SQLite, which is what lets us test the
whole Phase 1 pipeline locally against SQLite while shipping MySQL as the
first supported live target.

Output is a list of `TableSchema` dataclasses -- structured, serializable,
and easy to both (a) render into text chunks for embedding and (b) render
into a compact prompt block for the LLM later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from sqlalchemy import inspect
from sqlalchemy.engine import Engine


@dataclass
class ColumnSchema:
    name: str
    type: str
    nullable: bool
    default: str | None
    is_primary_key: bool
    is_foreign_key: bool = False
    references: str | None = None  # "other_table.other_column"


@dataclass
class IndexSchema:
    name: str
    columns: list[str]
    unique: bool


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnSchema] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[dict] = field(default_factory=list)
    indexes: list[IndexSchema] = field(default_factory=list)
    is_view: bool = False
    row_count_estimate: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """Renders the table into a compact natural-language + DDL-ish
        block. This is the exact string that gets embedded and stored in
        FAISS, and later re-used as context in the SQL-generation prompt --
        keeping one canonical representation avoids drift between what the
        retriever matches on and what the LLM actually sees.
        """
        lines = [f"Table: {self.name}" + (" (view)" if self.is_view else "")]
        col_lines = []
        for col in self.columns:
            tags = []
            if col.name in self.primary_keys:
                tags.append("PK")
            if col.is_foreign_key:
                tags.append(f"FK -> {col.references}")
            if not col.nullable:
                tags.append("NOT NULL")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            col_lines.append(f"  - {col.name}: {col.type}{tag_str}")
        lines.append("Columns:")
        lines.extend(col_lines)

        if self.foreign_keys:
            lines.append("Relationships:")
            for fk in self.foreign_keys:
                cols = ", ".join(fk["constrained_columns"])
                ref_cols = ", ".join(fk["referred_columns"])
                lines.append(
                    f"  - {self.name}({cols}) -> {fk['referred_table']}({ref_cols})"
                )

        if self.indexes:
            lines.append("Indexes:")
            for idx in self.indexes:
                uniq = "UNIQUE " if idx.unique else ""
                lines.append(f"  - {uniq}{idx.name} ({', '.join(idx.columns)})")

        if self.row_count_estimate is not None:
            lines.append(f"Approx. rows: {self.row_count_estimate}")

        return "\n".join(lines)


class SchemaExtractor:
    """Extracts full schema metadata from any SQLAlchemy-compatible engine."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.inspector = inspect(engine)

    def extract(self, include_views: bool = True, estimate_row_counts: bool = False) -> list[TableSchema]:
        tables: list[TableSchema] = []

        for table_name in self.inspector.get_table_names():
            tables.append(self._extract_table(table_name, is_view=False, estimate_row_counts=estimate_row_counts))

        if include_views:
            try:
                for view_name in self.inspector.get_view_names():
                    tables.append(self._extract_table(view_name, is_view=True, estimate_row_counts=False))
            except NotImplementedError:
                # Some dialects (rare) don't support view introspection.
                pass

        return tables

    def _extract_table(self, table_name: str, is_view: bool, estimate_row_counts: bool) -> TableSchema:
        pk_constraint = self.inspector.get_pk_constraint(table_name)
        primary_keys = pk_constraint.get("constrained_columns") or []

        fks = self.inspector.get_foreign_keys(table_name)
        fk_columns: dict[str, str] = {}
        for fk in fks:
            for local_col, remote_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                fk_columns[local_col] = f"{fk['referred_table']}.{remote_col}"

        columns = []
        for col in self.inspector.get_columns(table_name):
            col_name = col["name"]
            columns.append(
                ColumnSchema(
                    name=col_name,
                    type=str(col["type"]),
                    nullable=col.get("nullable", True),
                    default=str(col["default"]) if col.get("default") is not None else None,
                    is_primary_key=col_name in primary_keys,
                    is_foreign_key=col_name in fk_columns,
                    references=fk_columns.get(col_name),
                )
            )

        indexes = [
            IndexSchema(
                name=idx["name"] or f"{table_name}_idx",
                columns=idx["column_names"] or [],
                unique=idx.get("unique", False),
            )
            for idx in self.inspector.get_indexes(table_name)
        ]

        row_count = None
        if estimate_row_counts and not is_view:
            row_count = self._estimate_row_count(table_name)

        return TableSchema(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=fks,
            indexes=indexes,
            is_view=is_view,
            row_count_estimate=row_count,
        )

    def _estimate_row_count(self, table_name: str) -> int | None:
        """Cheap COUNT(*) estimate. Skipped by default (estimate_row_counts=
        False) because on large production tables this can be slow; the
        optimizer node (Phase 5) will use dialect-specific catalog stats
        (e.g. information_schema.TABLES for MySQL) instead for real cost
        estimation.
        """
        from sqlalchemy import text

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
                return int(result.scalar())
        except Exception:  # noqa: BLE001 - best-effort only
            return None


def schema_to_json(tables: list[TableSchema]) -> str:
    return json.dumps([t.to_dict() for t in tables], indent=2, default=str)