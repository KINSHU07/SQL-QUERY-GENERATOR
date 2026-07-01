"""
tests/test_faiss_retrieval.py

End-to-end test of Phase 1: connect -> extract schema -> embed -> build
FAISS index -> retrieve relevant tables for a natural-language question.

Uses a temp directory for the index so tests don't pollute (or depend on)
models/faiss_indexes/ on disk, and don't interfere with each other.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from database.schema_extractor import SchemaExtractor
from embeddings.faiss_store import FAISSSchemaStore, make_db_id
from tests.fakes import FakeEmbedder

DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    state TEXT,
    signup_date TEXT
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    total_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    price REAL
);

CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    department TEXT,
    hire_date TEXT
);
"""


@pytest.fixture(scope="module")
def tmp_index_dir():
    d = tempfile.mkdtemp(prefix="faiss_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def built_store(tmp_index_dir):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        for statement in DDL.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))

    tables = SchemaExtractor(engine).extract()
    db_id = make_db_id("sqlite:///:memory:test-faiss")
    store = FAISSSchemaStore(db_id=db_id, index_dir=tmp_index_dir, embedder=FakeEmbedder())
    store.build_from_schema(tables)
    return store


def test_index_contains_all_tables(built_store):
    assert set(built_store.all_tables()) == {"customers", "orders", "products", "employees"}


def test_retrieval_finds_relevant_table_for_customer_question(built_store):
    results = built_store.query("Which customers signed up most recently?", top_k=2)
    top_names = [chunk.table_name for chunk, score in results]
    assert "customers" in top_names


def test_retrieval_finds_relevant_table_for_order_revenue_question(built_store):
    results = built_store.query("Show total revenue per order last month", top_k=2)
    top_names = [chunk.table_name for chunk, score in results]
    assert "orders" in top_names


def test_retrieval_finds_relevant_table_for_employee_question(built_store):
    results = built_store.query("List employees hired in the engineering department", top_k=2)
    top_names = [chunk.table_name for chunk, score in results]
    assert "employees" in top_names


def test_scores_are_sorted_descending(built_store):
    results = built_store.query("customer orders", top_k=4)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_persistence_round_trip(tmp_index_dir):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        for statement in DDL.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    tables = SchemaExtractor(engine).extract()

    db_id = make_db_id("sqlite:///:memory:test-persistence")
    store = FAISSSchemaStore(db_id=db_id, index_dir=tmp_index_dir, embedder=FakeEmbedder())
    store.build_from_schema(tables)

    # Fresh store instance pointing at the same db_id/dir must load from disk.
    reloaded = FAISSSchemaStore(db_id=db_id, index_dir=tmp_index_dir, embedder=FakeEmbedder())
    assert reloaded.exists()
    results = reloaded.query("employee department", top_k=1)
    assert results[0][0].table_name == "employees"