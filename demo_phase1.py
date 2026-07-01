"""
demo_phase1.py

Runnable end-to-end demo of Phase 1: schema extraction + embeddings + FAISS
retrieval. By default it spins up a small in-memory SQLite e-commerce
schema so you can run this with zero setup. Pass --mysql to instead connect
to a real MySQL database using the MYSQL_* environment variables from
config/settings.py.

Usage:
    python demo_phase1.py                 # SQLite demo, no setup required
    MYSQL_HOST=... MYSQL_USER=... MYSQL_PASSWORD=... MYSQL_DATABASE=... \\
        python demo_phase1.py --mysql     # real MySQL database
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import create_engine, text

from database.connectors.mysql_connector import MySQLConnector
from database.connectors.sqlite_connector import SQLiteConnector
from embeddings.schema_indexer import SchemaIndexer


DEMO_DDL = """
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
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    department TEXT,
    hire_date TEXT
);
"""

DEMO_QUESTIONS = [
    "Show top 10 customers by revenue in the last 6 months",
    "How many employees work in engineering?",
    "List all products ordered by quantity",
    "Which customers are from California?",
]


def build_demo_sqlite() -> SQLiteConnector:
    connector = SQLiteConnector(path="file:phase1_demo?mode=memory&cache=shared")
    with connector.engine.begin() as conn:
        for stmt in DEMO_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    return connector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mysql", action="store_true", help="Connect to real MySQL via env vars")
    args = parser.parse_args()

    if args.mysql:
        connector = MySQLConnector()
        ok, msg = connector.test_connection()
        if not ok:
            print(f"❌ Could not connect to MySQL: {msg}")
            sys.exit(1)
        print(f"✅ Connected to MySQL at {connector.config.host}:{connector.config.port}/{connector.config.database}")
    else:
        connector = build_demo_sqlite()
        print("✅ Using in-memory SQLite demo schema (customers, orders, order_items, employees)")

    indexer = SchemaIndexer(connector)
    print("\n📖 Extracting schema...")
    tables = indexer.sync(force_rebuild=True)
    for t in tables:
        kind = "view" if t.is_view else "table"
        print(f"  - [{kind}] {t.name} ({len(t.columns)} columns, {len(t.foreign_keys)} FKs)")

    print("\n🔎 Building FAISS index and testing retrieval...\n")
    for question in DEMO_QUESTIONS:
        results = indexer.retrieve_relevant_tables(question, top_k=2)
        print(f"Q: {question}")
        for name, _text_block, score in results:
            print(f"   -> {name}  (score={score:.3f})")
        print()


if __name__ == "__main__":
    main()