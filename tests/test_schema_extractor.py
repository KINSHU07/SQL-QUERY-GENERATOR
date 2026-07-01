"""
tests/test_schema_extractor.py

Validates SchemaExtractor against a realistic mini e-commerce schema
(customers, orders, order_items) built in-memory with SQLite. The
extractor code path is identical for MySQL (SQLAlchemy Inspector is
dialect-agnostic), so this proves correctness without needing a live
MySQL server in CI.
"""

from sqlalchemy import create_engine, text

from database.schema_extractor import SchemaExtractor


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

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE VIEW customer_totals AS
    SELECT c.customer_id, c.name, SUM(o.total_amount) AS lifetime_value
    FROM customers c JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id;
"""


def make_test_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        for statement in DDL.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    return engine


def test_extracts_all_tables_and_views():
    engine = make_test_engine()
    extractor = SchemaExtractor(engine)
    tables = extractor.extract(include_views=True)

    names = {t.name for t in tables}
    assert {"customers", "orders", "order_items", "customer_totals"} <= names

    view = next(t for t in tables if t.name == "customer_totals")
    assert view.is_view is True

    real_table = next(t for t in tables if t.name == "customers")
    assert real_table.is_view is False


def test_primary_keys_detected():
    engine = make_test_engine()
    tables = SchemaExtractor(engine).extract()
    orders = next(t for t in tables if t.name == "orders")
    assert orders.primary_keys == ["order_id"]


def test_foreign_keys_detected_and_linked_to_columns():
    engine = make_test_engine()
    tables = SchemaExtractor(engine).extract()
    orders = next(t for t in tables if t.name == "orders")

    assert len(orders.foreign_keys) == 1
    fk = orders.foreign_keys[0]
    assert fk["referred_table"] == "customers"

    customer_id_col = next(c for c in orders.columns if c.name == "customer_id")
    assert customer_id_col.is_foreign_key is True
    assert customer_id_col.references == "customers.customer_id"


def test_indexes_detected():
    engine = make_test_engine()
    tables = SchemaExtractor(engine).extract()
    orders = next(t for t in tables if t.name == "orders")
    index_names = [idx.name for idx in orders.indexes]
    assert "idx_orders_customer" in index_names


def test_to_text_renders_relationships_and_columns():
    engine = make_test_engine()
    tables = SchemaExtractor(engine).extract()
    orders = next(t for t in tables if t.name == "orders")
    text_block = orders.to_text()

    assert "Table: orders" in text_block
    assert "customer_id" in text_block
    assert "orders(customer_id) -> customers(customer_id)" in text_block


def test_row_count_estimation_optional():
    engine = make_test_engine()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO customers (customer_id, name, state) VALUES (1, 'Alice', 'CA')"))
        conn.execute(text("INSERT INTO customers (customer_id, name, state) VALUES (2, 'Bob', 'NY')"))

    tables = SchemaExtractor(engine).extract(estimate_row_counts=True)
    customers = next(t for t in tables if t.name == "customers")
    assert customers.row_count_estimate == 2