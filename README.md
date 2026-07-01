# SQL AI Assistant — Phase 1: Schema Extraction + Embeddings + FAISS

This is Phase 1 of a production-grade LLM-powered SQL query generator,
built entirely on free tools. Phase 1 covers the retrieval foundation the
rest of the system depends on: given a connected database, extract its
full schema, embed it via the HuggingFace Inference API, and retrieve the
tables relevant to any natural language question.

## What's implemented

- **`database/connectors/`** — `MySQLConnector` (primary target, via
  pymysql), `SQLiteConnector` (used for local dev/testing), and a shared
  `BaseConnector` interface so Postgres/MSSQL can be added later without
  touching anything downstream. `MySQLConnector` enforces read-only mode
  by default (`MYSQL_READ_ONLY=true`) — only `SELECT/SHOW/EXPLAIN/WITH`
  statements are allowed to execute.
- **`database/schema_extractor.py`** — dialect-agnostic schema extraction
  via `SQLAlchemy.Inspector`: tables, columns, types, nullability,
  primary keys, foreign keys (with resolved relationships), indexes,
  views, and optional row-count estimates. Produces both structured
  dataclasses (`TableSchema`) and a canonical text rendering used for
  embedding.
- **`embeddings/embedder.py`** — calls the HuggingFace Inference API's
  feature-extraction endpoint for `sentence-transformers/all-MiniLM-L6-v2`
  (384-dim). No model weights are downloaded or run locally — every
  embedding is an HTTP call, with retry/backoff for the free tier's
  cold-start behavior.
- **`embeddings/faiss_store.py`** — `FAISSSchemaStore`: builds/persists/
  loads a `FAISS IndexFlatIP` (cosine similarity) per connected database,
  keyed by a hash of the connection string so multiple databases don't
  collide. Embedder is injectable for testing.
- **`embeddings/schema_indexer.py`** — orchestrates connector → extractor →
  FAISS store. This is the single entry point later phases (the FastAPI
  backend, the LangGraph "retrieve relevant tables" node) will call.

## Why these choices

- **SQLAlchemy Inspector** instead of hand-rolled `information_schema`
  queries per dialect: one code path works for MySQL, Postgres, SQLite,
  and MSSQL, which matters a lot once Phase 2+ needs multi-DB support.
- **Embeddings via HF Inference API, not local**: keeps the environment
  free of ML runtime dependencies (`torch`, `transformers`) entirely —
  everything HuggingFace-related, embeddings and the Phase 2 LLM alike,
  goes through one API surface with one token.
- **IndexFlatIP over IVF/HNSW**: schemas are small (tens–low hundreds of
  tables), so exact search is fast enough — no need for approximate
  index complexity at this scale.
- **One chunk per table** (not per column): keeps retrieval granularity
  aligned with what the LLM needs in its prompt — a full table block with
  its FKs, not fragmented column-level snippets that lose relational
  context.
- **Tests use a `FakeEmbedder` and mocked HTTP** instead of real HF API
  calls. This keeps the test suite fast, free, and network-independent —
  important for CI (GitHub Actions shouldn't need a live token/network on
  every push) and for offline development.

## Setup

```bash
# 1. Clone/copy the project, then create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies (no torch/transformers — API-only design)
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# then edit .env:
#   - HF_API_TOKEN=<your free token from https://huggingface.co/settings/tokens>
#     required for embeddings (and for the Phase 2 LLM later)
#   - MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE
#     only needed if you want to run the demo against a real MySQL DB
```

## Running it

```bash
# Run the test suite (fast, no network or token required — uses fakes/mocks)
pytest -v

# Demo against an in-memory SQLite schema (zero DB setup, but DOES call
# the real HF Inference API for embeddings, so HF_API_TOKEN must be set)
python demo_phase1.py

# Demo against a real MySQL database
python demo_phase1.py --mysql
```

`demo_phase1.py` will:
1. Build (or connect to) a sample schema
2. Extract it via `SchemaExtractor`
3. Embed every table via the HF Inference API and build a FAISS index
4. Run a few natural-language questions through `retrieve_relevant_tables()`
   and print which tables were retrieved, with similarity scores

The **first** call may be slow (a few seconds) if the model is cold on
HF's free tier — the embedder automatically retries with `wait_for_model`.

## Test coverage

```
tests/test_schema_extractor.py   6 tests — PK/FK detection, indexes, views, row counts, text rendering
tests/test_faiss_retrieval.py    6 tests — index build, retrieval relevance, score ordering, persistence
tests/test_embedder.py           6 tests — API call shape, normalization, batching, retries, error handling
```

All 18 pass with `pytest -v`, with zero network calls (HTTP is mocked).

## What's NOT in Phase 1 (coming next)

- LLM-based SQL generation (Phase 2 — LangGraph agent)
- SQL validation / error recovery / optimization
- FastAPI backend + Streamlit frontend
- Auth, conversation memory, feedback storage
- Evaluation harness (Spider/BIRD), Docker, CI/CD

## A note on the MySQL connection

`demo_phase1.py --mysql` and `MySQLConnector` are fully implemented and
tested against the *extraction logic* (via the dialect-agnostic SQLite
path). I don't have a live MySQL server in this environment to test
against, so when you point it at your real database, if you hit any
driver-specific quirk (e.g. an unusual column type SQLAlchemy's MySQL
dialect renders differently than expected), send the error over and I'll
patch it immediately.