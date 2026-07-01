"""
config/settings.py

Central, environment-driven configuration for the SQL AI Assistant.

Everything that could differ between local dev / Docker / HF Spaces / Render
lives here, read from environment variables with sane free-tier defaults.
Nothing in this file requires a paid service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


@dataclass(frozen=True)
class MySQLSettings:
    """Connection settings for the *target* database the user wants to query.

    NOTE: This is deliberately separate from the app's own metadata DB
    (users, history, feedback) which is configured in AppDBSettings below.
    Keeping them separate means a bug in query execution can never touch
    user accounts / history, and vice versa.
    """

    host: str = field(default_factory=lambda: _env("MYSQL_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("MYSQL_PORT", 3306))
    user: str = field(default_factory=lambda: _env("MYSQL_USER", "root"))
    password: str = field(default_factory=lambda: _env("MYSQL_PASSWORD", ""))
    database: str = field(default_factory=lambda: _env("MYSQL_DATABASE", ""))
    # Driver: pymysql is pure-python and free, no system deps -> best for
    # free-tier hosts (HF Spaces / Render) where you can't always apt-get.
    driver: str = field(default_factory=lambda: _env("MYSQL_DRIVER", "pymysql"))
    # Read-only enforcement: the executor node will refuse to run anything
    # that isn't SELECT unless this is explicitly disabled.
    read_only: bool = field(default_factory=lambda: _env_bool("MYSQL_READ_ONLY", True))
    connect_timeout: int = field(default_factory=lambda: _env_int("MYSQL_CONNECT_TIMEOUT", 10))

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+{self.driver}://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    """Config for the schema-embedding model, called via the HuggingFace
    Inference API (see embeddings/embedder.py) -- never downloaded or run
    locally, same as the SQL-generation LLM in LLMSettings below.
    """

    # all-MiniLM-L6-v2: 384-dim, small/fast, good default for schema
    # retrieval where we don't need heavy semantic depth -- table/column
    # names are short, structured text.
    model_name: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    dimension: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))


@dataclass(frozen=True)
class FAISSSettings:
    """Config for the FAISS schema index (one index per connected DB)."""

    index_dir: Path = field(default_factory=lambda: BASE_DIR / "models" / "faiss_indexes")
    # cosine similarity via normalized inner product (IndexFlatIP) is the
    # standard choice for sentence-transformer embeddings.
    metric: str = "cosine"
    top_k: int = field(default_factory=lambda: _env_int("FAISS_TOP_K", 5))


@dataclass(frozen=True)
class LLMSettings:
    """Config for the free HuggingFace text-to-SQL generation model."""

    # Primary model: best free SQL/code performance in the 7B class.
    primary_model: str = field(
        default_factory=lambda: _env("LLM_PRIMARY_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    )
    # Fallback: small enough for CPU-only free tiers (HF Spaces free CPU,
    # Render free tier) when the primary model isn't available.
    fallback_model: str = field(
        default_factory=lambda: _env("LLM_FALLBACK_MODEL", "microsoft/Phi-3-mini-4k-instruct")
    )
    hf_api_token: str | None = field(default_factory=lambda: _env("HF_API_TOKEN"))
    max_new_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_NEW_TOKENS", 512))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))


@dataclass(frozen=True)
class AppDBSettings:
    """The application's own metadata DB (users, history, feedback).

    Defaults to SQLite so the whole project runs with zero external
    services out of the box; point APP_DB_URL at Postgres in production.
    """

    url: str = field(
        default_factory=lambda: _env("APP_DB_URL", f"sqlite:///{BASE_DIR / 'app.db'}")
    )


@dataclass(frozen=True)
class Settings:
    mysql: MySQLSettings = field(default_factory=MySQLSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    faiss: FAISSSettings = field(default_factory=FAISSSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    app_db: AppDBSettings = field(default_factory=AppDBSettings)
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")
    env: str = field(default_factory=lambda: _env("APP_ENV", "development"))


settings = Settings()