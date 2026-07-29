from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(RuntimeError):
    """Raised when PostgreSQL configuration is missing or invalid."""


def load_environment() -> None:
    """
    Load .env when python-dotenv is installed.

    Environment variables already defined by the operating system keep
    priority because load_dotenv() does not overwrite them by default.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def get_database_url() -> str:
    load_environment()

    url = os.getenv("DATABASE_URL", "").strip()

    if not url:
        raise DatabaseConfigurationError(
            "DATABASE_URL is not configured. Example: "
            "postgresql+psycopg://postgres:password@localhost:5432/edtech"
        )

    # Make common PostgreSQL URL forms explicitly use Psycopg 3.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = (
            "postgresql+psycopg://"
            + url[len("postgresql://"):]
        )

    if not url.startswith("postgresql+psycopg://"):
        raise DatabaseConfigurationError(
            "DATABASE_URL must use PostgreSQL with Psycopg 3."
        )

    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Create the application engine lazily.

    Lazy creation means importing project modules does not immediately
    require a live database connection.
    """

    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Open one transaction.

    The transaction commits on success and rolls back on failure.
    """

    session = get_session_factory()()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()