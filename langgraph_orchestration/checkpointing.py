from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.memory import InMemorySaver


def thread_id_for_run(run_id: str) -> str:
    """Stable, short LangGraph thread id for one EDTech run."""
    value = str(run_id or "").strip()
    if not value:
        raise ValueError("run_id is required")
    thread_id = f"edtech:{value}"
    if len(thread_id) > 240:
        raise ValueError("run_id is too long for a safe LangGraph checkpoint thread id")
    return thread_id


def thread_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id_for_run(run_id)}}


def in_memory_checkpointer() -> InMemorySaver:
    """Test/dev checkpointer. It does NOT survive process restarts."""
    return InMemorySaver()


def resolve_checkpoint_db_url(explicit: str | None = None) -> str:
    """Resolve the Postgres URL used only for LangGraph checkpoints.

    Prefer LANGGRAPH_CHECKPOINT_DB_URL so checkpoint persistence can be
    configured independently. DATABASE_URL is accepted as a convenience if the
    EDTech project already exposes a plain PostgreSQL URL.
    """
    candidates = [
        explicit,
        os.getenv("LANGGRAPH_CHECKPOINT_DB_URL"),
        os.getenv("DATABASE_URL"),
    ]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        # AsyncPostgresSaver uses psycopg directly rather than SQLAlchemy.
        value = value.replace("postgresql+psycopg://", "postgresql://", 1)
        value = value.replace("postgres+psycopg://", "postgresql://", 1)
        if value.startswith("postgres://"):
            value = "postgresql://" + value[len("postgres://"):]
        if not value.startswith("postgresql://"):
            raise ValueError(
                "LangGraph checkpoint URL must be a PostgreSQL URL. "
                "Set LANGGRAPH_CHECKPOINT_DB_URL explicitly."
            )
        return value
    raise RuntimeError(
        "No LangGraph checkpoint database URL configured. Set "
        "LANGGRAPH_CHECKPOINT_DB_URL to your PostgreSQL connection URL."
    )


@asynccontextmanager
async def async_postgres_checkpointer(
    db_url: str | None = None,
    *,
    setup: bool = False,
) -> AsyncIterator[object]:
    """Open the production AsyncPostgresSaver.

    ``setup=True`` should only be used during the explicit one-time setup step;
    normal application runs should use ``setup=False``.
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph-checkpoint-postgres and psycopg[binary,pool] first."
        ) from exc

    url = resolve_checkpoint_db_url(db_url)
    async with AsyncPostgresSaver.from_conn_string(url) as checkpointer:
        if setup:
            await checkpointer.setup()
        yield checkpointer
