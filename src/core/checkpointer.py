"""
checkpointer.py -- Centralized Checkpointer Factory for LangGraph Persistence.

Supports:
  - 'memory': In-memory ephemeral checkpointer (MemorySaver).
  - 'postgres': Persistent Postgres checkpointer (PostgresSaver) with table setup.
  - 'redis': High-throughput Redis checkpointer (RedisSaver).

Provides graceful fallback to MemorySaver if optional database drivers
(psycopg, redis) are not installed or connections fail.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

__all__ = ["get_checkpointer"]


def get_checkpointer(
    backend: str | None = None,
    conn_string: str | None = None,
    *,
    fallback_to_memory: bool = True,
    **kwargs: Any,
) -> BaseCheckpointSaver:
    """
    Instantiate and return a configured LangGraph checkpointer.

    Parameters
    ----------
    backend : str | None
        One of 'memory', 'postgres', or 'redis'. Defaults to env var
        `CHECKPOINTER_BACKEND` or 'memory'.
    conn_string : str | None
        Optional explicit connection string. If omitted, checks `POSTGRES_URL` /
        `DATABASE_URL` for postgres, or `REDIS_URL` for redis.
    fallback_to_memory : bool
        If True (default), falls back to MemorySaver on import or connection error.
        If False, re-raises the underlying exception.
    **kwargs : Any
        Additional keyword arguments passed to checkpointer constructors.

    Returns
    -------
    BaseCheckpointSaver
        Configured checkpoint saver instance.
    """
    chosen_backend = (
        backend or os.getenv("CHECKPOINTER_BACKEND", "memory")
    ).lower().strip()

    if chosen_backend == "postgres":
        postgres_url = (
            conn_string
            or os.getenv("POSTGRES_URL")
            or os.getenv("DATABASE_URL")
            or "postgresql://ai_team:supersecretpassword@localhost:5432/agent_state"
        )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore

            saver = PostgresSaver.from_conn_string(postgres_url, **kwargs)
            saver.setup()
            logger.info("[CHECKPOINTER] Successfully connected to Postgres checkpointer")
            return saver
        except ImportError as err:
            logger.warning(
                "[CHECKPOINTER] 'langgraph-checkpoint-postgres' or 'psycopg' not installed (%s).",
                err,
            )
            if not fallback_to_memory:
                raise
        except Exception as exc:
            logger.warning(
                "[CHECKPOINTER] Failed to connect to Postgres (%s).",
                exc,
            )
            if not fallback_to_memory:
                raise

        logger.warning("[CHECKPOINTER] Falling back to in-memory MemorySaver.")
        return MemorySaver()

    elif chosen_backend == "redis":
        redis_url = (
            conn_string
            or os.getenv("REDIS_URL")
            or "redis://localhost:6379/0"
        )
        try:
            from langgraph.checkpoint.redis import RedisSaver  # type: ignore

            saver = RedisSaver.from_conn_info(redis_url, **kwargs)
            logger.info("[CHECKPOINTER] Successfully connected to Redis checkpointer")
            return saver
        except ImportError as err:
            logger.warning(
                "[CHECKPOINTER] 'langgraph-checkpoint-redis' or 'redis' not installed (%s).",
                err,
            )
            if not fallback_to_memory:
                raise
        except Exception as exc:
            logger.warning(
                "[CHECKPOINTER] Failed to connect to Redis (%s).",
                exc,
            )
            if not fallback_to_memory:
                raise

        logger.warning("[CHECKPOINTER] Falling back to in-memory MemorySaver.")
        return MemorySaver()

    elif chosen_backend == "memory":
        logger.info("[CHECKPOINTER] Using in-memory MemorySaver.")
        return MemorySaver()

    else:
        logger.warning(
            "[CHECKPOINTER] Unknown backend '%s'. Defaulting to MemorySaver.",
            chosen_backend,
        )
        return MemorySaver()
