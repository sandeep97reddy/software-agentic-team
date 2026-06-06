"""
observability.py — LangSmith tracing configuration.

Provides a single ``setup_langsmith()`` function that wires environment
variables so that every LangChain / LangGraph call in the process is
automatically traced and sent to LangSmith.

Environment variables (set these in .env or your shell):
────────────────────────────────────────────────────────
  LANGSMITH_API_KEY        Required – your LangSmith API key.
  LANGSMITH_PROJECT        Optional – project name in LangSmith UI.
                           Defaults to "ai-software-team".
  LANGSMITH_ENDPOINT       Optional – custom LangSmith endpoint.
                           Defaults to "https://api.smith.langchain.com".
  LANGCHAIN_TRACING_V2     Optional – set to "false" to disable tracing.
                           Defaults to "true" when LANGSMITH_API_KEY is set.

Usage (import at the very top of app.py, before any LangChain imports):
────────────────────────────────────────────────────────────────────────
  from src.core.observability import setup_langsmith
  setup_langsmith()
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def setup_langsmith() -> bool:
    """
    Configure LangSmith tracing for the entire process.

    Reads credentials from environment variables and sets the LangChain
    tracing env vars so that all subsequent LLM and LangGraph calls are
    captured automatically.

    Returns
    -------
    bool
        ``True`` if tracing was enabled, ``False`` if skipped (no API key).
    """
    api_key: str = os.getenv("LANGSMITH_API_KEY", "")

    # Allow explicit opt-out even when a key is present
    tracing_enabled: str = os.getenv("LANGCHAIN_TRACING_V2", "true" if api_key else "false")

    if not api_key or tracing_enabled.lower() == "false":
        logger.info(
            "[OBSERVABILITY] LangSmith tracing DISABLED "
            "(set LANGSMITH_API_KEY + LANGCHAIN_TRACING_V2=true to enable)"
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    project: str = os.getenv("LANGSMITH_PROJECT", "ai-software-team")
    endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # These env vars are read by langchain-core automatically
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    logger.info(
        "[OBSERVABILITY] LangSmith tracing ENABLED — project=%s  endpoint=%s",
        project,
        endpoint,
    )
    return True


def get_langsmith_metadata(project_id: str, node_name: str, **extra: Any) -> dict[str, Any]:
    """
    Build a standard metadata dict to attach to every LangSmith run.

    Pass the returned dict as ``metadata`` to any ``langchain`` or
    ``langgraph`` ``.invoke()`` / ``.stream()`` call so that runs are
    grouped and filterable in the LangSmith UI.

    Parameters
    ----------
    project_id : str
        UUID of the current pipeline run.
    node_name : str
        Name of the LangGraph node emitting this call.
    **extra
        Any additional key-value pairs to surface as metadata tags.

    Returns
    -------
    dict[str, Any]
        Ready-to-use metadata dict.
    """
    return {
        "project_id": project_id,
        "node_name": node_name,
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "ai-software-team"),
        **extra,
    }


def get_run_config(project_id: str, node_name: str, **extra: Any) -> dict[str, Any]:
    """
    Return a ``config`` dict suitable for passing to LangChain / LangGraph
    ``.invoke()`` / ``.stream()`` so that the run is tagged correctly in the
    LangSmith UI.

    Usage
    -----
    >>> config = get_run_config(project_id, "backend_engineer")
    >>> llm.invoke(messages, config=config)
    """
    return {
        "metadata": get_langsmith_metadata(project_id, node_name, **extra),
        "tags": [f"node:{node_name}", f"project:{project_id[:8]}"],
        "run_name": f"{node_name} [{project_id[:8]}]",
    }
