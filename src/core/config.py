"""
Centralised configuration for the AI Software Engineering Team.

All LLM and runtime settings live here so that agent modules never
hard-code model names, temperatures, or API keys.

Environment variables
---------------------
OPENAI_API_KEY          Required.  The OpenAI (or compatible) API key.
MODEL_NAME              Optional.  Defaults to ``gpt-4o``.
TEMPERATURE             Optional.  Defaults to ``0.2`` (deterministic-leaning).
MAX_TOKENS              Optional.  Defaults to ``4096``.

LangSmith (observability)
--------------------------
LANGSMITH_API_KEY       Required to enable tracing.
LANGSMITH_PROJECT       Optional.  Defaults to ``ai-software-team``.
LANGSMITH_ENDPOINT      Optional.  Defaults to https://api.smith.langchain.com.
LANGCHAIN_TRACING_V2    Optional.  Defaults to ``true`` when key is present.
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  LLM defaults (overridable via env vars)
# ──────────────────────────────────────────────────────────────
DEFAULT_MODEL: str = os.getenv("MODEL_NAME", "gpt-4o")
DEFAULT_TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))


def get_llm(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ChatOpenAI:
    """
    Return a configured ``ChatOpenAI`` instance.

    LangSmith tracing is applied automatically when the
    ``LANGSMITH_API_KEY`` environment variable is set.  Every call made
    through the returned LLM will appear as a child span under the active
    LangSmith run.

    Parameters
    ----------
    model : str
        Model identifier (e.g. ``gpt-4o``, ``gpt-4o-mini``).
    temperature : float
        Sampling temperature.  Lower = more deterministic.
    max_tokens : int
        Maximum tokens in the completion.

    Returns
    -------
    ChatOpenAI
        Ready-to-use LangChain chat model with tracing wired in.

    Raises
    ------
    RuntimeError
        If ``OPENAI_API_KEY`` is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before running the pipeline."
        )

    logger.info(
        "[CONFIG] LLM: model=%s  temp=%.2f  max_tokens=%d",
        model,
        temperature,
        max_tokens,
    )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        # LangSmith tracing is injected globally via env vars set in
        # observability.setup_langsmith(); no extra args needed here.
    )
