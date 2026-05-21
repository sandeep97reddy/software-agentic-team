"""
Retry Middleware — wraps any LangGraph node function with automatic retry
tracking, exponential back-off, and structured error logging.

Usage
─────
    from src.core.middleware import retry_middleware

    @retry_middleware(max_retries=3)
    def my_node(state: ProjectState) -> dict:
        ...

When the decorated node raises an exception:
 1. ``state["retry_counts"][node_name]`` is incremented.
 2. An ``ErrorRecord`` dict is appended to ``state["error_log"]``.
 3. If the retry ceiling has **not** been reached the node is re-invoked
    after an exponential back-off delay  (base_delay × 2^attempt).
 4. If the ceiling **is** reached the state is returned with
    ``status = "failed"`` so downstream routing can handle it.

This keeps retry logic *out* of every individual agent, following the
Single Responsibility Principle.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from src.core.state import ProjectState

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Default constants
# ──────────────────────────────────────────────────────────────
DEFAULT_MAX_RETRIES: int = 3
BASE_DELAY_SECONDS: float = 1.0  # first retry waits 1 s, then 2 s, 4 s …


def _build_error_record(
    node_name: str,
    exc: BaseException,
    attempt: int,
) -> dict[str, Any]:
    """Create a serialisable error record dict."""
    return {
        "node_name": node_name,
        "error_type": type(exc).__qualname__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "resolved": False,
    }


# ──────────────────────────────────────────────────────────────
#  Synchronous retry wrapper
# ──────────────────────────────────────────────────────────────


def retry_middleware(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = BASE_DELAY_SECONDS,
) -> Callable:
    """
    Decorator factory that wraps a **synchronous** LangGraph node function
    with retry logic and structured error tracking.

    Parameters
    ----------
    max_retries : int
        Maximum number of *retry* attempts (total calls = 1 + max_retries).
    base_delay : float
        Base delay in seconds for exponential back-off between retries.

    Returns
    -------
    Callable
        The decorated node function with identical signature.
    """

    def decorator(func: Callable[[ProjectState], dict[str, Any]]) -> Callable:
        node_name: str = func.__name__

        @functools.wraps(func)
        def wrapper(state: ProjectState) -> dict[str, Any]:
            # Initialise retry bookkeeping if absent
            retry_counts: dict[str, int] = dict(state.get("retry_counts", {}))
            current_count: int = retry_counts.get(node_name, 0)
            max_allowed: int = state.get("max_retries", max_retries)

            for attempt in range(1, max_allowed + 2):  # 1-indexed, includes initial try
                try:
                    logger.info(
                        ">> [%s] attempt %d / %d",
                        node_name,
                        attempt,
                        max_allowed + 1,
                    )
                    result: dict[str, Any] = func(state)

                    # Success — reset the failure counter for this node
                    retry_counts[node_name] = 0
                    result["retry_counts"] = retry_counts
                    logger.info("[OK] [%s] succeeded on attempt %d", node_name, attempt)
                    return result

                except Exception as exc:
                    current_count += 1
                    retry_counts[node_name] = current_count
                    error_record = _build_error_record(node_name, exc, attempt)

                    logger.warning(
                        "[FAIL] [%s] attempt %d failed: %s -- %s",
                        node_name,
                        attempt,
                        type(exc).__name__,
                        exc,
                    )

                    if attempt > max_allowed:
                        # Exhausted all retries — propagate failure via state
                        logger.error(
                            "[FAIL] [%s] exhausted %d retries -- marking FAILED",
                            node_name,
                            max_allowed,
                        )
                        return {
                            "retry_counts": retry_counts,
                            "error_log": [error_record],
                            "status": "failed",
                        }

                    # Back-off before next attempt
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info("[WAIT] [%s] retrying in %.1f s ...", node_name, delay)
                    time.sleep(delay)

            # Defensive — should never reach here
            return {
                "retry_counts": retry_counts,
                "status": "failed",
            }  # pragma: no cover

        return wrapper

    return decorator


# ──────────────────────────────────────────────────────────────
#  Async retry wrapper  (for async node functions)
# ──────────────────────────────────────────────────────────────


def async_retry_middleware(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = BASE_DELAY_SECONDS,
) -> Callable:
    """
    Same semantics as :func:`retry_middleware` but for ``async def`` node
    functions.
    """

    def decorator(func: Callable) -> Callable:
        node_name: str = func.__name__

        @functools.wraps(func)
        async def wrapper(state: ProjectState) -> dict[str, Any]:
            retry_counts: dict[str, int] = dict(state.get("retry_counts", {}))
            current_count: int = retry_counts.get(node_name, 0)
            max_allowed: int = state.get("max_retries", max_retries)

            for attempt in range(1, max_allowed + 2):
                try:
                    logger.info(
                        ">> [%s] async attempt %d / %d",
                        node_name,
                        attempt,
                        max_allowed + 1,
                    )
                    result: dict[str, Any] = await func(state)

                    retry_counts[node_name] = 0
                    result["retry_counts"] = retry_counts
                    logger.info("[OK] [%s] succeeded on attempt %d", node_name, attempt)
                    return result

                except Exception as exc:
                    current_count += 1
                    retry_counts[node_name] = current_count
                    error_record = _build_error_record(node_name, exc, attempt)

                    logger.warning(
                        "[FAIL] [%s] async attempt %d failed: %s -- %s",
                        node_name,
                        attempt,
                        type(exc).__name__,
                        exc,
                    )

                    if attempt > max_allowed:
                        logger.error(
                            "[FAIL] [%s] exhausted %d retries -- marking FAILED",
                            node_name,
                            max_allowed,
                        )
                        return {
                            "retry_counts": retry_counts,
                            "error_log": [error_record],
                            "status": "failed",
                        }

                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info("[WAIT] [%s] retrying in %.1f s ...", node_name, delay)
                    await asyncio.sleep(delay)

            return {
                "retry_counts": retry_counts,
                "status": "failed",
            }  # pragma: no cover

        return wrapper

    return decorator
