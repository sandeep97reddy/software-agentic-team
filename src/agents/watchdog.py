"""
Watchdog Node
=============

A special LangGraph node that monitors retry_counts.
If a task fails 3 times, it redirects the graph to a human_approval node.
"""

import logging
from typing import Any

from src.core.state import ProjectState

logger = logging.getLogger(__name__)


def watchdog_node(state: ProjectState) -> dict[str, Any]:
    """
    Watchdog node: monitors for infinite loops.
    Provides a waypoint for routing and logging.
    """
    logger.info("[WATCHDOG] Checking task retry counts to prevent infinite loops...")
    return {}


def human_approval_node(state: ProjectState) -> dict[str, Any]:
    """
    Fallback node when tasks fail multiple times.
    """
    logger.error(
        "[HUMAN] Pipeline paused. Human approval required to resolve recurring failures."
    )
    return {"status": "blocked"}
