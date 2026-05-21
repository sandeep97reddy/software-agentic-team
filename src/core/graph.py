"""
graph.py -- LangGraph pipeline definition.

This module wires together the StateGraph using ``ProjectState`` as its
schema.  The pipeline is structured in two phases:

Phase 1 -- Planning (Chunk 2):
    INIT --> REQ_ANALYZER --> ARCHITECT --> TASK_PLANNER

Phase 2 -- Execution (stubs for future chunks):
    TASK_PLANNER --> DEVELOPER --> TESTER --> REVIEWER --> END

Pipeline diagram:

    +------+    +----------+    +-----------+    +----------+
    | INIT |--->| REQ      |--->| ARCHITECT |--->| TASK     |
    +------+    | ANALYZER |    +-----------+    | PLANNER  |
                +----------+                     +----+-----+
                                                      |
                +----------+    +--------+    +----------+
                | REVIEWER |<---| TESTER |<---| DEVELOPER|
                +----+-----+    +--------+    +----------+
                     |
                    END

Every node is wrapped with ``retry_middleware`` so failures are caught,
logged, and retried transparently.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.architect import architect_node
from src.agents.backend_engineer import backend_engineer_node
from src.agents.frontend_engineer import frontend_engineer_node
from src.agents.memory import memory_compression_node
# Agents
from src.agents.requirement_analyzer import requirement_analyzer_node
from src.agents.reviewer import reviewer_node
from src.agents.task_planner import task_planner_node
from src.agents.tester import tester_node
from src.agents.watchdog import human_approval_node, watchdog_node
from src.core.middleware import retry_middleware
from src.core.state import ProjectState
from src.tools.filesystem import FileSystemManager
from src.tools.git_tracker import GitTracker

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Initializer node
# ──────────────────────────────────────────────────────────────


@retry_middleware(max_retries=1)
def initialize_project(state: ProjectState) -> dict[str, Any]:
    """
    Bootstrap node -- creates the workspace sandbox, initialises the git
    repository, and checks out the active branch.

    Responsibilities:
        - Ensure ``project_id`` exists.
        - Set status to ``running`` and phase to ``planning``.
        - Create the workspace directory via FileSystemManager.
        - Initialise a git repo and checkout ``active_branch`` via GitTracker.
        - Seed ``execution_trace`` with the tool records from setup.
    """
    project_id = state.get("project_id") or str(uuid.uuid4())
    workspace_dir = state.get("workspace_dir", "")
    active_branch = state.get("active_branch", "main")

    if not workspace_dir:
        import os
        import tempfile

        workspace_dir = os.path.join(
            tempfile.gettempdir(), "ai_team_workspaces", project_id
        )

    logger.info(
        "[INIT] Initialising project %s -- workspace: %s", project_id, workspace_dir
    )

    trace: list[dict[str, Any]] = []

    # -- Set up filesystem sandbox --
    fs = FileSystemManager(workspace_dir=workspace_dir, trace=trace)

    # Write a .gitkeep so the root is non-empty before git init
    project_name = state.get("project_name", f"project-{project_id[:8]}")
    fs.write_file(
        ".gitkeep",
        f"# {project_name} -- AI Software Engineering Team\n",
    )

    # -- Initialise git repo and checkout branch --
    git = GitTracker(workspace_dir=workspace_dir, trace=trace)
    git.init(default_branch="main")
    git.stage_all()
    git.commit(message="chore: initial workspace scaffold", allow_empty=False)
    git.ensure_branch(active_branch)

    logger.info(
        "[INIT] Sandbox ready -- branch=%s  trace_events=%d", active_branch, len(trace)
    )

    return {
        "project_id": project_id,
        "project_name": project_name,
        "requirements": state.get("requirements", ""),
        "workspace_dir": workspace_dir,
        "active_branch": active_branch,
        "current_phase": "planning",
        "iteration": state.get("iteration", 0) + 1,
        "max_retries": state.get("max_retries", 3),
        "status": "running",
        "retry_counts": state.get("retry_counts", {}),
        "execution_trace": trace,
    }


# ──────────────────────────────────────────────────────────────
#  Conditional edges  (routing logic)
# ──────────────────────────────────────────────────────────────


def route_to_workers(state: ProjectState) -> str:
    """Route after planner or workers -- check if tasks remain."""
    if state.get("status") == "failed":
        logger.warning("[HALT] Pipeline halted -- status is FAILED")
        return END

    task_queue = state.get("task_queue", [])
    if not task_queue:
        logger.info("[ROUTE] No tasks remaining -- moving to memory compression")
        return "memory_compression"

    next_task = task_queue[0]
    fp = next_task.get("file_path", "").lower()

    frontend_exts = [".tsx", ".jsx", ".ts", ".js", ".css", ".html"]
    if (
        any(fp.endswith(ext) for ext in frontend_exts)
        or "frontend" in fp
        or "components" in fp
    ):
        return "frontend_engineer"
    else:
        return "backend_engineer"


def route_after_tester(state: ProjectState) -> str:
    """Pass/Fail from QA node."""
    if state.get("status") == "failed":
        return END
    task_queue = state.get("task_queue", [])
    if task_queue:
        logger.info("[ROUTE] QA found failures. Sending to watchdog.")
        return "watchdog"
    else:
        logger.info("[ROUTE] QA passed. Sending to reviewer.")
        return "reviewer"


def route_after_reviewer(state: ProjectState) -> str:
    """Pass/Fail from Reviewer node."""
    if state.get("status") == "failed":
        return END
    task_queue = state.get("task_queue", [])
    if task_queue:
        logger.info("[ROUTE] Reviewer found issues. Sending to watchdog.")
        return "watchdog"
    return END


def route_after_watchdog(state: ProjectState) -> str:
    """Check if task retry counts exceed 3."""
    retry_counts = state.get("retry_counts", {})
    # Check if any task failed >= 3 times
    for k, v in retry_counts.items():
        if k.startswith("task_fail_") and v >= 3:
            logger.warning(
                f"[ROUTE] Watchdog caught infinite loop on {k}. Routing to human_approval."
            )
            return "human_approval"

    logger.info("[ROUTE] Watchdog cleared. Routing back to workers.")
    return route_to_workers(state)


# ──────────────────────────────────────────────────────────────
#  Graph builder
# ──────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Construct and compile the full LangGraph ``StateGraph``.

    Returns
    -------
    langgraph.graph.StateGraph
        A compiled graph ready to be invoked with an initial state dict.
    """
    graph = StateGraph(ProjectState)

    # ── Register nodes ────────────────────────────────────────
    # Phase 1: Planning
    graph.add_node("initializer", initialize_project)
    graph.add_node("requirement_analyzer", requirement_analyzer_node)
    graph.add_node("architect", architect_node)
    graph.add_node("task_planner", task_planner_node)

    # Phase 2: Execution (workers)
    graph.add_node("backend_engineer", backend_engineer_node)
    graph.add_node("frontend_engineer", frontend_engineer_node)

    # Phase 3: QA & Review (Chunk 5)
    graph.add_node("memory_compression", memory_compression_node)
    graph.add_node("tester", tester_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("watchdog", watchdog_node)
    graph.add_node("human_approval", human_approval_node)

    # ── Wire edges ────────────────────────────────────────────
    # Planning chain (linear)
    graph.set_entry_point("initializer")
    graph.add_edge("initializer", "requirement_analyzer")
    graph.add_edge("requirement_analyzer", "architect")
    graph.add_edge("architect", "task_planner")

    # Validation Logic: loop back to workers if task_queue is not empty
    graph.add_conditional_edges("task_planner", route_to_workers)
    graph.add_conditional_edges("backend_engineer", route_to_workers)
    graph.add_conditional_edges("frontend_engineer", route_to_workers)

    # QA & Review conditional logic
    graph.add_edge("memory_compression", "tester")
    graph.add_conditional_edges("tester", route_after_tester)
    graph.add_conditional_edges("reviewer", route_after_reviewer)
    graph.add_conditional_edges("watchdog", route_after_watchdog)
    graph.add_edge("human_approval", END)

    compiled = graph.compile()
    logger.info("[OK] LangGraph pipeline compiled successfully")
    return compiled
