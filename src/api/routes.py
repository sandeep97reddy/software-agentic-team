"""
API route handlers for the AI Software Engineering Team orchestration layer.

Endpoints
─────────
POST   /projects/run         — kick off a full pipeline run
GET    /projects/{id}/status  — poll for current state of a run
GET    /health                — liveness / readiness probe
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.graph import build_graph
from src.core.observability import get_run_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["projects"])

# ──────────────────────────────────────────────────────────────
#  In-memory run store  (fallback cache)
# ──────────────────────────────────────────────────────────────
_runs: dict[str, dict[str, Any]] = {}

# ──────────────────────────────────────────────────────────────
#  Compile the graph once at module-load time with default checkpointer
# ──────────────────────────────────────────────────────────────
_compiled_graph = build_graph()


# ──────────────────────────────────────────────────────────────
#  Request / Response schemas
# ──────────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    """Body for ``POST /projects/run``."""

    requirements: str = Field(
        ...,
        min_length=10,
        description="Natural-language requirements for the software project",
    )
    project_name: str = Field(
        default="",
        description="Optional human-friendly project name",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Per-node retry ceiling",
    )


class RunResponse(BaseModel):
    """Returned by ``POST /projects/run``."""

    project_id: str
    status: str
    current_phase: str
    iteration: int
    technical_specifications: dict[str, Any] = {}
    architecture_decisions: list[dict[str, Any]] = []
    project_structure: dict[str, Any] = {}
    task_queue: list[dict[str, Any]] = []
    completed_tasks: list[dict[str, Any]] = []
    code_artifacts: list[dict[str, Any]] = []
    workspace_dir: str = ""
    active_branch: str = "main"
    execution_trace: list[dict[str, Any]] = []
    error_log: list[dict[str, Any]] = []
    retry_counts: dict[str, int] = {}
    task_failures: dict[str, int] = {}


class StatusResponse(BaseModel):
    """Returned by ``GET /projects/{project_id}/status``."""

    project_id: str
    status: str
    current_phase: str
    summary: dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    graph_nodes: list[str]


# ──────────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────────


@router.post("/projects/run", response_model=RunResponse, status_code=200)
def run_project(body: RunRequest) -> RunResponse:
    """
    Execute the full LangGraph pipeline synchronously and return the
    final state.
    """
    project_id = str(uuid.uuid4())
    logger.info("[RUN] Received run request -- project_id=%s", project_id)

    import os
    import tempfile

    workspace_dir = os.path.join(
        tempfile.gettempdir(), "ai_team_workspaces", project_id
    )
    initial_state: dict[str, Any] = {
        "project_id": project_id,
        "project_name": body.project_name or f"project-{project_id[:8]}",
        "requirements": body.requirements,
        "technical_specifications": {},
        "architecture_decisions": [],
        "project_structure": {},
        "task_queue": [],
        "completed_tasks": [],
        "code_artifacts": [],
        "workspace_dir": workspace_dir,
        "active_branch": "main",
        "execution_trace": [],
        "retry_counts": {},
        "task_failures": {},
        "error_log": [],
        "current_phase": "initializing",
        "iteration": 0,
        "max_retries": body.max_retries,
        "status": "initialized",
    }

    try:
        run_config = get_run_config(
            project_id=project_id,
            node_name="pipeline",
            requirements_length=len(body.requirements),
        )
        final_state = _compiled_graph.invoke(initial_state, config=run_config)
    except Exception as exc:
        logger.exception("Pipeline invocation failed for %s", project_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Persist the result for in-memory fallback
    _runs[project_id] = final_state

    return RunResponse(
        project_id=final_state.get("project_id", project_id),
        status=final_state.get("status", "unknown"),
        current_phase=final_state.get("current_phase", "unknown"),
        iteration=final_state.get("iteration", 0),
        technical_specifications=final_state.get("technical_specifications", {}),
        architecture_decisions=final_state.get("architecture_decisions", []),
        project_structure=final_state.get("project_structure", {}),
        task_queue=final_state.get("task_queue", []),
        completed_tasks=final_state.get("completed_tasks", []),
        code_artifacts=final_state.get("code_artifacts", []),
        workspace_dir=final_state.get("workspace_dir", ""),
        active_branch=final_state.get("active_branch", "main"),
        execution_trace=final_state.get("execution_trace", []),
        error_log=final_state.get("error_log", []),
        retry_counts=final_state.get("retry_counts", {}),
        task_failures=final_state.get("task_failures", {}),
    )


@router.get("/projects/{project_id}/status", response_model=StatusResponse)
def get_project_status(project_id: str) -> StatusResponse:
    """Return a summary of a previously-executed run from checkpointer or memory store."""
    try:
        snapshot = _compiled_graph.get_state({"configurable": {"thread_id": project_id}})
        if snapshot and snapshot.values:
            run = snapshot.values
            return StatusResponse(
                project_id=project_id,
                status=run.get("status", "unknown"),
                current_phase=run.get("current_phase", "unknown"),
                summary={
                    "architecture_decisions_count": len(run.get("architecture_decisions", [])),
                    "completed_tasks_count": len(run.get("completed_tasks", [])),
                    "code_artifacts_count": len(run.get("code_artifacts", [])),
                    "errors_count": len(run.get("error_log", [])),
                    "retry_counts": run.get("retry_counts", {}),
                    "task_failures": run.get("task_failures", {}),
                },
            )
    except Exception as exc:
        logger.debug("Checkpointer snapshot query failed for %s: %s", project_id, exc)

    run = _runs.get(project_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {project_id} not found")

    return StatusResponse(
        project_id=project_id,
        status=run.get("status", "unknown"),
        current_phase=run.get("current_phase", "unknown"),
        summary={
            "architecture_decisions_count": len(run.get("architecture_decisions", [])),
            "completed_tasks_count": len(run.get("completed_tasks", [])),
            "code_artifacts_count": len(run.get("code_artifacts", [])),
            "errors_count": len(run.get("error_log", [])),
            "retry_counts": run.get("retry_counts", {}),
            "task_failures": run.get("task_failures", {}),
        },
    )


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness probe — confirms the server and graph are operational."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        graph_nodes=list(_compiled_graph.get_graph().nodes.keys()),
    )


# ──────────────────────────────────────────────────────────────
#  Chunk 6 Requirements Aliases
# ──────────────────────────────────────────────────────────────


@router.post("/execute", response_model=RunResponse, status_code=200)
def execute_pipeline(body: RunRequest) -> RunResponse:
    """Trigger the graph directly."""
    return run_project(body)


@router.get("/status", response_model=StatusResponse)
def stream_status(project_id: str) -> StatusResponse:
    """Stream or poll the current state of a given project run."""
    return get_project_status(project_id)
