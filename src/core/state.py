"""
ProjectState — the single Source of Truth for the entire LangGraph pipeline.

Every node in the graph reads from and writes to this TypedDict. Custom
reducers ensure deterministic state transitions, idempotence, deduplication,
and safe dictionary merges without unexpected key clobbering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from src.core.reducers import (
    CLEAR,
    ClearSignal,
    adr_reducer,
    artifact_reducer,
    clearable_list_reducer,
    dict_merge_reducer,
    task_queue_reducer,
)

__all__ = [
    "ClearSignal",
    "CLEAR",
    "clearable_list_reducer",
    "artifact_reducer",
    "task_queue_reducer",
    "dict_merge_reducer",
    "adr_reducer",
    "TaskItem",
    "CodeArtifact",
    "ArchitectureDecision",
    "ADR",
    "ErrorRecord",
    "ExecutionTraceItem",
    "ProjectState",
]


# ─────────────────────────────────────────────────────────────
#  Sub-models (Pydantic) — structured data at boundaries
# ─────────────────────────────────────────────────────────────


class TaskItem(BaseModel):
    """A discrete unit of work to be completed by an agent."""

    task_id: str = Field(..., description="Unique identifier, e.g. 'TASK-001'")
    title: str = Field(..., description="Short human-readable title")
    description: str = Field(default="", description="Detailed specification")
    file_path: str = Field(default="", description="Target file path for implementation")
    task_type: str = Field(
        default="implementation",
        description="implementation | configuration | test | documentation | integration",
    )
    assigned_to: str = Field(
        default="unassigned",
        description="Agent role responsible: architect | backend_engineer | frontend_engineer | tester | reviewer",
    )
    priority: int = Field(
        default=2,
        ge=0,
        le=4,
        description="0 = critical … 4 = nice-to-have",
    )
    status: str = Field(
        default="pending",
        description="pending | in_progress | completed | failed | blocked",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of task_ids that must finish first",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Verification criteria for this task",
    )
    related_requirements: list[str] = Field(
        default_factory=list,
        description="Requirement IDs addressed by this task",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeArtifact(BaseModel):
    """A file produced (or modified) by a developer agent."""

    file_path: str = Field(..., description="Relative path inside the output project")
    language: str = Field(default="python")
    content: str = Field(default="", description="Full source code of the file")
    version: int = Field(default=1, description="Incremented on every rewrite")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    tests_passed: bool | None = Field(
        default=None,
        description="None = not tested yet, True/False = last test result",
    )


class ArchitectureDecision(BaseModel):
    """An ADR (Architecture Decision Record) produced by the architect agent."""

    decision_id: str = Field(..., description="e.g. 'ADR-001'")
    title: str
    context: str = Field(default="", description="Why this decision was needed")
    decision: str = Field(default="", description="What was decided")
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Alternatives evaluated",
    )
    consequences: str = Field(default="", description="Trade-offs and implications")
    status: str = Field(
        default="proposed", description="proposed | accepted | superseded"
    )


# Alias for backward compatibility
ADR = ArchitectureDecision


class ErrorRecord(BaseModel):
    """Structured record of a node failure for observability."""

    node_name: str
    error_type: str
    error_message: str
    traceback: str = Field(default="")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    attempt: int = Field(default=1, description="Which retry attempt this was")
    resolved: bool = Field(default=False)


class ExecutionTraceItem(BaseModel):
    """Structured record of a tool execution event in the sandbox."""

    tool: str = Field(..., description="Tool name (e.g. filesystem, git, executor)")
    operation: str = Field(..., description="Operation performed (e.g. write_file, commit)")
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: Any = Field(default=None)
    success: bool = Field(default=True)
    duration_ms: float = Field(default=0.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────────────────────
#  ProjectState — LangGraph graph state (TypedDict)
# ─────────────────────────────────────────────────────────────


class ProjectState(TypedDict, total=False):
    """
    Central state that flows through every node of the LangGraph pipeline.

    All state mutation occurs through typed reducers:
    - architecture_decisions: Deduplicated by decision_id via adr_reducer.
    - task_queue: Upserted by task_id via task_queue_reducer.
    - completed_tasks: Clearable list via clearable_list_reducer.
    - code_artifacts: Deduplicated by file_path with auto-versioning via artifact_reducer.
    - execution_trace: Clearable list via clearable_list_reducer.
    - retry_counts: Merged by node_name via dict_merge_reducer.
    - task_failures: Merged by file_path/task_id via dict_merge_reducer.
    - error_log: Clearable list via clearable_list_reducer.
    """

    # ── Identity & Configuration ──
    project_id: str
    project_name: str
    requirements: str
    workspace_dir: str
    active_branch: str

    # ── Planning Artefacts ──
    technical_specifications: dict[str, Any]
    project_structure: dict[str, Any]
    architecture_decisions: Annotated[list[dict[str, Any]], adr_reducer]

    # ── Task Management ──
    task_queue: Annotated[list[dict[str, Any]], task_queue_reducer]
    completed_tasks: Annotated[list[dict[str, Any]], clearable_list_reducer]

    # ── Code Output & Artifacts ──
    code_artifacts: Annotated[list[dict[str, Any]], artifact_reducer]

    # ── Execution Sandbox & Trace ──
    execution_trace: Annotated[list[dict[str, Any]], clearable_list_reducer]

    # ── Observability & Reliability ──
    retry_counts: Annotated[dict[str, int], dict_merge_reducer]
    task_failures: Annotated[dict[str, int], dict_merge_reducer]
    error_log: Annotated[list[dict[str, Any]], clearable_list_reducer]

    # ── Lifecycle & Control Flow ──
    current_phase: str
    iteration: int
    max_retries: int
    status: str
