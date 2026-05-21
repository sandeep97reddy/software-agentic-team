"""
ProjectState — the single Source of Truth for the entire LangGraph pipeline.

Every node in the graph reads from and writes to this TypedDict.  LangGraph
uses the Annotated[..., operator.add] pattern to *merge* list values returned
by different nodes rather than overwriting them.

Design decisions
────────────────
• TypedDict (not a dataclass/BaseModel) because LangGraph's StateGraph
  expects a TypedDict schema; Pydantic validation lives at the API boundary
  instead (see `src/api/schemas.py` in later chunks).
• All mutable collections use `operator.add` so parallel branches can safely
  append without conflicts.
• `retry_counts` maps  node_name → cumulative failure count  and is used by
  the retry middleware to decide whether to re-invoke or abort.
• `error_log` gives full observability into what went wrong and when.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


def clearable_list_reducer(existing: list | None, new: list | str) -> list:
    """Reducer that appends lists, but clears them if the string 'CLEAR' is passed."""
    if existing is None:
        existing = []
    if new == "CLEAR":
        return []
    if isinstance(new, list):
        if new and new[0] == "CLEAR":
            return new[1:]
        return existing + new
    return existing + [new]


# ─────────────────────────────────────────────────────────────
#  Sub-models  (Pydantic)  — structured data inside the state
# ─────────────────────────────────────────────────────────────


class TaskItem(BaseModel):
    """A discrete unit of work to be completed by an agent."""

    task_id: str = Field(..., description="Unique identifier, e.g. 'TASK-001'")
    title: str = Field(..., description="Short human-readable title")
    description: str = Field(default="", description="Detailed specification")
    assigned_to: str = Field(
        default="unassigned",
        description="Agent role responsible: architect | developer | tester | reviewer",
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeArtifact(BaseModel):
    """A file produced (or modified) by the developer agent."""

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
    consequences: str = Field(default="", description="Trade-offs and implications")
    status: str = Field(
        default="proposed", description="proposed | accepted | superseded"
    )


class ErrorRecord(BaseModel):
    """Structured record of a node failure for observability."""

    node_name: str
    error_type: str
    error_message: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    attempt: int = Field(default=1, description="Which retry attempt this was")
    resolved: bool = Field(default=False)


# ─────────────────────────────────────────────────────────────
#  ProjectState  —  LangGraph graph state (TypedDict)
# ─────────────────────────────────────────────────────────────


class ProjectState(TypedDict, total=False):
    """
    Central state that flows through every node of the LangGraph pipeline.

    Fields using ``Annotated[list[…], operator.add]`` are *append-merged*:
    a node can return ``{"task_queue": [new_task]}`` and LangGraph will
    concatenate it with the existing list rather than replacing it.
    """

    # ── identity ──────────────────────────────────────────────
    project_id: str
    project_name: str

    # ── inputs ────────────────────────────────────────────────
    requirements: str  # raw user requirements (natural language)

    # ── planning artefacts ────────────────────────────────────
    architecture_decisions: Annotated[list[dict[str, Any]], operator.add]
    technical_specifications: dict[str, Any]  # output of Requirement Analyzer
    project_structure: dict[str, Any]  # output of Architect (folder tree + API schema)

    # ── work items ────────────────────────────────────────────
    task_queue: list[dict[str, Any]]
    completed_tasks: Annotated[list[dict[str, Any]], clearable_list_reducer]

    # ── code output ───────────────────────────────────────────
    code_artifacts: Annotated[list[dict[str, Any]], operator.add]

    # ── execution sandbox (Chunk 3) ───────────────────────────
    workspace_dir: str  # absolute path to the sandboxed output workspace
    active_branch: str  # git branch currently checked out in workspace
    execution_trace: Annotated[
        list[dict[str, Any]], clearable_list_reducer
    ]  # audit log of every tool call

    # ── reliability / observability ───────────────────────────
    retry_counts: dict[str, int]  # node_name → failure count
    error_log: Annotated[list[dict[str, Any]], operator.add]

    # ── control flow ──────────────────────────────────────────
    current_phase: str  # planning | development | testing | review
    iteration: int  # global loop counter
    max_retries: int  # per-node retry ceiling (default set in graph)
    status: str  # initialized | running | completed | failed
