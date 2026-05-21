"""
Task Planner Agent
==================

Decomposes the Architecture Blueprint into a prioritised queue of atomic,
developer-ready tasks.

Pipeline position:  architect --> **task_planner** --> developer

Each task maps to a single file or small group of closely related files.
Dependencies between tasks are tracked so the developer can process them
in the correct order.

Output written to state:
    - ``task_queue``   (list[dict] -- appended via operator.add)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.core.config import get_llm
from src.core.middleware import retry_middleware
from src.core.state import ProjectState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Structured output schema
# ──────────────────────────────────────────────────────────────


class PlannedTask(BaseModel):
    """A single atomic task for the developer agent."""

    task_id: str = Field(..., description="Unique ID, e.g. TASK-001")
    title: str = Field(..., description="Short, actionable title")
    description: str = Field(
        ...,
        description=(
            "Detailed implementation instructions: what to build, which "
            "patterns to follow, expected inputs/outputs"
        ),
    )
    file_path: str = Field(
        ...,
        description="Primary file this task produces or modifies",
    )
    task_type: str = Field(
        default="implementation",
        description=(
            "implementation | configuration | test | documentation | integration"
        ),
    )
    assigned_to: str = Field(
        default="developer",
        description="Agent role: developer | tester | devops",
    )
    priority: int = Field(
        default=2,
        ge=0,
        le=4,
        description="0 = critical (must do first) ... 4 = nice-to-have",
    )
    estimated_complexity: str = Field(
        default="medium",
        description="low | medium | high",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of task_ids that must be completed first",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="How to verify this task is done correctly",
    )
    related_requirements: list[str] = Field(
        default_factory=list,
        description="FR/NFR IDs this task addresses",
    )


class TaskPlan(BaseModel):
    """The complete task breakdown produced by the Task Planner."""

    tasks: list[PlannedTask] = Field(
        ...,
        description="Ordered list of atomic tasks",
    )
    execution_order_rationale: str = Field(
        default="",
        description="Brief explanation of why tasks are ordered this way",
    )
    estimated_total_files: int = Field(
        default=0,
        description="Total number of files that will be created or modified",
    )


# ──────────────────────────────────────────────────────────────
#  System prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a **Technical Project Manager** breaking down an architecture blueprint
into atomic, developer-ready tasks.

Rules:
1. Each task MUST map to a single file (or at most 2 tightly coupled files).
2. Give each task a unique ID: TASK-001, TASK-002, ...
3. Order tasks so that dependencies are respected (foundations first).
4. The ``description`` field must contain CONCRETE implementation instructions,
   not vague directives.  A junior developer should be able to implement the
   task from the description alone.
5. Include tasks for:
   - Project configuration files (pyproject.toml, Dockerfile, .env, etc.)
   - Database models / schemas
   - Core business logic / service layer
   - API route handlers
   - Authentication / middleware
   - Tests for every module
   - Documentation (README, API docs)
6. Set ``priority`` correctly: config & models = 0-1, business logic = 1-2,
   routes = 2, tests = 3, docs = 4.
7. Set ``dependencies`` accurately.  E.g., route tasks depend on service tasks,
   service tasks depend on model tasks.
8. Write clear ``acceptance_criteria`` for each task (at least 2 per task).
9. Link tasks to the relevant FR/NFR IDs via ``related_requirements``.
10. Aim for 8-20 tasks for a typical project.  Too few = too coarse;
    too many = unnecessarily granular.

Respond ONLY with the JSON matching the schema provided. No markdown fences.
"""


# ──────────────────────────────────────────────────────────────
#  Node function
# ──────────────────────────────────────────────────────────────


@retry_middleware(max_retries=3)
def task_planner_node(state: ProjectState) -> dict[str, Any]:
    """
    LangGraph node -- invokes the LLM to decompose the architecture
    into atomic tasks for the developer.

    Reads:   state["project_structure"], state["technical_specifications"],
             state["architecture_decisions"]
    Writes:  state["task_queue"], state["current_phase"]
    """
    project_structure = state.get("project_structure", {})
    tech_spec = state.get("technical_specifications", {})
    arch_decisions = state.get("architecture_decisions", [])

    if not project_structure:
        logger.warning("[PLAN] No project structure found -- cannot plan tasks")
        return {
            "task_queue": [],
            "current_phase": "development",
        }

    logger.info("[PLAN] Decomposing architecture into tasks ...")

    # Build context for the LLM
    user_message = "## Architecture Blueprint\n\n"

    user_message += "### Project Structure (Files)\n"
    for f in project_structure.get("files", []):
        user_message += f"- {f.get('path')} ({f.get('type')}) -- {f.get('purpose')}\n"

    user_message += "\n### API Endpoints\n"
    for ep in project_structure.get("api_endpoints", []):
        auth_tag = " [AUTH]" if ep.get("auth_required") else ""
        user_message += (
            f"- {ep.get('method')} {ep.get('path')}{auth_tag} -- {ep.get('summary')}\n"
        )

    user_message += "\n### Database Tables\n"
    for tbl in project_structure.get("database_tables", []):
        cols = ", ".join(
            f"{c.get('name')}:{c.get('type')}" for c in tbl.get("columns", [])
        )
        user_message += f"- {tbl.get('name')}: [{cols}]\n"
        for rel in tbl.get("relationships", []):
            user_message += f"  Relationship: {rel}\n"

    user_message += (
        f"\n### Design Patterns: {project_structure.get('design_patterns', [])}\n"
    )
    user_message += f"### Architecture Style: {project_structure.get('architecture_style', 'N/A')}\n"

    user_message += "\n### Architecture Decisions\n"
    for adr in arch_decisions:
        user_message += (
            f"- [{adr.get('decision_id')}] {adr.get('title')}: {adr.get('decision')}\n"
        )

    # Include tech spec overview for context
    user_message += (
        f"\n### Project Overview\n{tech_spec.get('project_overview', 'N/A')}\n"
    )

    user_message += "\n### Functional Requirements\n"
    for fr in tech_spec.get("functional_requirements", []):
        user_message += f"- [{fr.get('id')}] {fr.get('title')}\n"

    llm = get_llm(temperature=0.2, max_tokens=8192)
    structured_llm = llm.with_structured_output(TaskPlan)

    plan: TaskPlan = structured_llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    # Convert tasks to dicts for the state
    task_dicts = []
    for task in plan.tasks:
        task_dict = task.model_dump()
        task_dict["status"] = "pending"  # all tasks start as pending
        task_dicts.append(task_dict)

    logger.info(
        "[PLAN] Created %d tasks (estimated %d files). Order rationale: %s",
        len(plan.tasks),
        plan.estimated_total_files,
        (
            plan.execution_order_rationale[:120] + "..."
            if len(plan.execution_order_rationale) > 120
            else plan.execution_order_rationale
        ),
    )

    return {
        "task_queue": task_dicts,
        "current_phase": "development",
    }
