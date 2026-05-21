"""
Architect Agent
===============

Takes the structured ``TechnicalSpecification`` from the Requirement Analyzer
and produces:

1. A **project folder structure** (JSON tree).
2. An **API schema** (endpoints, methods, request/response shapes).
3. One or more **Architecture Decision Records** (ADRs).

Pipeline position:  requirement_analyzer --> **architect** --> task_planner

Output written to state:
    - ``architecture_decisions``   (list[dict] -- appended via operator.add)
    - ``project_structure``        (dict       -- the folder tree + API schema)
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
#  Structured output schemas
# ──────────────────────────────────────────────────────────────


class FileNode(BaseModel):
    """A single file or directory in the project tree."""

    path: str = Field(..., description="Relative path, e.g. 'src/api/routes.py'")
    type: str = Field(..., description="file | directory")
    purpose: str = Field(..., description="What this file/directory is for")


class EndpointSchema(BaseModel):
    """Specification of a single API endpoint."""

    method: str = Field(
        ..., description="HTTP method: GET | POST | PUT | PATCH | DELETE"
    )
    path: str = Field(..., description="URL path, e.g. '/api/v1/todos'")
    summary: str = Field(..., description="One-line description")
    request_body: str = Field(
        default="",
        description="JSON schema or description of the request body",
    )
    response_body: str = Field(
        default="",
        description="JSON schema or description of the response body",
    )
    auth_required: bool = Field(default=False)


class DatabaseSchema(BaseModel):
    """A database table / collection specification."""

    name: str = Field(..., description="Table or collection name")
    columns: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {name, type, constraints} dicts",
    )
    relationships: list[str] = Field(
        default_factory=list,
        description="e.g. 'users.id -> todos.user_id'",
    )


class ArchitectureBlueprint(BaseModel):
    """
    The complete structured output from the Architect Agent.
    """

    project_structure: list[FileNode] = Field(
        ...,
        description="Complete folder/file tree for the project",
    )
    api_endpoints: list[EndpointSchema] = Field(
        default_factory=list,
        description="All REST API endpoints",
    )
    database_tables: list[DatabaseSchema] = Field(
        default_factory=list,
        description="Database schema design",
    )
    design_patterns: list[str] = Field(
        default_factory=list,
        description="Key design patterns chosen (e.g. Repository, Service Layer)",
    )
    architecture_style: str = Field(
        default="monolith",
        description="monolith | microservices | serverless | modular-monolith",
    )
    adrs: list[ADR] = Field(
        default_factory=list,
        description="Architecture Decision Records",
    )


class ADR(BaseModel):
    """Architecture Decision Record produced by the Architect."""

    decision_id: str = Field(..., description="e.g. ADR-001")
    title: str
    context: str = Field(default="", description="Why this decision was needed")
    decision: str = Field(default="", description="What was decided")
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Other options that were evaluated",
    )
    consequences: str = Field(default="", description="Trade-offs and implications")
    status: str = Field(
        default="accepted", description="proposed | accepted | superseded"
    )


# Fix the forward reference -- ADR is used before definition in ArchitectureBlueprint
ArchitectureBlueprint.model_rebuild()


# ──────────────────────────────────────────────────────────────
#  System prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a **Principal Software Architect** designing a production-grade system.

You receive a Technical Specification (functional requirements, non-functional
requirements, tech stack) and must produce a complete Architecture Blueprint.

Rules:
1. **Project Structure**: Design a clean, production-ready folder/file tree.
   Include ALL files needed: source code modules, config files, tests,
   Dockerfiles, etc.  Each entry must have a clear "purpose".
2. **API Endpoints**: Define every REST endpoint with method, path, summary,
   request/response body descriptions, and whether auth is required.
3. **Database Schema**: Define tables with column names, types, and constraints.
   Include relationships (foreign keys).
4. **Design Patterns**: List the key design patterns you are using and why.
5. **Architecture Style**: Choose and justify the architecture style.
6. **ADRs**: Write 2-4 Architecture Decision Records for the most impactful
   decisions (e.g. framework choice, database choice, auth strategy, project
   structure approach).  Give each a unique ID (ADR-001, ADR-002, ...).
7. Be comprehensive.  A developer should be able to start coding from your
   blueprint without ambiguity.

Respond ONLY with the JSON matching the schema provided. No markdown fences.
"""


# ──────────────────────────────────────────────────────────────
#  Node function
# ──────────────────────────────────────────────────────────────


@retry_middleware(max_retries=3)
def architect_node(state: ProjectState) -> dict[str, Any]:
    """
    LangGraph node -- invokes the LLM to produce an ArchitectureBlueprint
    from the technical specifications.

    Reads:   state["technical_specifications"], state["requirements"]
    Writes:  state["architecture_decisions"], state["project_structure"]
    """
    tech_spec = state.get("technical_specifications", {})
    raw_requirements = state.get("requirements", "")

    if not tech_spec:
        logger.warning("[ARCH] No technical specifications found -- skipping")
        return {
            "architecture_decisions": [],
            "current_phase": "development",
        }

    logger.info("[ARCH] Designing architecture from technical spec ...")

    # Build the user message with full context
    user_message = (
        f"## Original Requirements\n{raw_requirements}\n\n"
        f"## Technical Specification\n"
        f"Project Overview: {tech_spec.get('project_overview', 'N/A')}\n\n"
        f"Functional Requirements:\n"
    )
    for fr in tech_spec.get("functional_requirements", []):
        user_message += (
            f"- [{fr.get('id')}] {fr.get('title')}: {fr.get('description')}\n"
        )

    user_message += f"\nNon-Functional Requirements:\n"
    for nfr in tech_spec.get("non_functional_requirements", []):
        user_message += (
            f"- [{nfr.get('id')}] ({nfr.get('category')}) {nfr.get('description')}\n"
        )

    user_message += f"\nTech Stack:\n"
    for ts in tech_spec.get("tech_stack", []):
        user_message += (
            f"- {ts.get('category')}: {ts.get('name')} -- {ts.get('rationale')}\n"
        )

    user_message += f"\nConstraints: {tech_spec.get('constraints', [])}\n"
    user_message += f"Assumptions: {tech_spec.get('assumptions', [])}\n"

    llm = get_llm(temperature=0.2, max_tokens=8192)
    structured_llm = llm.with_structured_output(ArchitectureBlueprint)

    blueprint: ArchitectureBlueprint = structured_llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    # Convert ADRs into the state's architecture_decisions format
    arch_decisions = []
    for adr in blueprint.adrs:
        arch_decisions.append(
            {
                "decision_id": adr.decision_id,
                "title": adr.title,
                "context": adr.context,
                "decision": adr.decision,
                "alternatives_considered": adr.alternatives_considered,
                "consequences": adr.consequences,
                "status": adr.status,
            }
        )

    # Build the project_structure payload
    project_structure = {
        "files": [f.model_dump() for f in blueprint.project_structure],
        "api_endpoints": [e.model_dump() for e in blueprint.api_endpoints],
        "database_tables": [t.model_dump() for t in blueprint.database_tables],
        "design_patterns": blueprint.design_patterns,
        "architecture_style": blueprint.architecture_style,
    }

    logger.info(
        "[ARCH] Blueprint complete: %d files, %d endpoints, %d tables, %d ADRs",
        len(blueprint.project_structure),
        len(blueprint.api_endpoints),
        len(blueprint.database_tables),
        len(blueprint.adrs),
    )

    return {
        "architecture_decisions": arch_decisions,
        "project_structure": project_structure,
        "current_phase": "planning",
    }
