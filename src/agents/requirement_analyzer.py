"""
Requirement Analyzer Agent
==========================

Parses raw natural-language user requirements into a structured
``TechnicalSpecification`` Pydantic model using an LLM with structured
output.

Pipeline position:  initializer --> **requirement_analyzer** --> architect

Output written to state:
    - ``technical_specifications``  (dict -- serialised TechnicalSpecification)
"""

from __future__ import annotations

import json
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


class FunctionalRequirement(BaseModel):
    """A single functional requirement extracted from the user input."""

    id: str = Field(..., description="e.g. FR-001")
    title: str = Field(..., description="Short title")
    description: str = Field(..., description="Detailed description")
    priority: str = Field(
        default="medium",
        description="critical | high | medium | low",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Testable acceptance criteria",
    )


class NonFunctionalRequirement(BaseModel):
    """A non-functional / quality attribute requirement."""

    id: str = Field(..., description="e.g. NFR-001")
    category: str = Field(
        ...,
        description="performance | security | scalability | reliability | usability",
    )
    description: str


class TechStackRecommendation(BaseModel):
    """Technology recommendation derived from the requirements."""

    category: str = Field(
        ..., description="language | framework | database | infrastructure | tool"
    )
    name: str = Field(..., description="e.g. Python, FastAPI, PostgreSQL")
    rationale: str = Field(..., description="Why this technology was chosen")


class TechnicalSpecification(BaseModel):
    """
    The full structured output produced by the Requirement Analyzer.
    This becomes the single source of truth for downstream agents.
    """

    project_overview: str = Field(
        ...,
        description="2-3 sentence high-level summary of what the project does",
    )
    functional_requirements: list[FunctionalRequirement] = Field(
        default_factory=list,
    )
    non_functional_requirements: list[NonFunctionalRequirement] = Field(
        default_factory=list,
    )
    tech_stack: list[TechStackRecommendation] = Field(
        default_factory=list,
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Hard constraints or limitations mentioned by the user",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made when requirements were ambiguous",
    )


# ──────────────────────────────────────────────────────────────
#  System prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a **Senior Requirements Analyst** for a software engineering team.

Your job is to take raw, possibly vague, natural-language requirements from a
client and transform them into a precise, structured Technical Specification.

Rules:
1. Extract ALL functional requirements. Give each one a unique ID (FR-001, FR-002, ...).
2. Identify non-functional requirements (performance, security, etc.).  ID them NFR-001, NFR-002, ...
3. Recommend a tech stack with clear rationale for each choice.
4. List any constraints the user mentioned explicitly.
5. List assumptions you are making where the requirements are ambiguous.
6. Write a concise project overview (2-3 sentences).
7. Be thorough -- do NOT skip requirements.  If the user mentions auth, CRUD,
   specific databases, etc., each one must appear as a separate FR.
8. Acceptance criteria must be concrete and testable.

Respond ONLY with the JSON matching the schema provided. No markdown fences,
no extra commentary.
"""


# ──────────────────────────────────────────────────────────────
#  Node function
# ──────────────────────────────────────────────────────────────


@retry_middleware(max_retries=3)
def requirement_analyzer_node(state: ProjectState) -> dict[str, Any]:
    """
    LangGraph node -- invokes the LLM to parse raw requirements into a
    ``TechnicalSpecification``.

    Reads:   state["requirements"]
    Writes:  state["technical_specifications"]
    """
    requirements = state.get("requirements", "")
    if not requirements:
        logger.warning("[REQ] No requirements provided -- returning empty spec")
        return {
            "technical_specifications": TechnicalSpecification(
                project_overview="No requirements provided.",
            ).model_dump(),
        }

    logger.info("[REQ] Analysing requirements (%d chars) ...", len(requirements))

    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(TechnicalSpecification)

    spec: TechnicalSpecification = structured_llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": requirements},
        ]
    )

    spec_dict = spec.model_dump()

    logger.info(
        "[REQ] Extracted %d functional, %d non-functional requirements, %d tech-stack items",
        len(spec.functional_requirements),
        len(spec.non_functional_requirements),
        len(spec.tech_stack),
    )

    return {
        "technical_specifications": spec_dict,
    }
