"""
Code Reviewer Agent
===================

Scans code for security flaws and hallucinated imports.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.core.config import get_llm
from src.core.middleware import retry_middleware
from src.core.state import ProjectState

logger = logging.getLogger(__name__)


class ReviewResult(BaseModel):
    approved: bool = Field(
        description="True if code is secure and has no hallucinated imports"
    )
    feedback: str = Field(
        description="Details of security flaws or hallucinated imports found. Empty if approved."
    )


SYSTEM_PROMPT = """\
You are an expert Security and Code Reviewer.
Scan the provided code for:
1. Security flaws (e.g., SQL injection, hardcoded secrets, XSS).
2. 'Hallucinated' or non-existent library imports.
Return approved=True if clean, otherwise approved=False with specific actionable feedback.
"""


@retry_middleware(max_retries=3)
def reviewer_node(state: ProjectState) -> dict[str, Any]:
    artifacts = state.get("code_artifacts", [])
    task_queue = list(state.get("task_queue", []))
    retry_counts = dict(state.get("retry_counts", {}))

    llm = get_llm(temperature=0.1, max_tokens=2048)
    structured_llm = llm.with_structured_output(ReviewResult)

    review_failed = False

    for artifact in artifacts:
        file_path = artifact.get("file_path", "")
        # Only review if it's actual source code, maybe skip tests
        if "test_" in file_path:
            continue

        logger.info(f"[REVIEW] Scanning {file_path} for security flaws...")
        result: ReviewResult = structured_llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"File: {file_path}\nCode:\n{artifact.get('content')}",
                },
            ]
        )

        if not result.approved:
            logger.warning(f"[REVIEW] Issues found in {file_path}: {result.feedback}")
            review_failed = True

            task_key = f"task_fail_{file_path}"
            retry_counts[task_key] = retry_counts.get(task_key, 0) + 1

            filename = file_path.split("/")[-1]
            fix_task = {
                "task_id": f"REVIEW-FIX-{filename}",
                "title": f"Address review feedback for {file_path}",
                "description": f"Code review failed for {file_path}. Feedback:\n{result.feedback}",
                "file_path": file_path,
                "assigned_to": (
                    "backend_engineer"
                    if artifact.get("language") == "python"
                    else "frontend_engineer"
                ),
                "status": "pending",
                "metadata": {"task_key": task_key},
            }
            if not any(t.get("file_path") == file_path for t in task_queue):
                task_queue.append(fix_task)

    if review_failed:
        return {
            "task_queue": task_queue,
            "retry_counts": retry_counts,
            "current_phase": "review",
        }

    logger.info("[REVIEW] All code approved!")
    return {"status": "completed", "current_phase": "completed"}
