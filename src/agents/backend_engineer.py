"""
Backend Engineer Agent
======================

Takes a task from the task_queue, references architecture_decisions,
and writes Python/FastAPI code using the tools from Chunk 3.

Implements a Stagnation Check to avoid infinite loops if the LLM
keeps generating the exact same code.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.core.config import get_llm
from src.core.middleware import retry_middleware
from src.core.state import ProjectState
from src.tools.filesystem import FileSystemManager
from src.tools.git_tracker import GitTracker

logger = logging.getLogger(__name__)


class GeneratedCode(BaseModel):
    file_path: str = Field(description="The path where the code should be written")
    content: str = Field(description="The full source code to write")
    explanation: str = Field(description="Brief explanation of the implementation")


SYSTEM_PROMPT = """\
You are an expert Backend Software Engineer specialising in Python and FastAPI.
Your job is to implement a specific task from the task queue.
You will be provided with the project's architecture decisions, project structure, and the task details.
Write clean, production-ready, well-documented Python code.
Output the COMPLETE file content. Do not use placeholders like '# implementation here' or '# rest of code'.
"""


@retry_middleware(max_retries=3)
def backend_engineer_node(state: ProjectState) -> dict[str, Any]:
    task_queue = list(state.get("task_queue", []))
    if not task_queue:
        return {}

    task = task_queue[0]
    workspace_dir = state.get("workspace_dir", "")
    trace: list[dict[str, Any]] = []

    fs = FileSystemManager(workspace_dir, trace)
    git = GitTracker(workspace_dir, trace)

    file_path = task.get("file_path", "")

    # Stagnation Check: Read current file content hash if it exists
    current_content = ""
    try:
        current_content = fs.read_file(file_path)
    except FileNotFoundError:
        pass

    user_message = f"""
## Task: {task.get('title')}
Description: {task.get('description')}
Target File: {file_path}
Acceptance Criteria: {task.get('acceptance_criteria', [])}

## Architecture Decisions
{state.get('architecture_decisions', [])}

## Project Structure
{state.get('project_structure', {})}

Please provide the complete implementation for {file_path}.
"""

    llm = get_llm(temperature=0.2, max_tokens=8192)
    structured_llm = llm.with_structured_output(GeneratedCode)

    logger.info(f"[BACKEND] Working on task: {task.get('title')} -> {file_path}")
    result: GeneratedCode = structured_llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    # Write the new code
    fs.write_file(result.file_path, result.content)

    new_content = result.content

    if new_content.strip() == current_content.strip() and new_content.strip() != "":
        # Stagnation detected
        task_metadata = task.get("metadata", {})
        stagnant_iterations = task_metadata.get("stagnant_iterations", 0) + 1
        task_metadata["stagnant_iterations"] = stagnant_iterations
        task["metadata"] = task_metadata

        logger.warning(
            f"[BACKEND] Code stagnated for {result.file_path} (Iteration {stagnant_iterations})"
        )

        if stagnant_iterations >= 2:
            task["status"] = "failed"
            error_msg = f"Stagnation Check Failed: Code hasn't changed in 2 iterations for {result.file_path}"
            logger.error(f"[BACKEND] {error_msg}")
            # Do NOT pop the task so it fails the pipeline, or pop it to continue?
            # If we fail the task, maybe we pop it so the pipeline continues?
            # Let's pop it but mark it failed, and add error to state.
            failed_task = task_queue.pop(0)
            return {
                "status": "failed",
                "task_queue": task_queue,
                "completed_tasks": [failed_task],
                "error_log": [
                    {
                        "node_name": "backend_engineer",
                        "error_type": "StagnationError",
                        "error_message": error_msg,
                        "attempt": 1,
                        "resolved": False,
                    }
                ],
                "execution_trace": trace,
            }

        # Put back in queue to try one more time
        task_queue[0] = task
        return {"task_queue": task_queue, "execution_trace": trace}

    # Success, commit changes
    git.stage_all()
    git.commit(f"feat(backend): {task.get('title')}")

    completed_task = task_queue.pop(0)
    completed_task["status"] = "completed"

    artifact = {
        "file_path": result.file_path,
        "language": "python",
        "content": result.content,
        "version": 1,
    }

    logger.info(f"[BACKEND] Task completed: {task.get('title')}")
    return {
        "task_queue": task_queue,
        "completed_tasks": [completed_task],
        "code_artifacts": [artifact],
        "execution_trace": trace,
    }
