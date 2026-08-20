"""
QA Tester Agent
===============

Automatically generates pytest files for any new python code in code_artifacts
and runs them via the SubprocessExecutor tool.
Handles Pass/Fail mapping by re-queueing tasks for failures.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.core.config import get_llm
from src.core.middleware import retry_middleware
from src.core.state import ProjectState
from src.tools.executor import SubprocessExecutor
from src.tools.filesystem import FileSystemManager

logger = logging.getLogger(__name__)


class TestCode(BaseModel):
    test_file_path: str = Field(description="Path to the pytest file to create")
    test_code: str = Field(description="Full pytest source code")


# Tell pytest not to discover TestCode as a test suite
TestCode.__test__ = False  # type: ignore[attr-defined]


SYSTEM_PROMPT = """\
You are an expert QA Tester.
Given the source code of a newly implemented Python file, write comprehensive pytest test cases for it.
Ensure the test file includes all necessary imports and uses mocking where appropriate.
Return the complete test file code. Do not use placeholders.
"""


@retry_middleware(max_retries=3)
def tester_node(state: ProjectState) -> dict[str, Any]:
    artifacts = state.get("code_artifacts", [])
    task_queue = state.get("task_queue", [])
    retry_counts = dict(state.get("retry_counts", {}))
    task_failures = dict(state.get("task_failures", {}))
    workspace_dir = state.get("workspace_dir", "")
    trace: list[dict[str, Any]] = []

    exe = SubprocessExecutor(workspace_dir, trace)
    fs = FileSystemManager(workspace_dir, trace)

    tests_failed = False
    new_tasks: list[dict[str, Any]] = []
    updated_artifacts: list[dict[str, Any]] = []

    # We test any python artifact that doesn't have a test file yet
    for artifact in artifacts:
        if artifact.get("language") != "python":
            continue

        file_path = artifact.get("file_path", "")
        # Focus on source files
        if not file_path.endswith(".py") or "test_" in file_path:
            continue

        filename = file_path.split("/")[-1]
        test_file_path = "tests/test_" + filename

        # If test doesn't exist, generate it
        if not fs.file_exists(test_file_path):
            logger.info(f"[QA] Generating tests for {file_path}")
            llm = get_llm(temperature=0.2, max_tokens=8192)
            structured_llm = llm.with_structured_output(TestCode)
            result: TestCode = structured_llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Target File: {file_path}\nCode:\n{artifact.get('content')}",
                    },
                ]
            )
            fs.write_file(result.test_file_path, result.test_code)
            test_file_path = result.test_file_path

        # Run test for this specific file
        logger.info(f"[QA] Running pytest on {test_file_path}...")
        test_result = exe.run_pytest(test_file_path)

        if not test_result.success:
            logger.warning(f"[QA] Test failed for {file_path}")
            tests_failed = True

            # Track failure count for Watchdog and retry bookkeeping
            task_key = f"task_fail_{file_path}"
            retry_counts[task_key] = retry_counts.get(task_key, 0) + 1
            task_failures[file_path] = task_failures.get(file_path, 0) + 1

            # Put task back in queue to be fixed
            fix_task = {
                "task_id": f"FIX-{filename}",
                "title": f"Fix test failures in {file_path}",
                "description": f"Tests failed for {file_path}. Pytest output:\n{test_result.stdout}\n{test_result.stderr}",
                "file_path": file_path,
                "assigned_to": "backend_engineer",
                "status": "pending",
                "metadata": {"task_key": file_path},
            }
            # Only add if not already in queue
            if not any(
                t.get("file_path") == file_path and t.get("status") == "pending"
                for t in task_queue
            ):
                new_tasks.append(fix_task)

            updated_artifacts.append(
                {"file_path": file_path, "tests_passed": False}
            )
        else:
            updated_artifacts.append(
                {"file_path": file_path, "tests_passed": True}
            )

    if not tests_failed:
        logger.info("[QA] All tests passed!")

    return {
        "task_queue": new_tasks,
        "retry_counts": retry_counts,
        "task_failures": task_failures,
        "code_artifacts": updated_artifacts,
        "execution_trace": trace,
        "current_phase": "testing",
    }


# Tell pytest not to discover tester_node as a test function
tester_node.__test__ = False  # type: ignore[attr-defined]
