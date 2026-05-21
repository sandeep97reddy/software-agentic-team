"""
Memory Compression Agent
========================

Summarizes completed_tasks and clears raw execution logs to keep
the token context window slim, while preserving Architectural context.
"""

import logging
from typing import Any

from src.core.config import get_llm
from src.core.middleware import retry_middleware
from src.core.state import ProjectState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert technical summarizer.
Summarize the following list of completed tasks into a single concise paragraph.
This summary will replace the raw list of tasks to save tokens. Do not lose 
critical details about what features, components, or files were successfully implemented.
"""


@retry_middleware(max_retries=3)
def memory_compression_node(state: ProjectState) -> dict[str, Any]:
    completed_tasks = state.get("completed_tasks", [])

    # We trigger compression if there are more than 5 completed tasks
    if len(completed_tasks) < 5:
        return {}

    logger.info(
        f"[MEMORY] Compressing {len(completed_tasks)} completed tasks to save tokens..."
    )

    tasks_text = "\n".join(
        [
            f"- {t.get('title')}: {t.get('status')} (File: {t.get('file_path', 'N/A')})"
            for t in completed_tasks
        ]
    )

    llm = get_llm(temperature=0.1)
    result = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tasks_text},
        ]
    )

    summary_text = result.content if hasattr(result, "content") else str(result)

    compressed_task = {
        "task_id": "COMPRESSED",
        "title": "Compressed Task History",
        "description": summary_text,
        "status": "completed",
        "metadata": {"compressed": True},
    }

    logger.info("[MEMORY] Compression complete. Clearing raw trace logs.")

    # Using the special 'CLEAR' command for our custom reducer
    return {"completed_tasks": ["CLEAR", compressed_task], "execution_trace": "CLEAR"}
