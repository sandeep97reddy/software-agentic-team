"""
tests/conftest.py
=================
Comprehensive, production-grade test fixtures and mocking harness for all test tiers
(Unit, Integration, E2E Tiers 1-4, Adversarial) of the Autonomous AI Software
Engineering Multi-Agent Team.

Fixture Inventory:
------------------
1. Mock LLM & Agent Simulation:
   - `MockChatModel`: Production-grade LangChain-compatible chat model test double.
   - `StructuredMockChatModel`: Structured output wrapper (`with_structured_output`).
   - `mock_llm_factory`: Factory for configurable mock LLMs with auto-patching of `get_llm`.
   - `mock_llm`: Pre-patched default mock LLM.
   - Error simulation classes (`MockRateLimitError`, `MockContextWindowOverflowError`, etc.).
   - Tool call synthesis helpers (`make_tool_call`, `make_find_files_call`, etc.).

2. Sandboxed Workspaces:
   - `WorkspaceHelper`: Utility wrapper for sandboxed directory operations.
   - `temp_workspace`: Pytest fixture yielding a clean, isolated temporary workspace.
   - `populated_workspace`: Pre-scaffolded workspace with starter Python code and tests.

3. Version Control & Git:
   - `isolated_git_repo`: Pytest fixture with a real initialized Git repo, branches, and commits.
   - `git_tracker_factory`: Factory creating configured `GitTracker` instances.

4. LangGraph Checkpointing & Persistence:
   - `MockCheckpointer`: Enhanced in-memory checkpointer with crash simulation and fault injection.
   - `mock_checkpointer`: Pytest fixture yielding a clean `MockCheckpointer`.
   - `file_checkpointer`: Fixture for persistent file-based checkpointer testing across restarts.

5. FastAPI & Streaming API Clients:
   - `async_test_client`: `httpx.AsyncClient` fixture with ASGI transport and lifespan management.
   - `sync_test_client`: `fastapi.testclient.TestClient` fixture.
   - `api_helper`: Helper with async convenience methods for runs, polling, SSE, and WebSockets.

6. Domain Mock Data Factories:
   - `project_state_factory`, `sample_project_state`, `sample_running_state`, `sample_failed_state`, `sample_blocked_state`.
   - `task_item_factory`, `task_item_dict_factory`, `sample_task_item`, `sample_task_list`.
   - `code_artifact_factory`, `code_artifact_dict_factory`, `sample_code_artifact`.
   - `adr_factory`, `adr_dict_factory`, `sample_adr`.
   - `trace_record_factory`, `sample_trace`.

7. Tool Test Doubles:
   - `MockSubprocessExecutor`, `mock_subprocess_executor_factory`, `mock_subprocess_executor`.
   - `mock_fs_manager`, `mock_git_tracker`.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict, Union

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field

# Project imports
from src.app import app
from src.core.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE
from src.core.state import (
    ArchitectureDecision,
    CodeArtifact,
    ErrorRecord,
    ProjectState,
    TaskItem,
)
from src.tools.executor import ExecutionResult, SubprocessExecutor
from src.tools.filesystem import FileSystemManager, _make_trace
from src.tools.git_tracker import GitTracker

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Custom Exceptions for LLM & Checkpointer Simulation
# ==============================================================================


class MockRateLimitError(Exception):
    """Simulates HTTP 429 Rate Limit from LLM provider."""

    def __init__(self, message: str = "Rate limit exceeded. Please retry after 20s."):
        super().__init__(message)
        self.status_code = 429


class MockContextWindowOverflowError(Exception):
    """Simulates context length / token budget exceeded error."""

    def __init__(
        self,
        message: str = "Context window limit of 8192 tokens exceeded (prompt contained 9500 tokens).",
    ):
        super().__init__(message)
        self.status_code = 400


class MockMalformedJSONError(Exception):
    """Simulates non-parseable or truncated JSON returned by LLM."""

    def __init__(
        self,
        message: str = "Malformed JSON output from model: Unterminated string at line 1 column 42.",
    ):
        super().__init__(message)


class MockLLMTimeoutError(Exception):
    """Simulates network timeout when contacting LLM API."""

    def __init__(self, message: str = "Request to LLM gateway timed out after 30.0s."):
        super().__init__(message)


class CheckpointerCrashError(RuntimeError):
    """Simulates database crash or connection drop in state checkpointer."""

    def __init__(
        self, message: str = "Simulated checkpointer connection loss / I/O failure."
    ):
        super().__init__(message)


# ==============================================================================
# 2. Mock LLM Implementation (LangChain BaseChatModel compatible)
# ==============================================================================


class ChatCallRecord(BaseModel):
    """Structured audit record of a call made to the mock LLM."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: list[dict[str, Any]]
    kwargs: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class StructuredMockChatModel(Runnable):
    """
    Wrapper returned by ``MockChatModel.with_structured_output(schema)``.
    Converts mock responses to target Pydantic models or JSON dictionaries.
    """

    def __init__(
        self,
        mock_model: MockChatModel,
        schema: type[BaseModel] | dict[str, Any] | type,
        **kwargs: Any,
    ):
        self.mock_model = mock_model
        self.schema = schema
        self.kwargs = kwargs

    def _convert_output(self, raw_output: Any) -> Any:
        # If raw_output is an Exception, re-raise
        if isinstance(raw_output, BaseException):
            raise raw_output

        # If raw output is AIMessage, extract content or tool calls
        if isinstance(raw_output, AIMessage):
            if raw_output.tool_calls:
                # Use args from the first tool call
                raw_output = raw_output.tool_calls[0].get("args", {})
            else:
                raw_output = raw_output.content

        # Handle schema conversion for Pydantic BaseModel
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            if isinstance(raw_output, self.schema):
                return raw_output
            if isinstance(raw_output, dict):
                return self.schema.model_validate(raw_output)
            if isinstance(raw_output, str):
                try:
                    data = json.loads(raw_output)
                    return self.schema.model_validate(data)
                except Exception as exc:
                    raise MockMalformedJSONError(
                        f"Failed to parse structured JSON into {self.schema.__name__}: {exc}"
                    ) from exc

        # Handle dict / TypedDict / generic schema
        if self.schema is dict or isinstance(self.schema, dict):
            if isinstance(raw_output, dict):
                return raw_output
            if isinstance(raw_output, BaseModel):
                return raw_output.model_dump()
            if isinstance(raw_output, str):
                try:
                    return json.loads(raw_output)
                except Exception as exc:
                    raise MockMalformedJSONError(f"Malformed JSON: {exc}") from exc

        return raw_output

    def invoke(
        self,
        input: list[BaseMessage] | list[dict[str, Any]] | str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        msg = self.mock_model.invoke(input, config=config, **kwargs)
        return self._convert_output(msg)

    async def ainvoke(
        self,
        input: list[BaseMessage] | list[dict[str, Any]] | str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        msg = await self.mock_model.ainvoke(input, config=config, **kwargs)
        return self._convert_output(msg)

    def stream(
        self,
        input: list[BaseMessage] | list[dict[str, Any]] | str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        yield self.invoke(input, config=config, **kwargs)

    async def astream(
        self,
        input: list[BaseMessage] | list[dict[str, Any]] | str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        yield await self.ainvoke(input, config=config, **kwargs)


class MockChatModel(BaseChatModel):
    """
    Production-grade Mock LLM implementation adhering to LangChain's ``BaseChatModel``.

    Capabilities:
    - Predefined sequential responses (cycles or pops from a queue).
    - Tool call simulation (`tool_calls` attribute on returned AIMessage).
    - Structured output support via `with_structured_output`.
    - Token streaming generator simulation via `stream` and `astream`.
    - Programmable error injection (RateLimit, ContextWindowOverflow, Timeouts).
    - Dynamic response generators (`callable(messages) -> response`).
    - Full call history recording for audit assertions.
    """

    model_name: str = "mock-gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    responses: list[Any] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    call_history: list[ChatCallRecord] = Field(default_factory=list)
    _call_idx: int = 0
    _repeat_last: bool = True
    _streaming_chunk_size: int = 4

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        model_name: str = "mock-gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        repeat_last: bool = True,
        streaming_chunk_size: int = 4,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            responses=responses or [],
            **kwargs,
        )
        self._call_idx = 0
        self._repeat_last = repeat_last
        self._streaming_chunk_size = streaming_chunk_size
        self.call_history = []

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def set_responses(self, responses: list[Any]) -> None:
        """Reset and replace the response sequence."""
        self.responses = list(responses)
        self._call_idx = 0

    def add_response(self, response: Any) -> None:
        """Append a single response to the queue."""
        self.responses.append(response)

    def reset(self) -> None:
        """Clear call history and reset response pointer."""
        self._call_idx = 0
        self.call_history.clear()

    @property
    def call_count(self) -> int:
        """Return total number of invocations."""
        return len(self.call_history)

    @property
    def last_call(self) -> ChatCallRecord | None:
        """Return the most recent call record, or None if no calls made."""
        return self.call_history[-1] if self.call_history else None

    def _get_next_raw_response(self, messages: list[BaseMessage]) -> Any:
        """Fetch the next response from queue or evaluate dynamic callable."""
        if not self.responses:
            # Default fallback message
            return AIMessage(
                content="Mock LLM default response: task completed successfully."
            )

        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
        else:
            resp = self.responses[-1] if self._repeat_last else self.responses[0]

        # Dynamic callable evaluation
        if callable(resp) and not isinstance(resp, type):
            return resp(messages)
        return resp

    def _normalize_to_ai_message(self, raw_resp: Any) -> AIMessage:
        """Normalize various response formats into an AIMessage."""
        if isinstance(raw_resp, BaseException):
            raise raw_resp

        if isinstance(raw_resp, AIMessage):
            return raw_resp

        if isinstance(raw_resp, BaseModel):
            return AIMessage(
                content=raw_resp.model_dump_json(),
                tool_calls=[],
                response_metadata={"model_name": self.model_name},
            )

        if isinstance(raw_resp, dict):
            # Check if dict represents a tool call structure
            if "tool_calls" in raw_resp:
                return AIMessage(
                    content=raw_resp.get("content", ""),
                    tool_calls=raw_resp["tool_calls"],
                    response_metadata={"model_name": self.model_name},
                )
            return AIMessage(
                content=json.dumps(raw_resp),
                tool_calls=[],
                response_metadata={"model_name": self.model_name},
            )

        if isinstance(raw_resp, str):
            return AIMessage(
                content=raw_resp,
                tool_calls=[],
                response_metadata={"model_name": self.model_name},
            )

        return AIMessage(
            content=str(raw_resp),
            tool_calls=[],
            response_metadata={"model_name": self.model_name},
        )

    def _record_invocation(
        self, messages: list[BaseMessage] | list[dict[str, Any]], **kwargs: Any
    ) -> None:
        """Save call details to call_history."""
        serialized_msgs = []
        for m in messages:
            if isinstance(m, BaseMessage):
                serialized_msgs.append(
                    {"type": m.__class__.__name__, "content": m.content}
                )
            elif isinstance(m, dict):
                serialized_msgs.append(m)
            else:
                serialized_msgs.append({"type": "unknown", "content": str(m)})

        self.call_history.append(
            ChatCallRecord(
                messages=serialized_msgs,
                kwargs=kwargs,
            )
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._record_invocation(messages, stop=stop, **kwargs)
        raw_resp = self._get_next_raw_response(messages)
        ai_msg = self._normalize_to_ai_message(raw_resp)
        generation = ChatGeneration(message=ai_msg)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Re-use sync generation logic (non-blocking in-memory)
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._record_invocation(messages, stop=stop, **kwargs)
        raw_resp = self._get_next_raw_response(messages)
        ai_msg = self._normalize_to_ai_message(raw_resp)

        content = ai_msg.content or ""
        chunk_size = max(1, self._streaming_chunk_size)

        if not content and ai_msg.tool_calls:
            # Yield full tool call in single chunk
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_calls=ai_msg.tool_calls,
                )
            )
            return

        for i in range(0, len(content), chunk_size):
            token = content[i : i + chunk_size]
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> MockChatModel:
        """Bind tool definitions to the mock model and return a copy."""
        model_copy = copy.copy(self)
        model_copy.bound_tools = list(tools)
        return model_copy

    def with_structured_output(
        self,
        schema: type[BaseModel] | dict[str, Any] | type,
        **kwargs: Any,
    ) -> StructuredMockChatModel:
        """Return structured output runner wrapper."""
        return StructuredMockChatModel(mock_model=self, schema=schema, **kwargs)


class MockLLMFactory:
    """Builder factory for creating configured MockChatModel instances."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch | None = None):
        self._monkeypatch = monkeypatch
        self._created_instances: list[MockChatModel] = []

    def create(
        self,
        responses: list[Any] | None = None,
        *,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        auto_patch: bool = True,
        repeat_last: bool = True,
    ) -> MockChatModel:
        """
        Create a MockChatModel instance and optionally monkeypatch
        ``src.core.config.get_llm`` so all agent nodes use this mock.
        """
        model = MockChatModel(
            responses=responses or [],
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            repeat_last=repeat_last,
        )
        self._created_instances.append(model)

        if auto_patch and self._monkeypatch is not None:
            self._monkeypatch.setattr(
                "src.core.config.get_llm",
                lambda *args, **kwargs: model,
            )
            # Also patch direct module references if any
            for mod in [
                "src.agents.architect",
                "src.agents.backend_engineer",
                "src.agents.frontend_engineer",
                "src.agents.requirement_analyzer",
                "src.agents.reviewer",
                "src.agents.task_planner",
                "src.agents.tester",
                "src.agents.watchdog",
                "src.agents.memory",
            ]:
                try:
                    self._monkeypatch.setattr(
                        f"{mod}.get_llm", lambda *args, **kwargs: model
                    )
                except Exception:
                    pass

        return model


@pytest.fixture
def mock_llm_factory(monkeypatch: pytest.MonkeyPatch) -> MockLLMFactory:
    """Fixture providing a factory to create and patch MockChatModel instances."""
    return MockLLMFactory(monkeypatch=monkeypatch)


@pytest.fixture
def mock_llm(mock_llm_factory: MockLLMFactory) -> MockChatModel:
    """Pre-patched default MockChatModel fixture."""
    return mock_llm_factory.create()


# ==============================================================================
# 3. Tool Call Builders for ReAct & Agent Testing
# ==============================================================================


def make_tool_call(
    name: str,
    args: dict[str, Any],
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    """Create a standardized LangChain/OpenAI tool call dictionary."""
    return {
        "name": name,
        "args": args,
        "id": tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
        "type": "tool_call",
    }


def make_find_files_call(
    pattern: str = "*.py",
    search_dir: str = ".",
    max_depth: int | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Helper creating a FindFilesTool invocation dict."""
    args: dict[str, Any] = {"pattern": pattern, "search_dir": search_dir}
    if max_depth is not None:
        args["max_depth"] = max_depth
    if exclude_patterns is not None:
        args["exclude_patterns"] = exclude_patterns
    return make_tool_call("FindFilesTool", args)


def make_grep_call(
    query: str,
    path_pattern: str = "**/*",
    case_sensitive: bool = True,
    max_results: int = 50,
) -> dict[str, Any]:
    """Helper creating a GrepSearchTool invocation dict."""
    return make_tool_call(
        "GrepSearchTool",
        {
            "query": query,
            "path_pattern": path_pattern,
            "case_sensitive": case_sensitive,
            "max_results": max_results,
        },
    )


def make_ast_symbol_call(
    symbol_name: str,
    file_pattern: str = "**/*.py",
) -> dict[str, Any]:
    """Helper creating an ASTSymbolNavigator invocation dict."""
    return make_tool_call(
        "ASTSymbolNavigator",
        {
            "symbol_name": symbol_name,
            "file_pattern": file_pattern,
        },
    )


def make_view_file_call(
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
    show_line_numbers: bool = True,
) -> dict[str, Any]:
    """Helper creating a ViewFileTool invocation dict."""
    args: dict[str, Any] = {
        "file_path": file_path,
        "start_line": start_line,
        "show_line_numbers": show_line_numbers,
    }
    if end_line is not None:
        args["end_line"] = end_line
    return make_tool_call("ViewFileTool", args)


def make_replace_content_call(
    file_path: str,
    target_content: str,
    replacement_content: str,
    allow_multiple: bool = False,
) -> dict[str, Any]:
    """Helper creating a ReplaceContentTool invocation dict."""
    return make_tool_call(
        "ReplaceContentTool",
        {
            "file_path": file_path,
            "target_content": target_content,
            "replacement_content": replacement_content,
            "allow_multiple": allow_multiple,
        },
    )


def make_run_test_call(
    test_path: str = "tests/",
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Helper creating a RunTestTool invocation dict."""
    return make_tool_call(
        "RunTestTool",
        {
            "test_path": test_path,
            "extra_args": extra_args or [],
        },
    )


def make_run_linter_call(
    path: str = ".",
    linter: str = "ruff",
) -> dict[str, Any]:
    """Helper creating a RunLinterTool invocation dict."""
    return make_tool_call(
        "RunLinterTool",
        {
            "path": path,
            "linter": linter,
        },
    )


# ==============================================================================
# 4. Sandboxed Workspace & Helper
# ==============================================================================


class WorkspaceHelper:
    """
    Test helper providing ergonomic file system, git, and execution capabilities
    within an isolated temporary directory.
    """

    def __init__(self, workspace_path: Path):
        self.path = workspace_path.resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.str_path = str(self.path)

    def write_file(
        self,
        relative_path: str | Path,
        content: str,
        encoding: str = "utf-8",
    ) -> Path:
        """Write content to a file inside the workspace, creating parent directories."""
        clean_rel = str(relative_path).lstrip("/\\")
        target = (self.path / clean_rel).resolve()
        # Verify containment
        target.relative_to(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        return target

    def read_file(self, relative_path: str | Path, encoding: str = "utf-8") -> str:
        """Read content of a file inside the workspace."""
        clean_rel = str(relative_path).lstrip("/\\")
        target = (self.path / clean_rel).resolve()
        target.relative_to(self.path)
        return target.read_text(encoding=encoding)

    def exists(self, relative_path: str | Path) -> bool:
        """Check if file or directory exists inside the workspace."""
        try:
            clean_rel = str(relative_path).lstrip("/\\")
            target = (self.path / clean_rel).resolve()
            target.relative_to(self.path)
            return target.exists()
        except (ValueError, PermissionError):
            return False

    def delete_file(self, relative_path: str | Path) -> None:
        """Delete a file inside the workspace."""
        clean_rel = str(relative_path).lstrip("/\\")
        target = (self.path / clean_rel).resolve()
        target.relative_to(self.path)
        if target.is_file():
            target.unlink()

    def create_file_tree(self, files: dict[str, str]) -> None:
        """Create multiple files from a mapping of relative_path -> content."""
        for rel_path, content in files.items():
            self.write_file(rel_path, content)

    def list_files(
        self, relative_path: str = ".", recursive: bool = True
    ) -> list[str]:
        """List relative file paths inside the workspace."""
        base = (self.path / relative_path.lstrip("/\\")).resolve()
        base.relative_to(self.path)
        if not base.exists():
            return []
        iterator = base.rglob("*") if recursive else base.iterdir()
        return [
            item.relative_to(self.path).as_posix()
            for item in iterator
            if item.is_file()
        ]

    def init_git(
        self,
        default_branch: str = "main",
        user_name: str = "Test Developer",
        user_email: str = "test@ai-team.local",
    ) -> GitTracker:
        """Initialize a real git repository inside this workspace."""
        trace: list[dict[str, Any]] = []
        git = GitTracker(workspace_dir=self.path, trace=trace)
        git.init(default_branch=default_branch)

        # Set explicit local config
        subprocess.run(
            ["git", "config", "user.name", user_name],
            cwd=str(self.path),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", user_email],
            cwd=str(self.path),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "commit.gpgsign", "false"],
            cwd=str(self.path),
            check=True,
            capture_output=True,
        )
        return git

    def fs_manager(
        self, trace: list[dict[str, Any]] | None = None
    ) -> FileSystemManager:
        """Return a FileSystemManager instance attached to this workspace."""
        return FileSystemManager(
            workspace_dir=self.path, trace=[] if trace is None else trace
        )

    def git_tracker(self, trace: list[dict[str, Any]] | None = None) -> GitTracker:
        """Return a GitTracker instance attached to this workspace."""
        return GitTracker(
            workspace_dir=self.path, trace=[] if trace is None else trace
        )

    def subprocess_executor(
        self,
        trace: list[dict[str, Any]] | None = None,
        allowed_commands: Any = False,
        **kwargs: Any,
    ) -> SubprocessExecutor:
        """Return a SubprocessExecutor instance attached to this workspace."""
        return SubprocessExecutor(
            workspace_dir=self.path,
            trace=[] if trace is None else trace,
            allowed_commands=allowed_commands,
            **kwargs,
        )


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Generator[WorkspaceHelper, None, None]:
    """
    Pytest fixture yielding a clean, temporary sandboxed WorkspaceHelper.
    Cleans up all files automatically on teardown.
    """
    ws_dir = tmp_path / f"test_workspace_{uuid.uuid4().hex[:6]}"
    ws_dir.mkdir(parents=True, exist_ok=True)
    helper = WorkspaceHelper(ws_dir)
    yield helper
    shutil.rmtree(ws_dir, ignore_errors=True)


@pytest.fixture
def sample_workspace_files() -> dict[str, str]:
    """Mapping of standard starter files for Python project scaffolding."""
    return {
        ".gitkeep": "# Workspace Root\n",
        "README.md": "# Sample Microservice\n\nFastAPI backend service.\n",
        "pyproject.toml": (
            "[project]\n"
            'name = "sample-service"\n'
            'version = "0.1.0"\n'
            'requires-python = ">=3.11"\n'
        ),
        "src/__init__.py": '"""Source package."""\n',
        "src/calculator.py": (
            "def add(a: int, b: int) -> int:\n"
            '    """Add two numbers."""\n'
            "    return a + b\n\n"
            "def divide(a: float, b: float) -> float:\n"
            '    """Divide two numbers with zero check."""\n'
            "    if b == 0:\n"
            '        raise ValueError("Cannot divide by zero")\n'
            "    return a / b\n"
        ),
        "tests/__init__.py": "",
        "tests/test_calculator.py": (
            "from src.calculator import add, divide\n"
            "import pytest\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n"
            "    assert add(-1, 1) == 0\n\n"
            "def test_divide():\n"
            "    assert divide(10, 2) == 5.0\n"
            "    with pytest.raises(ValueError):\n"
            "        divide(5, 0)\n"
        ),
    }


@pytest.fixture
def populated_workspace(
    temp_workspace: WorkspaceHelper, sample_workspace_files: dict[str, str]
) -> WorkspaceHelper:
    """Fixture yielding a WorkspaceHelper pre-populated with standard project files."""
    temp_workspace.create_file_tree(sample_workspace_files)
    return temp_workspace


# ==============================================================================
# 5. Isolated Git Repository Fixture
# ==============================================================================


@pytest.fixture
def isolated_git_repo(
    populated_workspace: WorkspaceHelper,
) -> Generator[WorkspaceHelper, None, None]:
    """
    Fixture initializing a real, isolated Git repository with:
    - Default branch ``main``
    - Initial commit containing sample starter files
    - Auxiliary branches ``develop`` and ``feature/test-branch``
    - Clean HEAD state on branch ``main``
    """
    git = populated_workspace.init_git(default_branch="main")
    git.stage_all()
    git.commit(message="chore: initial repository scaffold", allow_empty=False)

    # Create develop and feature branch
    git.ensure_branch("develop")
    git.ensure_branch("feature/test-branch")

    # Switch back to main
    git.ensure_branch("main")

    yield populated_workspace


@pytest.fixture
def git_tracker_factory() -> (
    Callable[[str | Path, list[dict[str, Any]] | None], GitTracker]
):
    """Factory fixture to create GitTracker instances."""

    def _factory(
        workspace_dir: str | Path,
        trace: list[dict[str, Any]] | None = None,
    ) -> GitTracker:
        return GitTracker(
            workspace_dir=workspace_dir, trace=[] if trace is None else trace
        )

    return _factory


# ==============================================================================
# 6. Mock Checkpointer for LangGraph State Persistence
# ==============================================================================


class MockCheckpointer:
    """
    Production-grade in-memory state checkpointer compatible with LangGraph.

    Capabilities:
    - Thread-isolated state snapshot storage.
    - Checkpoint saving, retrieval, listing, and write tracking.
    - Crash simulation (`crash_on_put`, `crash_on_get`).
    - State corruption simulation for resilience / recovery testing.
    - Full telemetry and audit inspection counters.
    """

    def __init__(self) -> None:
        # Storage: thread_id -> list of CheckpointRecord dicts
        self._storage: dict[str, list[dict[str, Any]]] = {}
        self._writes_storage: dict[str, list[dict[str, Any]]] = {}
        self.put_count: int = 0
        self.get_count: int = 0
        self.list_count: int = 0
        self.crash_put_counter: int = 0
        self.crash_get_counter: int = 0
        self.simulated_delay_s: float = 0.0

    def simulate_crash_on_put(self, n_times: int = 1) -> None:
        """Cause the next *n_times* `put()` calls to raise CheckpointerCrashError."""
        self.crash_put_counter = n_times

    def simulate_crash_on_get(self, n_times: int = 1) -> None:
        """Cause the next *n_times* `get()` calls to raise CheckpointerCrashError."""
        self.crash_get_counter = n_times

    def set_delay(self, seconds: float) -> None:
        """Inject artificial latency into checkpointer operations."""
        self.simulated_delay_s = max(0.0, seconds)

    def corrupt_state(
        self,
        thread_id: str,
        mutation_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Corrupt the latest stored checkpoint for a thread to test recovery."""
        records = self._storage.get(thread_id, [])
        if not records:
            return
        latest = records[-1]
        if mutation_fn:
            latest["checkpoint"]["channel_values"] = mutation_fn(
                latest["checkpoint"].get("channel_values", {})
            )
        else:
            # Default corruption: wipe task queue and inject invalid status
            latest["checkpoint"]["channel_values"]["status"] = "corrupted"
            latest["checkpoint"]["channel_values"]["task_queue"] = None

    def clear(self) -> None:
        """Reset all stored checkpoints and counters."""
        self._storage.clear()
        self._writes_storage.clear()
        self.put_count = 0
        self.get_count = 0
        self.list_count = 0
        self.crash_put_counter = 0
        self.crash_get_counter = 0

    def put(
        self,
        config: dict[str, Any] | RunnableConfig,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save a checkpoint snapshot for the given thread_id."""
        if self.simulated_delay_s > 0:
            time.sleep(self.simulated_delay_s)

        if self.crash_put_counter > 0:
            self.crash_put_counter -= 1
            raise CheckpointerCrashError(
                "Checkpointer failed to commit write transaction: database connection dropped."
            )

        self.put_count += 1
        conf_dict = config if isinstance(config, dict) else dict(config)
        configurable = conf_dict.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")
        checkpoint_id = configurable.get(
            "checkpoint_id", str(uuid.uuid4())
        ) or str(uuid.uuid4())

        record = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "checkpoint": copy.deepcopy(checkpoint),
            "metadata": copy.deepcopy(metadata or {}),
            "new_versions": copy.deepcopy(new_versions or {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if thread_id not in self._storage:
            self._storage[thread_id] = []
        self._storage[thread_id].append(record)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def get_tuple(
        self, config: dict[str, Any] | RunnableConfig
    ) -> Any | None:
        """Retrieve the latest checkpoint tuple for the configured thread_id."""
        if self.simulated_delay_s > 0:
            time.sleep(self.simulated_delay_s)

        if self.crash_get_counter > 0:
            self.crash_get_counter -= 1
            raise CheckpointerCrashError(
                "Checkpointer failed to read snapshot: connection timeout."
            )

        self.get_count += 1
        conf_dict = config if isinstance(config, dict) else dict(config)
        configurable = conf_dict.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")
        checkpoint_id = configurable.get("checkpoint_id")

        records = self._storage.get(thread_id, [])
        if not records:
            return None

        if checkpoint_id:
            for r in reversed(records):
                if r["checkpoint_id"] == checkpoint_id:
                    return self._build_tuple(r)
            return None

        # Return latest
        return self._build_tuple(records[-1])

    def _build_tuple(self, record: dict[str, Any]) -> Any:
        """Create a CheckpointTuple-like object with attributes."""

        class _SimpleTuple:

            def __init__(self, rec: dict[str, Any]):
                self.config = {
                    "configurable": {
                        "thread_id": rec["thread_id"],
                        "checkpoint_id": rec["checkpoint_id"],
                    }
                }
                self.checkpoint = rec["checkpoint"]
                self.metadata = rec["metadata"]
                self.parent_config = None
                self.pending_writes = []

        return _SimpleTuple(record)

    def list(
        self,
        config: dict[str, Any] | RunnableConfig | None = None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[Any]:
        """List checkpoints for a thread in reverse chronological order."""
        self.list_count += 1
        thread_id = "default"
        if config:
            conf_dict = config if isinstance(config, dict) else dict(config)
            thread_id = conf_dict.get("configurable", {}).get(
                "thread_id", "default"
            )

        records = list(reversed(self._storage.get(thread_id, [])))
        if limit:
            records = records[:limit]

        for r in records:
            yield self._build_tuple(r)

    # Async compatibility wrappers
    async def aput(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.put(*args, **kwargs)

    async def aget_tuple(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_tuple(*args, **kwargs)

    async def alist(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        for item in self.list(*args, **kwargs):
            yield item

    def get_latest_state(self, thread_id: str) -> dict[str, Any] | None:
        """Convenience method returning the channel_values dictionary of latest state."""
        records = self._storage.get(thread_id, [])
        if not records:
            return None
        return records[-1]["checkpoint"].get("channel_values", {})


@pytest.fixture
def mock_checkpointer() -> MockCheckpointer:
    """Fixture yielding a fresh in-memory MockCheckpointer instance."""
    return MockCheckpointer()


@pytest.fixture
def file_checkpointer(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Fixture yielding a temporary directory configured for file-based
    state checkpoint testing (verifying state persistence across crashes).
    """
    storage_dir = tmp_path / "checkpointer_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    yield storage_dir
    shutil.rmtree(storage_dir, ignore_errors=True)


# ==============================================================================
# 7. FastAPI Async & Sync HTTP Test Clients
# ==============================================================================


class APIHelper:
    """Helper utilities for interacting with FastAPI endpoints during testing."""

    def __init__(self, async_client: httpx.AsyncClient):
        self.client = async_client

    async def run_pipeline(
        self,
        requirements: str,
        project_name: str = "Test Suite Project",
        max_retries: int = 3,
    ) -> httpx.Response:
        """Trigger POST /api/v1/projects/run."""
        payload = {
            "requirements": requirements,
            "project_name": project_name,
            "max_retries": max_retries,
        }
        return await self.client.post("/api/v1/projects/run", json=payload)

    async def get_status(self, project_id: str) -> httpx.Response:
        """Trigger GET /api/v1/projects/{project_id}/status."""
        return await self.client.get(f"/api/v1/projects/{project_id}/status")

    async def resume(
        self, project_id: str, feedback: str = "Proceed"
    ) -> httpx.Response:
        """Trigger POST /api/v1/projects/{project_id}/resume."""
        payload = {"feedback": feedback}
        return await self.client.post(
            f"/api/v1/projects/{project_id}/resume", json=payload
        )

    async def check_health(self) -> httpx.Response:
        """Trigger GET /api/v1/health."""
        return await self.client.get("/api/v1/health")


@pytest.fixture
async def async_test_client() -> AsyncIterator[httpx.AsyncClient]:
    """
    Async HTTP client fixture configured with ASGI transport against the FastAPI app.
    Properly runs startup and shutdown lifespans.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client


@pytest.fixture
def sync_test_client() -> Generator[TestClient, None, None]:
    """
    Synchronous FastAPI TestClient fixture for standard REST endpoint and
    WebSocket integration tests.
    """
    with TestClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def api_helper(async_test_client: httpx.AsyncClient) -> APIHelper:
    """Fixture providing APIHelper instance."""
    return APIHelper(async_test_client)


# ==============================================================================
# 8. Domain Mock Data Factories (ProjectState, TaskItem, CodeArtifact, ADRs)
# ==============================================================================


def project_state_factory(
    *,
    project_id: str | None = None,
    project_name: str = "Test Microservice",
    requirements: str = "Build a FastAPI CRUD microservice with SQLite backend.",
    status: str = "initialized",
    current_phase: str = "planning",
    iteration: int = 0,
    max_retries: int = 3,
    workspace_dir: str = "",
    active_branch: str = "main",
    technical_specifications: dict[str, Any] | None = None,
    architecture_decisions: list[dict[str, Any]] | None = None,
    project_structure: dict[str, Any] | None = None,
    task_queue: list[dict[str, Any]] | None = None,
    completed_tasks: list[dict[str, Any]] | None = None,
    code_artifacts: list[dict[str, Any]] | None = None,
    execution_trace: list[dict[str, Any]] | None = None,
    retry_counts: dict[str, int] | None = None,
    error_log: list[dict[str, Any]] | None = None,
) -> ProjectState:
    """Generate a fully populated, valid ProjectState TypedDict."""
    pid = project_id or f"proj-{uuid.uuid4().hex[:8]}"
    return {
        "project_id": pid,
        "project_name": project_name,
        "requirements": requirements,
        "technical_specifications": technical_specifications or {},
        "architecture_decisions": architecture_decisions or [],
        "project_structure": project_structure or {},
        "task_queue": task_queue or [],
        "completed_tasks": completed_tasks or [],
        "code_artifacts": code_artifacts or [],
        "workspace_dir": workspace_dir,
        "active_branch": active_branch,
        "execution_trace": execution_trace or [],
        "retry_counts": retry_counts or {},
        "error_log": error_log or [],
        "current_phase": current_phase,
        "iteration": iteration,
        "max_retries": max_retries,
        "status": status,
    }


def task_item_factory(
    *,
    task_id: str = "TASK-001",
    title: str = "Implement API endpoints",
    description: str = "Create CRUD routes in src/api/routes.py",
    assigned_to: str = "backend_engineer",
    priority: int = 1,
    status: str = "pending",
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskItem:
    """Generate a valid TaskItem Pydantic model."""
    meta = metadata or {}
    if "file_path" not in meta:
        meta["file_path"] = "src/api/routes.py"
    return TaskItem(
        task_id=task_id,
        title=title,
        description=description,
        assigned_to=assigned_to,
        priority=priority,
        status=status,
        dependencies=dependencies or [],
        metadata=meta,
    )


def task_item_dict_factory(**kwargs: Any) -> dict[str, Any]:
    """Generate a task item dictionary."""
    return task_item_factory(**kwargs).model_dump()


def code_artifact_factory(
    *,
    file_path: str = "src/api/routes.py",
    language: str = "python",
    content: str = (
        "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n"
        "@router.get('/health')\ndef health():\n    return {'status': 'ok'}\n"
    ),
    version: int = 1,
    tests_passed: bool | None = None,
    created_at: str | None = None,
) -> CodeArtifact:
    """Generate a valid CodeArtifact Pydantic model."""
    return CodeArtifact(
        file_path=file_path,
        language=language,
        content=content,
        version=version,
        tests_passed=tests_passed,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


def code_artifact_dict_factory(**kwargs: Any) -> dict[str, Any]:
    """Generate a code artifact dictionary."""
    return code_artifact_factory(**kwargs).model_dump()


def adr_factory(
    *,
    decision_id: str = "ADR-001",
    title: str = "Use SQLite for Embedded Local Persistence",
    context: str = "Requires lightweight zero-dependency local DB.",
    decision: str = "Adopt SQLite with standard sqlite3 driver.",
    consequences: str = "Simplified testing, single-file storage.",
    status: str = "accepted",
) -> ArchitectureDecision:
    """Generate a valid ArchitectureDecision Pydantic model."""
    return ArchitectureDecision(
        decision_id=decision_id,
        title=title,
        context=context,
        decision=decision,
        consequences=consequences,
        status=status,
    )


def adr_dict_factory(**kwargs: Any) -> dict[str, Any]:
    """Generate an ArchitectureDecision dictionary."""
    return adr_factory(**kwargs).model_dump()


def trace_record_factory(
    tool: str = "FileSystemManager",
    operation: str = "write_file",
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    success: bool = True,
    error: str | None = None,
    duration_ms: float = 12.5,
) -> dict[str, Any]:
    """Generate a standardized execution trace entry."""
    return _make_trace(
        tool=tool,
        operation=operation,
        inputs=inputs or {"relative_path": "src/main.py"},
        outputs=outputs or {"bytes_written": 128},
        success=success,
        error=error,
        duration_ms=duration_ms,
    )


@pytest.fixture
def sample_project_state(temp_workspace: WorkspaceHelper) -> ProjectState:
    """Fixture returning an initial valid ProjectState bound to temp_workspace."""
    return project_state_factory(workspace_dir=temp_workspace.str_path)


@pytest.fixture
def sample_running_state(temp_workspace: WorkspaceHelper) -> ProjectState:
    """Fixture returning a running ProjectState with planned tasks."""
    tasks = [
        task_item_dict_factory(
            task_id="TASK-001",
            title="Create Data Model",
            metadata={"file_path": "src/models.py"},
        ),
        task_item_dict_factory(
            task_id="TASK-002",
            title="Create API Endpoints",
            metadata={"file_path": "src/api.py"},
            dependencies=["TASK-001"],
        ),
    ]
    adrs = [adr_dict_factory(decision_id="ADR-001")]
    artifacts = [
        code_artifact_dict_factory(
            file_path="src/models.py", content="class Item: pass\n"
        )
    ]
    return project_state_factory(
        workspace_dir=temp_workspace.str_path,
        status="running",
        current_phase="development",
        iteration=1,
        task_queue=tasks,
        architecture_decisions=adrs,
        code_artifacts=artifacts,
    )


@pytest.fixture
def sample_failed_state(temp_workspace: WorkspaceHelper) -> ProjectState:
    """Fixture returning a failed ProjectState with error logs."""
    return project_state_factory(
        workspace_dir=temp_workspace.str_path,
        status="failed",
        current_phase="testing",
        iteration=3,
        retry_counts={"tester": 3},
        error_log=[
            {
                "node_name": "tester",
                "error_type": "PytestFailure",
                "error_message": "2 tests failed in tests/test_calculator.py",
                "attempt": 3,
                "resolved": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )


@pytest.fixture
def sample_blocked_state(temp_workspace: WorkspaceHelper) -> ProjectState:
    """Fixture returning a blocked ProjectState awaiting human intervention."""
    return project_state_factory(
        workspace_dir=temp_workspace.str_path,
        status="blocked",
        current_phase="review",
        iteration=4,
        retry_counts={"task_fail_TASK-001": 3},
    )


@pytest.fixture
def sample_task_item() -> TaskItem:
    """Fixture returning a single sample TaskItem."""
    return task_item_factory()


@pytest.fixture
def sample_task_list() -> list[dict[str, Any]]:
    """Fixture returning a sequence of 3 dependent task items as dicts."""
    return [
        task_item_dict_factory(
            task_id="TASK-001",
            title="Database Schema",
            priority=0,
            metadata={"file_path": "src/db.py"},
        ),
        task_item_dict_factory(
            task_id="TASK-002",
            title="Service Layer",
            priority=1,
            dependencies=["TASK-001"],
            metadata={"file_path": "src/service.py"},
        ),
        task_item_dict_factory(
            task_id="TASK-003",
            title="REST Routes",
            priority=2,
            dependencies=["TASK-002"],
            metadata={"file_path": "src/routes.py"},
        ),
    ]


@pytest.fixture
def sample_code_artifact() -> CodeArtifact:
    """Fixture returning a sample CodeArtifact."""
    return code_artifact_factory()


@pytest.fixture
def sample_adr() -> ArchitectureDecision:
    """Fixture returning a sample ArchitectureDecision."""
    return adr_factory()


@pytest.fixture
def sample_trace() -> list[dict[str, Any]]:
    """Fixture returning a list of diverse execution trace records."""
    return [
        trace_record_factory(
            tool="FileSystemManager",
            operation="write_file",
            inputs={"relative_path": "src/main.py"},
        ),
        trace_record_factory(
            tool="GitTracker",
            operation="commit",
            inputs={"message": "feat: initial"},
            outputs={"exit_code": 0},
        ),
        trace_record_factory(
            tool="SubprocessExecutor",
            operation="run:pytest",
            inputs={"command": "pytest tests/"},
            outputs={"exit_code": 0, "stdout": "3 passed"},
        ),
    ]


# ==============================================================================
# 9. Programmable SubprocessExecutor Test Double
# ==============================================================================


class MockSubprocessExecutor:
    """
    Mock test double for SubprocessExecutor supporting programmable execution results,
    custom exit codes, stdout/stderr strings, and timeout simulations.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        trace: list[dict[str, Any]] | None = None,
        default_success: bool = True,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.trace = trace if trace is not None else []
        self.default_success = default_success
        # Mapping: command pattern/prefix -> ExecutionResult or callable
        self._rules: list[tuple[str, Union[ExecutionResult, Callable[..., ExecutionResult]]]] = []
        self.executed_commands: list[list[str]] = []

    def set_result(
        self,
        command_substr: str,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        duration_ms: float = 10.0,
    ) -> None:
        """Register a canned ExecutionResult for any command matching command_substr."""
        res = ExecutionResult(
            command=[command_substr],
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )
        self._rules.insert(0, (command_substr, res))

    def set_timeout(self, command_substr: str, timeout_s: float = 30.0) -> None:
        """Simulate a command timeout."""
        res = ExecutionResult(
            command=[command_substr],
            exit_code=-1,
            stdout="",
            stderr=f"[TIMEOUT] Process killed after {timeout_s}s",
            timed_out=True,
            duration_ms=timeout_s * 1000,
        )
        self._rules.insert(0, (command_substr, res))

    def _match_result(self, command: list[str]) -> ExecutionResult:
        self.executed_commands.append(command)
        cmd_str = " ".join(command)

        for pattern, res_or_fn in self._rules:
            if pattern in cmd_str or any(pattern in arg for arg in command):
                if callable(res_or_fn):
                    return res_or_fn(command)
                return res_or_fn

        # Default fallback
        exit_code = 0 if self.default_success else 1
        stdout = "mock execution succeeded" if self.default_success else ""
        stderr = "" if self.default_success else "mock execution failed"
        return ExecutionResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            duration_ms=5.0,
        )

    def run_sync(
        self, command: list[str], timeout: float | None = None, **kwargs: Any
    ) -> ExecutionResult:
        """Synchronous execution wrapper."""
        result = self._match_result(command)
        self.trace.append(
            _make_trace(
                tool="SubprocessExecutor",
                operation=f"run:{Path(command[0]).name if command else 'exec'}",
                inputs={"command": " ".join(command), "cwd": str(self.workspace_dir)},
                outputs=result.as_dict(),
                success=result.success,
                error=result.stderr if not result.success else None,
                duration_ms=result.duration_ms,
            )
        )
        return result

    async def run_async(
        self, command: list[str], timeout: float | None = None, **kwargs: Any
    ) -> ExecutionResult:
        """Async execution wrapper."""
        return self.run_sync(command, timeout=timeout, **kwargs)

    def run_pytest(
        self, test_path: str = "tests/", **kwargs: Any
    ) -> ExecutionResult:
        """Simulate pytest execution."""
        return self.run_sync([sys.executable, "-m", "pytest", test_path])

    def run_linter(
        self, path: str = ".", linter: str = "ruff", **kwargs: Any
    ) -> ExecutionResult:
        """Simulate linter execution."""
        return self.run_sync([sys.executable, "-m", linter, path])


@pytest.fixture
def mock_subprocess_executor_factory() -> (
    Callable[[str | Path, list[dict[str, Any]] | None], MockSubprocessExecutor]
):
    """Factory fixture for creating MockSubprocessExecutor instances."""

    def _factory(
        workspace_dir: str | Path,
        trace: list[dict[str, Any]] | None = None,
        default_success: bool = True,
    ) -> MockSubprocessExecutor:
        return MockSubprocessExecutor(
            workspace_dir=workspace_dir,
            trace=trace,
            default_success=default_success,
        )

    return _factory


@pytest.fixture
def mock_subprocess_executor(
    temp_workspace: WorkspaceHelper,
) -> MockSubprocessExecutor:
    """Fixture providing a default MockSubprocessExecutor bound to temp_workspace."""
    return MockSubprocessExecutor(workspace_dir=temp_workspace.path)


@pytest.fixture
def mock_fs_manager(temp_workspace: WorkspaceHelper) -> FileSystemManager:
    """Fixture providing a real FileSystemManager bound to temp_workspace."""
    return temp_workspace.fs_manager()


@pytest.fixture
def mock_git_tracker(isolated_git_repo: WorkspaceHelper) -> GitTracker:
    """Fixture providing a real GitTracker bound to isolated_git_repo."""
    return isolated_git_repo.git_tracker()
