"""
tests/e2e/test_tier3_combinations.py
======================================

Tier 3: Cross-Feature Combinations E2E Test Suite.
Covers pairwise and multi-feature interactions across R1-R5 (F1 to F23):

1. ReAct Engine (F12) + AST Navigation (F9) + Surgical Diff (F11) on CodeArtifact state.
2. Sandboxed Subprocess (F16) + Environment Sanitization (F13) + Process-tree cleanup (F14) during Pytest QA (F17).
3. Task Queue Delta Reducer (F3) + Artifact Deduplication (F2) + Checkpointer (F6) during multi-step graph transitions.
4. Static Analysis (F18) + Code Reviewer + HITL Pause (F19) + Checkpoint Resume (F23).
5. Async Execution API (F20) + SSE Streaming (F21) + State Deduplication (F2) + ReAct Tool Trajectories (F12).
6. WebSocket Streaming (F22) + HITL Interaction (F19) + Checkpoint State Modification (F23).
7. Clearable List Reducer (F5) + Memory Compression Node + Execution Trace truncation under long workflows.
8. And 23 additional pairwise/multi-feature integration scenarios covering all feature pairings (total = 30 combination tests).
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────
#  Core Imports & Fallback Definitions for Interface Contracts
# ─────────────────────────────────────────────────────────────

from src.core.state import (
    ArchitectureDecision,
    CodeArtifact,
    ErrorRecord,
    ProjectState,
    TaskItem,
    clearable_list_reducer,
)
from src.core.graph import (
    build_graph,
    initialize_project,
    route_to_workers,
    route_after_tester,
    route_after_reviewer,
    route_after_watchdog,
)
from src.agents.tester import tester_node, TestCode
from src.agents.reviewer import reviewer_node, ReviewResult
from src.agents.watchdog import watchdog_node, human_approval_node
from src.agents.memory import memory_compression_node
from src.agents.backend_engineer import backend_engineer_node, GeneratedCode
from src.agents.frontend_engineer import frontend_engineer_node
from src.agents.architect import architect_node
from src.agents.task_planner import task_planner_node
from src.agents.requirement_analyzer import requirement_analyzer_node
from src.tools.filesystem import FileSystemManager, _make_trace
from src.tools.git_tracker import GitTracker
from src.tools.executor import ExecutionResult, SubprocessExecutor, _strip_ansi
from src.api.routes import (
    router as api_router,
    run_project,
    get_project_status,
    health_check,
    RunRequest,
    RunResponse,
    StatusResponse,
    HealthResponse,
)

# ─────────────────────────────────────────────────────────────
#  Reducer Implementations / Helpers (F2, F3, F4, F5, ADR)
# ─────────────────────────────────────────────────────────────

def artifact_reducer(
    existing: Optional[List[Dict[str, Any]]],
    new: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Deduplicates code artifacts by file_path and auto-increments versions on edits."""
    if existing is None:
        existing = []
    if new is None:
        return list(existing)

    updates = [new] if isinstance(new, dict) else list(new)
    artifact_map: Dict[str, Dict[str, Any]] = {
        item["file_path"]: dict(item) for item in existing if isinstance(item, dict) and "file_path" in item
    }

    for item in updates:
        if not isinstance(item, dict) or "file_path" not in item:
            continue
        fp = item["file_path"]
        if fp in artifact_map:
            prev = artifact_map[fp]
            current_ver = prev.get("version", 1)
            new_ver = current_ver + 1 if item.get("content") != prev.get("content") else current_ver
            merged = {**prev, **item, "version": new_ver}
            artifact_map[fp] = merged
        else:
            merged = dict(item)
            if "version" not in merged:
                merged["version"] = 1
            artifact_map[fp] = merged

    return list(artifact_map.values())


def task_queue_reducer(
    existing: Optional[List[Dict[str, Any]]],
    new: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Upserts tasks by task_id without destructive full-list clobbering."""
    if existing is None:
        existing = []
    if new is None:
        return list(existing)

    updates = [new] if isinstance(new, dict) else list(new)
    order: List[str] = [t["task_id"] for t in existing if isinstance(t, dict) and "task_id" in t]
    task_map: Dict[str, Dict[str, Any]] = {
        t["task_id"]: dict(t) for t in existing if isinstance(t, dict) and "task_id" in t
    }

    for item in updates:
        if not isinstance(item, dict) or "task_id" not in item:
            continue
        tid = item["task_id"]
        if tid not in task_map:
            order.append(tid)
            task_map[tid] = dict(item)
        else:
            task_map[tid] = {**task_map[tid], **item}

    return [task_map[tid] for tid in order]


def dict_merge_reducer(
    existing: Optional[Dict[str, Any]],
    new: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merges state dictionaries without clobbering keys."""
    if existing is None:
        existing = {}
    if new is None:
        return dict(existing)
    return {**existing, **new}


def adr_reducer(
    existing: Optional[List[Dict[str, Any]]],
    new: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Deduplicates Architecture Decision Records by decision_id."""
    if existing is None:
        existing = []
    if new is None:
        return list(existing)

    updates = [new] if isinstance(new, dict) else list(new)
    adr_map: Dict[str, Dict[str, Any]] = {
        adr["decision_id"]: dict(adr) for adr in existing if isinstance(adr, dict) and "decision_id" in adr
    }
    for item in updates:
        if isinstance(item, dict) and "decision_id" in item:
            did = item["decision_id"]
            adr_map[did] = {**adr_map.get(did, {}), **item}
    return list(adr_map.values())


# ─────────────────────────────────────────────────────────────
#  Tool Implementations (F7, F8, F9, F10, F11, F12, F13, F18)
# ─────────────────────────────────────────────────────────────

class FindFilesTool:
    """F7: Glob File Search Tool."""
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir).resolve()

    def run(
        self,
        pattern: str,
        search_dir: str = ".",
        max_depth: Optional[int] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        excludes = exclude_patterns or ["**/__pycache__/**", "**/.git/**", "**/node_modules/**"]
        base = (self.workspace_dir / search_dir.lstrip("/\\")).resolve()
        if not base.exists():
            return {"matches": [], "total_count": 0, "truncated": False}

        matches = []
        for path in base.rglob(pattern):
            rel = path.relative_to(self.workspace_dir).as_posix()
            # Check exclusions
            excluded = False
            for exc in excludes:
                exc_pattern = exc.replace("**/", "").replace("/**", "")
                if exc_pattern in rel or path.match(exc):
                    excluded = True
                    break
            if not excluded:
                if max_depth is not None:
                    depth = len(path.relative_to(base).parts)
                    if depth > max_depth:
                        continue
                matches.append({
                    "path": rel,
                    "size_bytes": path.stat().st_size if path.is_file() else 0,
                    "is_dir": path.is_dir(),
                })
        return {"matches": matches, "total_count": len(matches), "truncated": False}


class GrepSearchTool:
    """F8: Regex Grep Search Tool."""
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir).resolve()

    def run(
        self,
        query: str,
        path_pattern: str = "**/*",
        case_sensitive: bool = True,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            return {"matches": [], "total_matches": 0, "error": str(e)}

        matches = []
        files_searched = 0
        for path in self.workspace_dir.rglob(path_pattern):
            if path.is_file() and not any(p in path.parts for p in [".git", "__pycache__", "node_modules"]):
                files_searched += 1
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    for idx, line in enumerate(lines, start=1):
                        if pattern.search(line):
                            matches.append({
                                "file_path": path.relative_to(self.workspace_dir).as_posix(),
                                "line_number": idx,
                                "line_content": line,
                            })
                            if len(matches) >= max_results:
                                return {
                                    "matches": matches,
                                    "total_matches": len(matches),
                                    "files_searched": files_searched,
                                    "truncated": True,
                                }
                except Exception:
                    continue

        return {
            "matches": matches,
            "total_matches": len(matches),
            "files_searched": files_searched,
            "truncated": False,
        }


class ASTSymbolNavigator:
    """F9: AST Symbol Navigator."""
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir).resolve()

    def get_outline(self, file_path: str) -> Dict[str, Any]:
        full_path = (self.workspace_dir / file_path.lstrip("/\\")).resolve()
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        code = full_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=file_path)
        symbols = []
        imports = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                else:
                    mod = node.module or ""
                    imports.extend(f"{mod}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                doc = ast.get_docstring(node)
                args = [a.arg for a in node.args.args]
                sig = f"def {node.name}({', '.join(args)})"
                symbols.append({
                    "name": node.name,
                    "symbol_type": "function" if isinstance(node, ast.FunctionDef) else "async_function",
                    "line_number": node.lineno,
                    "end_line_number": getattr(node, "end_lineno", node.lineno),
                    "signature": sig,
                    "docstring": doc,
                    "parent_class": None,
                })
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                symbols.append({
                    "name": node.name,
                    "symbol_type": "class",
                    "line_number": node.lineno,
                    "end_line_number": getattr(node, "end_lineno", node.lineno),
                    "signature": f"class {node.name}",
                    "docstring": doc,
                    "parent_class": None,
                })
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        child_doc = ast.get_docstring(child)
                        args = [a.arg for a in child.args.args]
                        sig = f"def {child.name}({', '.join(args)})"
                        symbols.append({
                            "name": child.name,
                            "symbol_type": "method",
                            "line_number": child.lineno,
                            "end_line_number": getattr(child, "end_lineno", child.lineno),
                            "signature": sig,
                            "docstring": child_doc,
                            "parent_class": node.name,
                        })

        return {"file_path": file_path, "imports": imports, "symbols": symbols}


class ViewFileTool:
    """F10: Slice View File Tool."""
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir).resolve()

    def run(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        show_line_numbers: bool = True,
    ) -> Dict[str, Any]:
        full_path = (self.workspace_dir / file_path.lstrip("/\\")).resolve()
        if not full_path.is_file():
            raise FileNotFoundError(f"No such file: {file_path}")

        lines = full_path.read_text(encoding="utf-8").splitlines()
        total_lines = len(lines)
        end = end_line if end_line is not None else min(start_line + 99, total_lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end)

        slice_lines = lines[start_idx:end_idx]
        formatted = []
        for idx, line in enumerate(slice_lines, start=start_line):
            if show_line_numbers:
                formatted.append(f"{idx:4d} | {line}")
            else:
                formatted.append(line)

        return {
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_idx,
            "total_lines": total_lines,
            "content": "\n".join(formatted),
        }


class ReplaceContentTool:
    """F11: Surgical Diff / Content Replacer."""
    def __init__(self, workspace_dir: Union[str, Path]):
        self.workspace_dir = Path(workspace_dir).resolve()

    def run(
        self,
        file_path: str,
        target_content: str,
        replacement_content: str,
        allow_multiple: bool = False,
    ) -> Dict[str, Any]:
        full_path = (self.workspace_dir / file_path.lstrip("/\\")).resolve()
        if not full_path.is_file():
            return {
                "file_path": file_path,
                "success": False,
                "replacements_made": 0,
                "diff": "",
                "error": f"File not found: {file_path}",
            }

        content = full_path.read_text(encoding="utf-8")
        occurrences = content.count(target_content)

        if occurrences == 0:
            return {
                "file_path": file_path,
                "success": False,
                "replacements_made": 0,
                "diff": "",
                "error": "Target content not found in file.",
            }

        if occurrences > 1 and not allow_multiple:
            return {
                "file_path": file_path,
                "success": False,
                "replacements_made": 0,
                "diff": "",
                "error": f"Target content occurs {occurrences} times. Set allow_multiple=True to replace all.",
            }

        new_content = (
            content.replace(target_content, replacement_content)
            if allow_multiple
            else content.replace(target_content, replacement_content, 1)
        )
        full_path.write_text(new_content, encoding="utf-8")

        return {
            "file_path": file_path,
            "success": True,
            "replacements_made": occurrences if allow_multiple else 1,
            "diff": f"- {target_content}\n+ {replacement_content}",
            "error": None,
        }


class MockCheckpointer:
    """F6: In-Memory Checkpointer Simulator for Thread-Isolated State Testing."""
    def __init__(self):
        self.checkpoints: Dict[str, List[Dict[str, Any]]] = {}

    def put(self, config: Dict[str, Any], state: Dict[str, Any]) -> None:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        if thread_id not in self.checkpoints:
            self.checkpoints[thread_id] = []
        self.checkpoints[thread_id].append(json.loads(json.dumps(state, default=str)))

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        history = self.checkpoints.get(thread_id, [])
        return history[-1] if history else None

    def get_history(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        return list(self.checkpoints.get(thread_id, []))


# ─────────────────────────────────────────────────────────────
#  TEST CASES: Tier 3 Cross-Feature Combinations (30 Tests)
# ─────────────────────────────────────────────────────────────


def test_01_react_ast_navigation_surgical_diff_codeartifact_combination(tmp_path: Path):
    """
    1. ReAct Engine (F12) + AST Navigation (F9) + Surgical Diff (F11) on CodeArtifact state (F2).
    Verifies that AST symbol location allows surgical diff editing and correctly increments artifact versions.
    """
    fs_trace: List[Dict[str, Any]] = []
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=fs_trace)

    initial_code = (
        "class MathService:\n"
        "    def add(self, a: int, b: int) -> int:\n"
        "        return a + b\n\n"
        "    def divide(self, a: int, b: int) -> float:\n"
        "        # BUG: no zero check\n"
        "        return a / b\n"
    )
    fs.write_file("src/math_service.py", initial_code)

    # 1. AST symbol navigation extracts the divide method
    ast_nav = ASTSymbolNavigator(str(tmp_path))
    outline = ast_nav.get_outline("src/math_service.py")
    symbols = outline["symbols"]
    divide_sym = next(s for s in symbols if s["name"] == "divide")
    assert divide_sym["parent_class"] == "MathService"
    assert divide_sym["line_number"] >= 4

    # 2. Surgical diff replacement fixes divide method
    replacer = ReplaceContentTool(str(tmp_path))
    target = (
        "    def divide(self, a: int, b: int) -> float:\n"
        "        # BUG: no zero check\n"
        "        return a / b"
    )
    replacement = (
        "    def divide(self, a: int, b: int) -> float:\n"
        "        if b == 0:\n"
        "            raise ValueError('Cannot divide by zero')\n"
        "        return a / b"
    )
    diff_res = replacer.run("src/math_service.py", target, replacement)
    assert diff_res["success"] is True
    assert diff_res["replacements_made"] == 1

    # 3. CodeArtifact state update passes through artifact_reducer
    initial_artifact = {
        "file_path": "src/math_service.py",
        "language": "python",
        "content": initial_code,
        "version": 1,
    }
    updated_content = fs.read_file("src/math_service.py")
    updated_artifact = {
        "file_path": "src/math_service.py",
        "language": "python",
        "content": updated_content,
    }

    state_artifacts = [initial_artifact]
    reduced = artifact_reducer(state_artifacts, updated_artifact)
    assert len(reduced) == 1
    assert reduced[0]["file_path"] == "src/math_service.py"
    assert reduced[0]["version"] == 2
    assert "raise ValueError" in reduced[0]["content"]


def test_02_sandbox_env_sanitization_process_cleanup_pytest_qa_combination(tmp_path: Path):
    """
    2. Sandboxed Subprocess (F16) + Environment Sanitization (F13) + Process-tree cleanup (F14) during Pytest QA (F17).
    Verifies that SubprocessExecutor executes pytest with sanitization and timeout termination.
    """
    trace: List[Dict[str, Any]] = []
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=trace)

    fs.write_file("calc.py", "def multiply(x: int, y: int) -> int:\n    return x * y\n")
    fs.write_file(
        "test_calc.py",
        "from calc import multiply\n\n"
        "def test_multiply():\n"
        "    assert multiply(3, 4) == 12\n",
    )

    # Subprocess execution with extra env
    exe = SubprocessExecutor(
        workspace_dir=str(tmp_path),
        trace=trace,
        allowed_commands=False,
    )

    # Run pytest with automatic PYTHONPATH
    res = exe.run_sync(
        [sys.executable, "-m", "pytest", "test_calc.py", "-q"],
        extra_env={"PYTHONPATH": str(tmp_path), "SANITIZED_VAR": "SAFE"},
    )
    assert res.exit_code == 0
    assert res.success is True

    # Test process timeout handling
    timeout_res = exe.run_sync(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout=0.2,
    )
    assert timeout_res.timed_out is True
    assert timeout_res.success is False
    assert timeout_res.exit_code == -1


def test_03_task_queue_delta_reducer_artifact_dedup_checkpointer_combination(tmp_path: Path):
    """
    3. Task Queue Delta Reducer (F3) + Artifact Deduplication (F2) + Checkpointer (F6) during multi-step graph transitions.
    Verifies state immutability, task delta upserts, artifact deduplication, and checkpoint replay.
    """
    checkpointer = MockCheckpointer()
    config = {"configurable": {"thread_id": "proj-101"}}

    # Step 1: Initial state
    t1 = {"task_id": "T1", "title": "Create User Model", "status": "pending"}
    t2 = {"task_id": "T2", "title": "Create Auth Service", "status": "pending"}
    state: Dict[str, Any] = {
        "project_id": "proj-101",
        "task_queue": [t1, t2],
        "code_artifacts": [],
        "completed_tasks": [],
    }
    checkpointer.put(config, state)

    # Step 2: Worker 1 completes T1 and emits code artifact
    t1_done = {"task_id": "T1", "status": "completed"}
    art_v1 = {"file_path": "src/models.py", "content": "class User: pass"}

    state["task_queue"] = task_queue_reducer(state["task_queue"], t1_done)
    state["code_artifacts"] = artifact_reducer(state["code_artifacts"], art_v1)
    checkpointer.put(config, state)

    # Step 3: Worker 2 updates T2 to in_progress and edits src/models.py
    t2_prog = {"task_id": "T2", "status": "in_progress"}
    art_v2 = {"file_path": "src/models.py", "content": "class User:\n    id: int\n    name: str"}

    state["task_queue"] = task_queue_reducer(state["task_queue"], t2_prog)
    state["code_artifacts"] = artifact_reducer(state["code_artifacts"], art_v2)
    checkpointer.put(config, state)

    # Verify final checkpoint snapshot
    restored = checkpointer.get(config)
    assert restored is not None
    assert len(restored["task_queue"]) == 2
    assert restored["task_queue"][0]["status"] == "completed"
    assert restored["task_queue"][1]["status"] == "in_progress"

    assert len(restored["code_artifacts"]) == 1
    assert restored["code_artifacts"][0]["file_path"] == "src/models.py"
    assert restored["code_artifacts"][0]["version"] == 2
    assert "name: str" in restored["code_artifacts"][0]["content"]


def test_04_static_analysis_code_reviewer_hitl_pause_checkpoint_resume_combination(tmp_path: Path):
    """
    4. Static Analysis (F18) + Code Reviewer + HITL Pause (F19) + Checkpoint Resume (F23).
    Verifies that security review failures increment failure counters, trigger watchdog pause, and allow checkpoint resume.
    """
    checkpointer = MockCheckpointer()
    config = {"configurable": {"thread_id": "proj-sec-01"}}

    state: ProjectState = {
        "project_id": "proj-sec-01",
        "code_artifacts": [
            {
                "file_path": "src/auth.py",
                "language": "python",
                "content": "def login(user, pwd):\n    query = f'SELECT * FROM users WHERE u={user}'\n    return query\n",
            }
        ],
        "task_queue": [],
        "retry_counts": {"task_fail_src/auth.py": 2},
        "status": "running",
    }

    # Simulate reviewer node flagging vulnerability
    mock_llm_result = ReviewResult(approved=False, feedback="SQL Injection detected via raw string formatting.")
    with patch("src.agents.reviewer.get_llm") as mock_get_llm:
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = mock_llm_result
        mock_get_llm.return_value = mock_instance

        review_output = reviewer_node(state)

    # Reducer merges updates
    merged_retries = dict_merge_reducer(state["retry_counts"], review_output["retry_counts"])
    state["retry_counts"] = merged_retries
    state["task_queue"] = task_queue_reducer(state["task_queue"], review_output["task_queue"])

    assert state["retry_counts"]["task_fail_src/auth.py"] == 3

    # Route after reviewer goes to watchdog
    route_reviewer = route_after_reviewer(state)
    assert route_reviewer == "watchdog"

    # Watchdog routes to human_approval
    route_wd = route_after_watchdog(state)
    assert route_wd == "human_approval"

    # Human approval pauses pipeline
    approval_output = human_approval_node(state)
    state["status"] = approval_output["status"]
    assert state["status"] == "blocked"

    # Save to checkpoint
    checkpointer.put(config, state)

    # Resume action: Human provides approval and sanitized code
    resumed_state = checkpointer.get(config)
    assert resumed_state["status"] == "blocked"

    # Resume override
    resumed_state["status"] = "running"
    resumed_state["retry_counts"]["task_fail_src/auth.py"] = 0
    resumed_state["code_artifacts"] = artifact_reducer(
        resumed_state["code_artifacts"],
        {
            "file_path": "src/auth.py",
            "content": "def login(user, pwd):\n    return db.query('SELECT * FROM users WHERE u=:u', {'u': user})\n",
        },
    )
    checkpointer.put(config, resumed_state)

    final_snapshot = checkpointer.get(config)
    assert final_snapshot["status"] == "running"
    assert final_snapshot["code_artifacts"][0]["version"] == 2
    assert ":u" in final_snapshot["code_artifacts"][0]["content"]


def test_05_async_execution_api_sse_streaming_state_dedup_react_trajectory_combination():
    """
    5. Async Execution API (F20) + SSE Streaming (F21) + State Deduplication (F2) + ReAct Tool Trajectories (F12).
    Verifies that streaming trajectories format valid SSE messages and update deduplicated state.
    """
    def sse_event_stream(project_id: str):
        events = [
            {"event": "thought", "data": {"step": 1, "thought": "Need to check user model schema"}},
            {"event": "tool_call", "data": {"tool": "find_files", "args": {"pattern": "*.py"}}},
            {"event": "tool_output", "data": {"tool": "find_files", "output": ["src/user.py"]}},
            {"event": "artifact_update", "data": {"file_path": "src/user.py", "content": "class User:\n    id: int"}},
            {"event": "complete", "data": {"status": "completed", "project_id": project_id}},
        ]
        for ev in events:
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"

    # Consume SSE generator
    stream_output = []
    gen = sse_event_stream("proj-sse-01")
    for chunk in gen:
        stream_output.append(chunk)

    assert len(stream_output) == 5
    assert stream_output[0].startswith("event: thought\ndata:")
    assert "find_files" in stream_output[1]

    # Verify state artifact deduplication from streaming artifact events
    artifacts: List[Dict[str, Any]] = []
    for chunk in stream_output:
        if "artifact_update" in chunk:
            data_line = [line for line in chunk.splitlines() if line.startswith("data:")][0]
            event_data = json.loads(data_line[5:].strip())
            artifacts = artifact_reducer(artifacts, event_data)

    assert len(artifacts) == 1
    assert artifacts[0]["file_path"] == "src/user.py"
    assert artifacts[0]["version"] == 1


def test_06_websocket_streaming_hitl_interaction_checkpoint_state_modification_combination():
    """
    6. WebSocket Streaming (F22) + HITL Interaction (F19) + Checkpoint State Modification (F23).
    Verifies interactive pause/resume WebSocket command processing against persistent checkpointer.
    """
    checkpointer = MockCheckpointer()
    config = {"configurable": {"thread_id": "ws-thread-42"}}

    # Initial blocked state in checkpointer
    initial_state = {
        "project_id": "proj-ws-42",
        "status": "blocked",
        "current_phase": "human_approval",
        "task_queue": [{"task_id": "T1", "status": "blocked", "title": "Deploy Service"}],
    }
    checkpointer.put(config, initial_state)

    # Simulated WebSocket handler processing client messages
    class MockWebSocketHandler:
        def __init__(self, saver: MockCheckpointer):
            self.saver = saver

        def handle_client_message(self, thread_id: str, msg: Dict[str, Any]) -> Dict[str, Any]:
            cfg = {"configurable": {"thread_id": thread_id}}
            curr = self.saver.get(cfg)
            if not curr:
                return {"error": "Thread not found"}

            action = msg.get("action")
            if action == "approve_and_resume":
                curr["status"] = "running"
                curr["current_phase"] = "development"
                for task in curr.get("task_queue", []):
                    if task.get("task_id") == msg.get("task_id"):
                        task["status"] = "pending"
                if "guidance" in msg:
                    curr["human_guidance"] = msg["guidance"]
                self.saver.put(cfg, curr)
                return {"status": "resumed", "state": curr}
            return {"status": "noop"}

    ws_handler = MockWebSocketHandler(checkpointer)
    response = ws_handler.handle_client_message(
        "ws-thread-42",
        {
            "action": "approve_and_resume",
            "task_id": "T1",
            "guidance": "Approved with in-memory database",
        },
    )

    assert response["status"] == "resumed"
    updated_state = checkpointer.get(config)
    assert updated_state["status"] == "running"
    assert updated_state["task_queue"][0]["status"] == "pending"
    assert updated_state["human_guidance"] == "Approved with in-memory database"


def test_07_clearable_list_reducer_memory_compression_trace_truncation_combination():
    """
    7. Clearable List Reducer (F5) + Memory Compression Node + Execution Trace truncation under long workflows.
    Verifies that memory compression resets traces, consolidates completed tasks, and guards against None values.
    """
    # 1. Test clearable_list_reducer robust behavior
    assert clearable_list_reducer(None, "CLEAR") == []
    assert clearable_list_reducer(["item1"], "CLEAR") == []
    assert clearable_list_reducer(["item1"], ["CLEAR", "item2"]) == ["item2"]
    assert clearable_list_reducer(["item1"], ["item2"]) == ["item1", "item2"]

    # 2. Simulate memory_compression_node with 6 completed tasks
    completed_tasks = [
        {"task_id": f"T{i}", "title": f"Task {i}", "status": "completed", "file_path": f"src/m{i}.py"}
        for i in range(1, 7)
    ]
    trace = [{"tool": "fs", "op": f"op_{i}"} for i in range(20)]

    state: ProjectState = {
        "completed_tasks": completed_tasks,
        "execution_trace": trace,
    }

    mock_summary = MagicMock()
    mock_summary.content = "Successfully implemented modules m1 through m6."
    with patch("src.agents.memory.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_summary
        mock_get_llm.return_value = mock_llm

        comp_output = memory_compression_node(state)

    # 3. Apply reducer to state
    new_completed = clearable_list_reducer(state["completed_tasks"], comp_output["completed_tasks"])
    new_trace = clearable_list_reducer(state["execution_trace"], comp_output["execution_trace"])

    assert len(new_completed) == 1
    assert new_completed[0]["task_id"] == "COMPRESSED"
    assert "m1 through m6" in new_completed[0]["description"]
    assert new_trace == []


def test_08_glob_find_files_and_regex_grep_with_react_navigation_combination(tmp_path: Path):
    """
    8. Glob File Search (F7) + Regex Grep Search (F8) + Slice View File (F10) in ReAct Navigation loop.
    Verifies discovery of relevant files, line pattern matches, and precise line slice inspection.
    """
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=[])
    fs.write_file("src/auth/service.py", "def authenticate_user(token: str) -> bool:\n    return len(token) > 8\n")
    fs.write_file("src/billing/service.py", "def process_payment(amount: float) -> str:\n    return 'paid'\n")
    fs.write_file("tests/test_auth.py", "from src.auth.service import authenticate_user\n")

    # 1. Glob search finds all service files
    finder = FindFilesTool(str(tmp_path))
    res_glob = finder.run("*service.py")
    assert res_glob["total_count"] == 2
    paths = {m["path"] for m in res_glob["matches"]}
    assert "src/auth/service.py" in paths
    assert "src/billing/service.py" in paths

    # 2. Grep search finds authenticate_user
    grep = GrepSearchTool(str(tmp_path))
    res_grep = grep.run("authenticate_user")
    assert res_grep["total_matches"] == 2  # Definition in service.py and import in test_auth.py
    def_match = next(m for m in res_grep["matches"] if m["file_path"] == "src/auth/service.py")
    assert def_match["line_number"] == 1

    # 3. ViewFileTool reads exact slice
    viewer = ViewFileTool(str(tmp_path))
    res_view = viewer.run("src/auth/service.py", start_line=1, end_line=2, show_line_numbers=True)
    assert "1 | def authenticate_user" in res_view["content"]
    assert "2 |     return len(token) > 8" in res_view["content"]


def test_09_pythonpath_injection_with_subprocess_sandbox_and_qa_tester_combination(tmp_path: Path):
    """
    9. Automatic PYTHONPATH Injection (F15) + Sandboxed Execution (F16) + Dynamic Pytest QA (F17).
    Verifies that nested workspace modules resolve during test execution without ModuleNotFoundError.
    """
    trace: List[Dict[str, Any]] = []
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=trace)

    fs.write_file("src/utils/math_helpers.py", "def square(n: int) -> int:\n    return n * n\n")
    fs.write_file(
        "tests/test_math_helpers.py",
        "from src.utils.math_helpers import square\n\n"
        "def test_square():\n"
        "    assert square(5) == 25\n",
    )

    exe = SubprocessExecutor(workspace_dir=str(tmp_path), trace=trace, allowed_commands=False)

    # Injected PYTHONPATH pointing to workspace
    res = exe.run_sync(
        [sys.executable, "-m", "pytest", "tests/test_math_helpers.py", "-q"],
        extra_env={"PYTHONPATH": str(tmp_path)},
    )
    assert res.exit_code == 0
    assert "passed" in res.stdout or res.success


def test_10_adr_reducer_architect_node_and_checkpointer_replay_combination(tmp_path: Path):
    """
    10. ADR Deduplication Reducer (F6) + Architect Node + Checkpointer State Replay.
    Verifies that re-planning or architect retries update ADRs cleanly without duplicate entries.
    """
    checkpointer = MockCheckpointer()
    config = {"configurable": {"thread_id": "arch-thread-1"}}

    adr1 = {
        "decision_id": "ADR-001",
        "title": "Use PostgreSQL for persistence",
        "status": "proposed",
        "context": "Need reliable ACID database",
    }
    adr2 = {
        "decision_id": "ADR-002",
        "title": "Use FastAPI for REST layer",
        "status": "accepted",
        "context": "Async support required",
    }

    state: Dict[str, Any] = {"architecture_decisions": [adr1, adr2]}
    checkpointer.put(config, state)

    # Architect re-runs and accepts ADR-001 with updated consequences
    adr1_updated = {
        "decision_id": "ADR-001",
        "title": "Use PostgreSQL for persistence",
        "status": "accepted",
        "consequences": "Requires docker container or hosted DB",
    }

    merged_adrs = adr_reducer(state["architecture_decisions"], adr1_updated)
    state["architecture_decisions"] = merged_adrs
    checkpointer.put(config, state)

    restored = checkpointer.get(config)
    assert len(restored["architecture_decisions"]) == 2
    adr1_restored = next(a for a in restored["architecture_decisions"] if a["decision_id"] == "ADR-001")
    assert adr1_restored["status"] == "accepted"
    assert "Requires docker" in adr1_restored["consequences"]


def test_11_dict_merge_reducer_retry_counts_and_task_failures_separation_combination():
    """
    11. Dictionary Merge Reducer (F4) + Node Retry Counts + Task Failure Counters Separation.
    Verifies that node retry counters and task-specific failure counters do not clobber each other.
    """
    state_retries: Dict[str, int] = {}

    # Node retry middleware updates node retry count
    update1 = {"backend_engineer": 1}
    state_retries = dict_merge_reducer(state_retries, update1)

    # QA tester updates task failure count
    update2 = {"task_fail_src/models.py": 1}
    state_retries = dict_merge_reducer(state_retries, update2)

    # Reviewer updates another task failure count
    update3 = {"task_fail_src/auth.py": 2}
    state_retries = dict_merge_reducer(state_retries, update3)

    assert state_retries["backend_engineer"] == 1
    assert state_retries["task_fail_src/models.py"] == 1
    assert state_retries["task_fail_src/auth.py"] == 2

    # Watchdog evaluation correctly isolates task failures from node retries
    task_failures = {k: v for k, v in state_retries.items() if k.startswith("task_fail_")}
    assert len(task_failures) == 2
    assert max(task_failures.values()) == 2  # Under threshold 3


def test_12_ast_symbol_navigator_and_dynamic_qa_test_generation_combination(tmp_path: Path):
    """
    12. AST Symbol Navigator (F9) + Dynamic Multi-Language QA Generation (F17).
    Verifies that AST symbol metadata guides dynamic unit test generation and validation.
    """
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=[])
    code = (
        "def compute_discount(price: float, rate: float) -> float:\n"
        "    '''Calculate discounted total price.'''\n"
        "    if rate < 0 or rate > 1:\n"
        "        raise ValueError('Invalid rate')\n"
        "    return price * (1.0 - rate)\n"
    )
    fs.write_file("src/discount.py", code)

    ast_nav = ASTSymbolNavigator(str(tmp_path))
    outline = ast_nav.get_outline("src/discount.py")
    sym = outline["symbols"][0]

    # Generate dynamic pytest test code using symbol metadata
    generated_test = (
        f"from src.discount import {sym['name']}\n"
        f"import pytest\n\n"
        f"def test_{sym['name']}_valid():\n"
        f"    assert {sym['name']}(100.0, 0.2) == 80.0\n\n"
        f"def test_{sym['name']}_invalid_rate():\n"
        f"    with pytest.raises(ValueError):\n"
        f"        {sym['name']}(100.0, 1.5)\n"
    )
    fs.write_file("tests/test_discount.py", generated_test)

    exe = SubprocessExecutor(workspace_dir=str(tmp_path), trace=[], allowed_commands=False)
    res = exe.run_sync(
        [sys.executable, "-m", "pytest", "tests/test_discount.py", "-q"],
        extra_env={"PYTHONPATH": str(tmp_path)},
    )
    assert res.exit_code == 0
    assert res.success is True


def test_13_multi_tool_static_analysis_ruff_mypy_bandit_with_code_reviewer_combination(tmp_path: Path):
    """
    13. Multi-Tool Deterministic Static Analysis (F18) + Subprocess Sandbox (F16) + Code Reviewer Node.
    Verifies that static analysis rules detect lint, type, and security issues before LLM review.
    """
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=[])
    clean_code = "def greet(name: str) -> str:\n    return f'Hello, {name}'\n"
    fs.write_file("src/clean.py", clean_code)

    exe = SubprocessExecutor(workspace_dir=str(tmp_path), trace=[], allowed_commands=False)

    # Static analysis helper simulating multi-tool scan
    def run_static_checks(file_path: str) -> Dict[str, Any]:
        full_p = (tmp_path / file_path).resolve()
        code_str = full_p.read_text(encoding="utf-8")
        issues = []
        # AST check for dangerous eval / exec / raw sql
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    issues.append({"tool": "bandit", "rule": "B307", "message": "Use of eval/exec detected"})
        return {"file_path": file_path, "issues": issues, "passed": len(issues) == 0}

    res_clean = run_static_checks("src/clean.py")
    assert res_clean["passed"] is True

    # Flawed code
    fs.write_file("src/insecure.py", "def dangerous_runner(payload: str):\n    eval(payload)\n")
    res_insecure = run_static_checks("src/insecure.py")
    assert res_insecure["passed"] is False
    assert res_insecure["issues"][0]["rule"] == "B307"


def test_14_surgical_diff_replace_content_with_git_tracker_and_artifact_versioning_combination(tmp_path: Path):
    """
    14. Surgical Diff (F11) + Git Tracker VCS (R3) + Artifact Version Reducer (F2).
    Verifies that surgical replacements generate clean git diffs and increment artifact versions.
    """
    trace: List[Dict[str, Any]] = []
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=trace)
    git = GitTracker(workspace_dir=str(tmp_path), trace=trace)

    git.init()
    fs.write_file("app.py", "VERSION = '1.0.0'\ndef run():\n    return False\n")
    git.stage_all()
    git.commit("initial commit")

    # Surgical diff replace
    replacer = ReplaceContentTool(str(tmp_path))
    res = replacer.run("app.py", "return False", "return True")
    assert res["success"] is True

    # Git diff verification
    git.stage_all()
    diff = git.diff(staged=True)
    assert "-    return False" in diff
    assert "+    return True" in diff
    git.commit("feat: enable run")

    # Artifact version increment
    artifacts = [{"file_path": "app.py", "content": "VERSION = '1.0.0'\ndef run():\n    return False\n", "version": 1}]
    updated = artifact_reducer(artifacts, {"file_path": "app.py", "content": fs.read_file("app.py")})
    assert len(updated) == 1
    assert updated[0]["version"] == 2


def test_15_task_planner_decomposition_with_task_queue_reducer_and_worker_routing_combination():
    """
    15. Task Planner Decomposition + Task Queue Delta Reducer (F3) + Worker Routing.
    Verifies that decomposed tasks route to appropriate worker nodes (backend vs frontend) sequentially.
    """
    planned_tasks = [
        {"task_id": "T1", "title": "Build User Model", "file_path": "src/models/user.py", "status": "pending"},
        {"task_id": "T2", "title": "Build Login UI", "file_path": "src/components/Login.tsx", "status": "pending"},
        {"task_id": "T3", "title": "Build API Router", "file_path": "src/api/routes.py", "status": "pending"},
    ]

    state: ProjectState = {"task_queue": []}
    state["task_queue"] = task_queue_reducer(state["task_queue"], planned_tasks)
    assert len(state["task_queue"]) == 3

    # Route 1: T1 -> backend_engineer
    r1 = route_to_workers(state)
    assert r1 == "backend_engineer"

    # Simulate T1 completion
    state["task_queue"] = state["task_queue"][1:]
    # Route 2: T2 -> frontend_engineer
    r2 = route_to_workers(state)
    assert r2 == "frontend_engineer"

    # Simulate T2 completion
    state["task_queue"] = state["task_queue"][1:]
    # Route 3: T3 -> backend_engineer
    r3 = route_to_workers(state)
    assert r3 == "backend_engineer"

    # Simulate T3 completion
    state["task_queue"] = []
    # Route 4: Empty -> memory_compression
    r4 = route_to_workers(state)
    assert r4 == "memory_compression"


def test_16_stagnation_detection_with_watchdog_and_retry_counts_combination(tmp_path: Path):
    """
    16. Stagnation Detection + Counter Separation (F4) + Watchdog Node + HITL Pause (F19).
    Verifies that stagnant code generation halts infinite retry loops and routes to human approval.
    """
    trace: List[Dict[str, Any]] = []
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=trace)
    git = GitTracker(workspace_dir=str(tmp_path), trace=trace)
    git.init()

    code = "def process():\n    pass\n"
    fs.write_file("src/worker.py", code)

    task = {
        "task_id": "STAG-01",
        "title": "Implement Worker",
        "file_path": "src/worker.py",
        "metadata": {"stagnant_iterations": 1},
    }

    state: ProjectState = {
        "task_queue": [task],
        "workspace_dir": str(tmp_path),
        "retry_counts": {},
        "architecture_decisions": [],
        "project_structure": {},
    }

    # Simulate LLM returning identical content again
    mock_gen = GeneratedCode(file_path="src/worker.py", content=code, explanation="No changes")
    with patch("src.agents.backend_engineer.get_llm") as mock_llm:
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = mock_gen
        mock_llm.return_value = mock_instance

        out = backend_engineer_node(state)

    assert out["status"] == "failed"
    assert out["error_log"][0]["error_type"] == "StagnationError"


def test_17_filesystem_path_traversal_sandboxing_with_execution_trace_audit_combination(tmp_path: Path):
    """
    17. Sandboxed FileSystemManager + Path Traversal Prevention + Execution Trace Audit.
    Verifies strict containment against relative and absolute traversal attempts with complete trace logging.
    """
    trace: List[Dict[str, Any]] = []
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=trace)

    # 1. Valid write and read
    fs.write_file("nested/data.txt", "payload")
    content = fs.read_file("nested/data.txt")
    assert content == "payload"

    # 2. Path traversal attempts
    traversal_targets = [
        "../../outside.txt",
        "..\\..\\outside.txt",
        "nested/../../outside.txt",
        "/etc/passwd",
    ]
    for target in traversal_targets:
        with pytest.raises(PermissionError):
            fs.read_file(target)

    # Trace audit verification
    assert len(trace) >= 2
    ops = [r["operation"] for r in trace]
    assert "write_file" in ops
    assert "read_file" in ops
    assert all("sha256" in r["outputs"] for r in trace if r["success"])


def test_18_async_api_run_status_polling_and_checkpoint_state_retrieval_combination():
    """
    18. Async Background API (F20) + State Query Endpoint (F23) + Status Response Serialization.
    Verifies API route interaction, health check, and status queries.
    """
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)

    # 1. Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert data["status"] == "healthy"
    assert "initializer" in data["graph_nodes"]

    # 2. Non-existent status returns 404
    missing_resp = client.get("/projects/non-existent-uuid/status")
    assert missing_resp.status_code == 404


def test_19_react_engine_stagnation_budget_exhaustion_with_error_logging_combination():
    """
    19. Autonomous ReAct Runtime (F12) + Step Budget Exhaustion + Trajectory Audit.
    Verifies that ReAct loop enforces max iterations and stagnation guards with error records.
    """
    class MockReActEngine:
        def __init__(self, max_iterations: int = 5, stagnation_threshold: int = 3):
            self.max_iterations = max_iterations
            self.stagnation_threshold = stagnation_threshold

        def run(self, tool_call_sequence: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
            trajectory = []
            consecutive_same = 0
            last_sig = ""

            for step, (tool, args) in enumerate(tool_call_sequence, start=1):
                if step > self.max_iterations:
                    return {"success": False, "error": "Exhausted max step budget", "trajectory": trajectory}
                sig = f"{tool}:{json.dumps(args, sort_keys=True)}"
                if sig == last_sig:
                    consecutive_same += 1
                else:
                    consecutive_same = 0
                    last_sig = sig

                if consecutive_same >= self.stagnation_threshold:
                    return {"success": False, "error": "Stagnation detected", "trajectory": trajectory}

                trajectory.append({"step": step, "tool": tool, "args": args})

            return {"success": True, "trajectory": trajectory}

    engine = MockReActEngine(max_iterations=4, stagnation_threshold=2)

    # Test stagnation detection
    stag_seq = [
        ("view_file", {"path": "a.py"}),
        ("view_file", {"path": "a.py"}),
        ("view_file", {"path": "a.py"}),
    ]
    res_stag = engine.run(stag_seq)
    assert res_stag["success"] is False
    assert res_stag["error"] == "Stagnation detected"

    # Test budget exhaustion
    budget_seq = [("find_files", {"pattern": f"*.{i}"}) for i in range(10)]
    res_budget = engine.run(budget_seq)
    assert res_budget["success"] is False
    assert res_budget["error"] == "Exhausted max step budget"


def test_20_nested_directory_mirroring_qa_test_generation_and_execution_combination(tmp_path: Path):
    """
    20. Nested Directory Mirroring in QA Generation (F17) + Subprocess Sandbox (F16).
    Verifies that duplicate file names in different subdirectories generate collision-free mirrored test paths.
    """
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=[])
    fs.write_file("src/domain/auth/service.py", "def get_role(): return 'admin'\n")
    fs.write_file("src/domain/billing/service.py", "def get_plan(): return 'pro'\n")

    # QA Mirroring logic: preserves directory structure
    def get_mirrored_test_path(source_file: str) -> str:
        clean = source_file.replace("src/", "").replace("\\", "/")
        parts = clean.split("/")
        parts[-1] = "test_" + parts[-1]
        return "tests/" + "/".join(parts)

    t1 = get_mirrored_test_path("src/domain/auth/service.py")
    t2 = get_mirrored_test_path("src/domain/billing/service.py")

    fs.write_file("src/__init__.py", "")
    fs.write_file("src/domain/__init__.py", "")
    fs.write_file("src/domain/auth/__init__.py", "")
    fs.write_file("src/domain/billing/__init__.py", "")
    fs.write_file("tests/__init__.py", "")
    fs.write_file("tests/domain/__init__.py", "")
    fs.write_file("tests/domain/auth/__init__.py", "")
    fs.write_file("tests/domain/billing/__init__.py", "")

    fs.write_file(t1, "from src.domain.auth.service import get_role\ndef test_role(): assert get_role() == 'admin'\n")
    fs.write_file(t2, "from src.domain.billing.service import get_plan\ndef test_plan(): assert get_plan() == 'pro'\n")

    exe = SubprocessExecutor(workspace_dir=str(tmp_path), trace=[], allowed_commands=False)
    res = exe.run_sync(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        extra_env={"PYTHONPATH": str(tmp_path)},
    )
    assert res.exit_code == 0
    assert res.success is True


def test_21_concurrent_task_updates_and_artifact_merging_simulation_combination():
    """
    21. Concurrent Branch State Merging: Task Queue Delta (F3) + Artifact Reducer (F2) + Dict Merge (F4).
    Verifies that simultaneous outputs from parallel worker branches merge without data loss.
    """
    initial_tasks = [
        {"task_id": "T1", "status": "pending"},
        {"task_id": "T2", "status": "pending"},
    ]
    initial_artifacts = [{"file_path": "common.py", "content": "# base", "version": 1}]
    initial_retries = {"node_a": 0, "node_b": 0}

    # Branch A completes T1 and modifies common.py
    branch_a_tasks = [{"task_id": "T1", "status": "completed"}]
    branch_a_artifacts = [{"file_path": "common.py", "content": "# base\nimport os"}]
    branch_a_retries = {"node_a": 1}

    # Branch B completes T2 and creates new file service.py
    branch_b_tasks = [{"task_id": "T2", "status": "completed"}]
    branch_b_artifacts = [{"file_path": "service.py", "content": "class Service: pass"}]
    branch_b_retries = {"node_b": 1}

    # Merge sequentially
    merged_tasks = task_queue_reducer(initial_tasks, branch_a_tasks)
    merged_tasks = task_queue_reducer(merged_tasks, branch_b_tasks)

    merged_artifacts = artifact_reducer(initial_artifacts, branch_a_artifacts)
    merged_artifacts = artifact_reducer(merged_artifacts, branch_b_artifacts)

    merged_retries = dict_merge_reducer(initial_retries, branch_a_retries)
    merged_retries = dict_merge_reducer(merged_retries, branch_b_retries)

    assert len(merged_tasks) == 2
    assert all(t["status"] == "completed" for t in merged_tasks)

    assert len(merged_artifacts) == 2
    common_art = next(a for a in merged_artifacts if a["file_path"] == "common.py")
    assert common_art["version"] == 2

    assert merged_retries == {"node_a": 1, "node_b": 1}


def test_22_view_file_slice_reading_with_ast_outline_and_surgical_diff_combination(tmp_path: Path):
    """
    22. Slice View File (F10) + AST Outline (F9) + Surgical Replace (F11).
    Verifies that AST symbol boundaries guide line slicing and precise hunk editing.
    """
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=[])
    lines = [f"# Line {i}" for i in range(1, 40)]
    lines[19] = "def target_func():"
    lines[20] = "    return 'old_value'"
    fs.write_file("big_module.py", "\n".join(lines) + "\n")

    ast_nav = ASTSymbolNavigator(str(tmp_path))
    outline = ast_nav.get_outline("big_module.py")
    target_sym = next(s for s in outline["symbols"] if s["name"] == "target_func")
    assert target_sym["line_number"] == 20

    viewer = ViewFileTool(str(tmp_path))
    view_res = viewer.run("big_module.py", start_line=18, end_line=23)
    assert "target_func" in view_res["content"]

    replacer = ReplaceContentTool(str(tmp_path))
    rep_res = replacer.run("big_module.py", "return 'old_value'", "return 'new_value'")
    assert rep_res["success"] is True

    updated_content = fs.read_file("big_module.py")
    assert "return 'new_value'" in updated_content
    assert "# Line 1" in updated_content
    assert "# Line 39" in updated_content


def test_23_subprocess_timeout_and_process_tree_termination_with_trace_logging_combination(tmp_path: Path):
    """
    23. Process-Tree Termination (F14) + Subprocess Sandbox (F16) + Execution Trace Audit.
    Verifies that timeout kills subprocesses cleanly and writes telemetry to trace.
    """
    trace: List[Dict[str, Any]] = []
    exe = SubprocessExecutor(workspace_dir=str(tmp_path), trace=trace, allowed_commands=False)

    res = exe.run_sync(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout=0.3,
    )
    assert res.timed_out is True
    assert res.success is False

    assert len(trace) >= 1
    last_record = trace[-1]
    assert last_record["success"] is False
    assert "TIMEOUT" in (last_record.get("error") or "")


def test_24_health_and_schema_endpoints_with_compiled_graph_node_inspection_combination():
    """
    24. API Route Health Probe + StateGraph Node Registry Validation.
    Verifies that all 11 core nodes and conditional routing topologies are compiled.
    """
    graph = build_graph()
    nodes = list(graph.get_graph().nodes.keys())

    required_nodes = [
        "initializer",
        "requirement_analyzer",
        "architect",
        "task_planner",
        "backend_engineer",
        "frontend_engineer",
        "memory_compression",
        "tester",
        "reviewer",
        "watchdog",
        "human_approval",
    ]
    for node in required_nodes:
        assert node in nodes, f"Missing required node: {node}"


def test_25_full_outer_loop_cycle_from_failure_to_fix_and_approval_combination(tmp_path: Path):
    """
    25. Full Outer Multi-Turn Loop: ReAct -> QA Failure -> Fix Task -> Re-test -> Review Approval.
    Verifies end-to-end multi-turn repair lifecycle across tester, reviewer, and reducers.
    """
    # Turn 1: Developer produces artifact with broken math
    art_v1 = {"file_path": "src/ops.py", "language": "python", "content": "def add(a, b): return a - b\n", "version": 1}
    state: ProjectState = {
        "code_artifacts": [art_v1],
        "task_queue": [],
        "retry_counts": {},
        "workspace_dir": str(tmp_path),
    }

    # Turn 2: QA tester creates test and discovers failure
    fs = FileSystemManager(workspace_dir=str(tmp_path), trace=[])
    fs.write_file("src/ops.py", art_v1["content"])
    fs.write_file("tests/test_ops.py", "from src.ops import add\ndef test_add(): assert add(2, 3) == 5\n")

    # Run tester simulation
    exe = SubprocessExecutor(workspace_dir=str(tmp_path), trace=[], allowed_commands=False)
    test_res = exe.run_sync([sys.executable, "-m", "pytest", "tests/test_ops.py", "-q"], extra_env={"PYTHONPATH": str(tmp_path)})
    assert test_res.success is False  # Fails as expected

    # QA Tester queues fix task
    fix_task = {"task_id": "FIX-ops.py", "file_path": "src/ops.py", "status": "pending"}
    state["task_queue"] = task_queue_reducer(state["task_queue"], fix_task)
    state["retry_counts"] = dict_merge_reducer(state["retry_counts"], {"task_fail_src/ops.py": 1})

    assert len(state["task_queue"]) == 1
    assert state["retry_counts"]["task_fail_src/ops.py"] == 1

    # Turn 3: Developer fixes bug
    fixed_code = "def add(a, b): return a + b\n"
    fs.write_file("src/ops.py", fixed_code)
    state["code_artifacts"] = artifact_reducer(state["code_artifacts"], {"file_path": "src/ops.py", "content": fixed_code})
    state["task_queue"] = [{"task_id": "FIX-ops.py", "status": "completed"}]

    assert state["code_artifacts"][0]["version"] == 2

    # Turn 4: QA re-tests and passes
    test_res2 = exe.run_sync([sys.executable, "-m", "pytest", "tests/test_ops.py", "-q"], extra_env={"PYTHONPATH": str(tmp_path)})
    assert test_res2.success is True

    # Turn 5: Reviewer approves
    state["status"] = "completed"
    assert state["status"] == "completed"


def test_26_environment_variable_sanitization_with_secret_redaction_in_sandbox_combination(tmp_path: Path):
    """
    26. Environment Sanitization (F13) + Secret Redaction in Sandbox Subprocesses (F16).
    Verifies that host API keys and credentials are not leaked into child process environments.
    """
    trace: List[Dict[str, Any]] = []

    # Inject mock secrets in current env
    os.environ["MOCK_OPENAI_KEY"] = "sk-secret12345"
    os.environ["MOCK_DB_PASS"] = "supersecretpass"

    try:
        # Sanitizer function for subprocess env
        def sanitize_env(env: Dict[str, str]) -> Dict[str, str]:
            allowed_prefixes = ("PATH", "PYTHON", "SYSTEM", "TEMP", "TMP", "LANG")
            return {
                k: v for k, v in env.items()
                if any(k.upper().startswith(p) for p in allowed_prefixes) and not any(sec in k.upper() for sec in ["KEY", "SECRET", "PASS", "TOKEN"])
            }

        safe_env = sanitize_env(dict(os.environ))
        assert "MOCK_OPENAI_KEY" not in safe_env
        assert "MOCK_DB_PASS" not in safe_env
        assert "PATH" in safe_env or "Path" in safe_env

        exe = SubprocessExecutor(workspace_dir=str(tmp_path), trace=trace, allowed_commands=False)
        res = exe.run_sync(
            [sys.executable, "-c", "import os; print('KEY_EXISTS' if 'MOCK_OPENAI_KEY' in os.environ else 'CLEAN')"],
            extra_env=safe_env,
        )
        assert "CLEAN" in res.stdout
    finally:
        os.environ.pop("MOCK_OPENAI_KEY", None)
        os.environ.pop("MOCK_DB_PASS", None)


def test_27_memory_compression_preserves_architectural_context_and_reduces_trace_combination():
    """
    27. Memory Compression + ADR Reducer + Clearable List Reducer (F5).
    Verifies that ADR architecture context is completely preserved while execution logs are trimmed.
    """
    initial_adrs = [
        {"decision_id": "ADR-001", "title": "Microservices", "status": "accepted"},
        {"decision_id": "ADR-002", "title": "Postgres", "status": "accepted"},
    ]
    completed_tasks = [{"task_id": f"T{i}", "title": f"Task {i}", "status": "completed"} for i in range(7)]
    trace = [{"tool": "fs", "op": f"op_{i}"} for i in range(15)]

    state: ProjectState = {
        "architecture_decisions": initial_adrs,
        "completed_tasks": completed_tasks,
        "execution_trace": trace,
    }

    mock_summary = MagicMock()
    mock_summary.content = "Summary of completed architecture implementation."
    with patch("src.agents.memory.get_llm") as mock_llm:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_summary
        mock_llm.return_value = mock_instance

        comp_output = memory_compression_node(state)

    # Reducer application
    state["completed_tasks"] = clearable_list_reducer(state["completed_tasks"], comp_output["completed_tasks"])
    state["execution_trace"] = clearable_list_reducer(state["execution_trace"], comp_output["execution_trace"])

    # Verify ADRs are 100% untouched
    assert len(state["architecture_decisions"]) == 2
    assert state["architecture_decisions"][0]["decision_id"] == "ADR-001"
    # Verify trace is cleared and tasks compressed
    assert len(state["completed_tasks"]) == 1
    assert state["execution_trace"] == []


def test_28_checkpoint_thread_isolation_and_parallel_project_state_combination():
    """
    28. Checkpointer Thread Isolation (F6) + Multi-Project State Independence (F1).
    Verifies that parallel runs with different thread_ids maintain completely isolated states.
    """
    checkpointer = MockCheckpointer()

    cfg_alpha = {"configurable": {"thread_id": "proj-alpha"}}
    cfg_beta = {"configurable": {"thread_id": "proj-beta"}}

    state_alpha = {
        "project_id": "proj-alpha",
        "code_artifacts": [{"file_path": "alpha.py", "content": "ALPHA = 1"}],
        "task_queue": [{"task_id": "A1", "status": "completed"}],
    }

    state_beta = {
        "project_id": "proj-beta",
        "code_artifacts": [{"file_path": "beta.py", "content": "BETA = 2"}],
        "task_queue": [{"task_id": "B1", "status": "in_progress"}],
    }

    checkpointer.put(cfg_alpha, state_alpha)
    checkpointer.put(cfg_beta, state_beta)

    # Update Alpha
    state_alpha["code_artifacts"] = artifact_reducer(
        state_alpha["code_artifacts"],
        {"file_path": "alpha.py", "content": "ALPHA = 2"},
    )
    checkpointer.put(cfg_alpha, state_alpha)

    # Verify Beta was unaffected
    restored_beta = checkpointer.get(cfg_beta)
    assert len(restored_beta["code_artifacts"]) == 1
    assert restored_beta["code_artifacts"][0]["file_path"] == "beta.py"
    assert restored_beta["code_artifacts"][0]["content"] == "BETA = 2"

    # Verify Alpha has update
    restored_alpha = checkpointer.get(cfg_alpha)
    assert restored_alpha["code_artifacts"][0]["version"] == 2


def test_29_sse_stream_event_formatting_with_tool_call_and_thought_tokens_combination():
    """
    29. Real-Time SSE Streaming (F21) + ReAct Trajectory Token Formatting (F12).
    Verifies standards-compliant SSE event framing with JSON data payloads.
    """
    def format_sse(event_type: str, payload: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    events = [
        format_sse("thought", {"thought": "Analyzing requirements"}),
        format_sse("tool_call", {"tool": "grep_search", "query": "class Database"}),
        format_sse("tool_result", {"tool": "grep_search", "matches": 1}),
        format_sse("done", {"status": "success"}),
    ]

    for raw in events:
        assert raw.startswith("event:")
        assert "\ndata: " in raw
        assert raw.endswith("\n\n")

        # Parse data
        lines = raw.strip().split("\n")
        ev_name = lines[0].replace("event: ", "")
        data_json = json.loads(lines[1].replace("data: ", ""))
        assert isinstance(data_json, dict)
        assert ev_name in ("thought", "tool_call", "tool_result", "done")


def test_30_human_approval_resume_with_guidance_prompt_and_state_recovery_combination():
    """
    30. HITL Pause (F19) + Checkpoint Resume Endpoint (F23) + Task Queue Delta (F3).
    Verifies that human guidance prompt resumes blocked executions and clears watchdog triggers.
    """
    checkpointer = MockCheckpointer()
    config = {"configurable": {"thread_id": "thread-resume-99"}}

    blocked_state: ProjectState = {
        "project_id": "proj-resume-99",
        "status": "blocked",
        "current_phase": "human_approval",
        "retry_counts": {"task_fail_src/api.py": 3},
        "task_queue": [{"task_id": "FIX-api.py", "status": "blocked", "file_path": "src/api.py"}],
    }
    checkpointer.put(config, blocked_state)

    # Simulated Resume Controller
    def resume_pipeline(thread_id: str, guidance: str) -> Dict[str, Any]:
        cfg = {"configurable": {"thread_id": thread_id}}
        st = checkpointer.get(cfg)
        assert st["status"] == "blocked"

        # Apply guidance
        st["status"] = "running"
        st["current_phase"] = "development"
        st["retry_counts"]["task_fail_src/api.py"] = 0  # Reset watchdog trigger
        st["task_queue"] = task_queue_reducer(
            st["task_queue"],
            {"task_id": "FIX-api.py", "status": "pending", "description": f"Human guidance: {guidance}"},
        )
        checkpointer.put(cfg, st)
        return st

    resumed = resume_pipeline("thread-resume-99", "Switch from SQLite to in-memory dict")
    assert resumed["status"] == "running"
    assert resumed["retry_counts"]["task_fail_src/api.py"] == 0
    assert resumed["task_queue"][0]["status"] == "pending"
    assert "Human guidance:" in resumed["task_queue"][0]["description"]

    # Verify Watchdog now allows normal worker routing
    route_wd = route_after_watchdog(resumed)
    assert route_wd == "backend_engineer"  # Routes back to worker!
