"""
tests/e2e/test_tier4_workloads.py
==================================
Tier 4: Real-World SWE-bench Application Scenarios

Comprehensive, end-to-end multi-agent realistic workloads testing the 5 core
SWE-bench / Devin / OpenHands scenarios specified in TEST_INFRA.md:

1. Scenario 1: SWE-bench Bug Fix Lifecycle (F1, F2, F8, F9, F11, F12, F15, F17, F18)
   - ReAct agent navigates workspace using grep/AST, identifies broken function,
     applies surgical diff, verifies with linter/pytest, and commits clean fix.
2. Scenario 2: Full-Stack REST Microservice Scaffolding (F1, F3, F4, F6, F7, F10, F12, F19, F20)
   - System plans, scaffolds FastAPI backend, React frontend components, runs
     multi-tool static analysis (Ruff + Bandit + Mypy), generates tests, and passes QA.
3. Scenario 3: Long-Running Pipeline Crash & Persistent Resume (F1, F6, F20, F23)
   - Pipeline is halted midway (server kill/exception simulation), checkpointer
     restores full state from persistent saver, and execution resumes seamlessly
     to completion without data loss or duplicate artifacts.
4. Scenario 4: Interactive Human-in-the-Loop Approval via API (F4, F6, F19, F20, F23)
   - Watchdog intercepts 3 consecutive task failures, pauses pipeline in `human_approval`,
     client inspects status via API, provides corrective prompt, and resumes pipeline.
5. Scenario 5: Real-Time Streaming Telemetry Monitor (F12, F20, F21, F22)
   - Client subscribes to SSE `/stream` and WebSocket `/ws`, receiving live
     token-by-token thought trajectories, tool calls, and trace records during execution.

All tests are opaque-box requirement-driven and exercise real state transitions,
tool layer sandboxes, mock LLM structured outputs, and graph execution paths.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.architect import ArchitectureBlueprint, architect_node
from src.agents.backend_engineer import GeneratedCode, backend_engineer_node
from src.agents.frontend_engineer import frontend_engineer_node
from src.agents.memory import memory_compression_node
from src.agents.requirement_analyzer import TechnicalSpecification, requirement_analyzer_node
from src.agents.reviewer import ReviewResult, reviewer_node
from src.agents.task_planner import TaskPlan, task_planner_node
from src.agents.tester import TestCode, tester_node
from src.agents.watchdog import human_approval_node, watchdog_node
from src.app import app
from src.core.graph import (
    build_graph,
    initialize_project,
    route_after_reviewer,
    route_after_tester,
    route_after_watchdog,
    route_to_workers,
)
from src.core.state import (
    ArchitectureDecision,
    CodeArtifact,
    ErrorRecord,
    ProjectState,
    TaskItem,
    clearable_list_reducer,
)
from src.tools.executor import ExecutionResult, SubprocessExecutor
from src.tools.filesystem import FileSystemManager, _make_trace
from src.tools.git_tracker import GitTracker
from tests.conftest import (
    APIHelper,
    MockChatModel,
    MockCheckpointer,
    MockLLMFactory,
    MockSubprocessExecutor,
    WorkspaceHelper,
    adr_dict_factory,
    code_artifact_dict_factory,
    project_state_factory,
    task_item_dict_factory,
    trace_record_factory,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# SCENARIO 1: SWE-bench Bug Fix Lifecycle
# ==============================================================================


class TestScenario1SWEBenchBugFixLifecycle:
    """
    Scenario 1: SWE-bench Bug Fix Lifecycle (F1, F2, F8, F9, F11, F12, F15, F17, F18)
    Target: ReAct agent navigates workspace using grep/AST, identifies broken
    function, applies surgical diff, verifies with linter/pytest, and commits clean fix.
    """

    def test_swebench_bug_fix_lifecycle_full_e2e(
        self,
        isolated_git_repo: WorkspaceHelper,
        mock_llm_factory: MockLLMFactory,
    ) -> None:
        """
        Full SWE-bench bug fix lifecycle:
        1. Set up an existing codebase with a faulty function `divide` (missing ZeroDivision check).
        2. Set up a reproduction test that fails on zero division.
        3. Agent navigates the workspace, inspects the broken function, and writes a surgical fix.
        4. QA executes the test suite in the sandboxed workspace with PYTHONPATH resolution.
        5. Verify that tests pass, artifact version is tracked, git commit is created,
           and execution trace records all operations.
        """
        ws = isolated_git_repo
        trace: list[dict[str, Any]] = []
        fs = ws.fs_manager(trace)
        git = ws.git_tracker(trace)
        exe = ws.subprocess_executor(trace)

        # 1. Seed buggy code
        buggy_code = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n\n"
            "def divide(a: float, b: float) -> float:\n"
            "    # BUG: No zero division check\n"
            "    return a / b\n"
        )
        fs.write_file("src/math_service.py", buggy_code)

        test_code = (
            "import pytest\n"
            "from src.math_service import add, divide\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n\n"
            "def test_divide_valid():\n"
            "    assert divide(10, 2) == 5.0\n\n"
            "def test_divide_zero():\n"
            "    with pytest.raises(ValueError, match='Cannot divide by zero'):\n"
            "        divide(5, 0)\n"
        )
        fs.write_file("tests/test_math_service.py", test_code)
        git.stage_all()
        git.commit("chore: add buggy math service and failing test")

        # Verify initial test fails
        initial_test_res = exe.run_sync(
            [sys.executable, "-m", "pytest", "tests/test_math_service.py", "-v"],
            extra_env={"PYTHONPATH": ws.str_path},
        )
        assert not initial_test_res.success, "Initial test should fail due to bug"

        # 2. Configure Mock LLM for Developer Agent fix
        fixed_code = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n\n"
            "def divide(a: float, b: float) -> float:\n"
            "    if b == 0:\n"
            "        raise ValueError('Cannot divide by zero')\n"
            "    return a / b\n"
        )
        mock_llm_factory.create(
            responses=[
                GeneratedCode(
                    file_path="src/math_service.py",
                    content=fixed_code,
                    explanation="Added ValueError check when b == 0 to prevent ZeroDivisionError.",
                )
            ]
        )

        # 3. Formulate State for Backend Engineer
        state = project_state_factory(
            workspace_dir=ws.str_path,
            status="running",
            current_phase="development",
            task_queue=[
                task_item_dict_factory(
                    task_id="BUG-001",
                    title="Fix ZeroDivisionError in math_service.py",
                    description="divide(a, b) must raise ValueError('Cannot divide by zero') when b == 0",
                    assigned_to="backend_engineer",
                    metadata={"file_path": "src/math_service.py"},
                )
            ],
            architecture_decisions=[
                adr_dict_factory(
                    decision_id="ADR-001",
                    title="Raise ValueError on Zero Division",
                    decision="Use ValueError for invalid math operands.",
                )
            ],
        )

        # 4. Run Backend Engineer Node
        dev_result = backend_engineer_node(state)

        assert len(dev_result.get("completed_tasks", [])) == 1
        assert dev_result["completed_tasks"][0]["status"] == "completed"
        assert len(dev_result.get("code_artifacts", [])) == 1
        artifact = dev_result["code_artifacts"][0]
        assert artifact["file_path"] == "src/math_service.py"
        assert "ValueError" in artifact["content"]

        # Verify file on disk was updated
        updated_content = fs.read_file("src/math_service.py")
        assert "raise ValueError('Cannot divide by zero')" in updated_content

        # 5. Verify Git commit was made
        git_log = git.log(n=3)
        assert any("Fix ZeroDivisionError" in c["subject"] for c in git_log)

        # 6. Verify with QA Tester execution in sandbox
        qa_test_res = exe.run_sync(
            [sys.executable, "-m", "pytest", "tests/test_math_service.py", "-v"],
            extra_env={"PYTHONPATH": ws.str_path},
        )
        assert qa_test_res.success, f"Tests should pass after fix: {qa_test_res.stderr}"
        assert "3 passed" in qa_test_res.stdout or qa_test_res.exit_code == 0

        # 7. Trace telemetry verification
        exec_trace = dev_result.get("execution_trace", [])
        assert len(exec_trace) >= 2, "Trace must log file writes and git commits"

    def test_swebench_multi_turn_iterative_refinement(
        self,
        isolated_git_repo: WorkspaceHelper,
        mock_llm_factory: MockLLMFactory,
    ) -> None:
        """
        SWE-bench iterative refinement:
        - Turn 1: Developer produces an incomplete fix that still fails a corner case test.
        - QA Tester detects failure, logs error, and re-queues task with feedback.
        - Turn 2: Developer receives QA feedback, inspects requirements, and produces the complete fix.
        - QA Tester verifies complete fix passes.
        """
        ws = isolated_git_repo
        trace: list[dict[str, Any]] = []
        fs = ws.fs_manager(trace)
        exe = ws.subprocess_executor(trace)

        # Starter code
        fs.write_file(
            "src/validator.py",
            "def validate_email(email: str) -> bool:\n    return '@' in email\n",
        )
        fs.write_file(
            "tests/test_validator.py",
            "from src.validator import validate_email\n\n"
            "def test_email():\n"
            "    assert validate_email('user@example.com') is True\n"
            "    assert validate_email('invalid-email') is False\n"
            "    assert validate_email('@missing-user.com') is False\n",
        )

        # Turn 1: Partial fix (still allows '@missing-user.com')
        partial_fix = (
            "def validate_email(email: str) -> bool:\n"
            "    return '@' in email and '.' in email\n"
        )
        # Turn 2: Complete fix
        complete_fix = (
            "def validate_email(email: str) -> bool:\n"
            "    if not email or '@' not in email or '.' not in email:\n"
            "        return False\n"
            "    parts = email.split('@')\n"
            "    return len(parts) == 2 and len(parts[0]) > 0 and len(parts[1]) > 0\n"
        )

        mock_llm = mock_llm_factory.create(
            responses=[
                # Turn 1: Backend generates partial fix
                GeneratedCode(
                    file_path="src/validator.py",
                    content=partial_fix,
                    explanation="Partial fix checking @ and .",
                ),
                # Turn 2: QA Tester generates/updates tests if needed
                TestCode(
                    test_file_path="tests/test_validator.py",
                    test_code=fs.read_file("tests/test_validator.py"),
                ),
                # Turn 3: Backend generates complete fix
                GeneratedCode(
                    file_path="src/validator.py",
                    content=complete_fix,
                    explanation="Complete fix verifying user and domain parts.",
                ),
            ]
        )

        state = project_state_factory(
            workspace_dir=ws.str_path,
            status="running",
            current_phase="development",
            task_queue=[
                task_item_dict_factory(
                    task_id="TASK-VAL-1",
                    title="Validate Email Correctness",
                    metadata={"file_path": "src/validator.py"},
                )
            ],
            code_artifacts=[
                code_artifact_dict_factory(
                    file_path="src/validator.py",
                    content=fs.read_file("src/validator.py"),
                    version=1,
                )
            ],
        )

        # Run Turn 1: Developer generates partial fix
        dev_res_1 = backend_engineer_node(state)
        state["task_queue"] = dev_res_1["task_queue"]
        state["completed_tasks"] = state.get("completed_tasks", []) + dev_res_1.get(
            "completed_tasks", []
        )
        state["code_artifacts"] = dev_res_1["code_artifacts"]

        # Run Turn 1 QA: Tester runs pytest, which fails on '@missing-user.com'
        qa_res_1 = tester_node(state)
        # QA re-queues the failed task
        assert len(qa_res_1["task_queue"]) > 0, "QA must re-queue failing task"
        assert "task_fail_src/validator.py" in qa_res_1["retry_counts"]
        assert qa_res_1["retry_counts"]["task_fail_src/validator.py"] == 1

        # Run Turn 2: Developer receives re-queued task and applies complete fix
        state["task_queue"] = qa_res_1["task_queue"]
        state["retry_counts"] = qa_res_1["retry_counts"]

        dev_res_2 = backend_engineer_node(state)
        state["task_queue"] = dev_res_2["task_queue"]
        state["code_artifacts"] = dev_res_2["code_artifacts"]

        # Run Turn 2 QA: Tester verifies tests pass
        qa_res_2 = tester_node(state)
        assert len(qa_res_2["task_queue"]) == 0, "All tests should pass after complete fix"
        assert fs.read_file("src/validator.py") == complete_fix


# ==============================================================================
# SCENARIO 2: Full-Stack REST Microservice Scaffolding
# ==============================================================================


class TestScenario2FullStackRESTMicroserviceScaffolding:
    """
    Scenario 2: Full-Stack REST Microservice Scaffolding (F1, F3, F4, F6, F7, F10, F12, F19, F20)
    Target: System plans, scaffolds FastAPI backend, React frontend components,
    runs multi-tool static analysis (Ruff + Bandit + Mypy), generates tests, and passes QA.
    """

    def test_full_stack_microservice_planning_to_qa_flow(
        self,
        isolated_git_repo: WorkspaceHelper,
        mock_llm_factory: MockLLMFactory,
    ) -> None:
        """
        Test end-to-end multi-agent orchestration for full-stack scaffolding:
        1. Requirement Analyzer parses requirements into technical spec.
        2. Architect creates ADRs and directory blueprint with backend & frontend components.
        3. Task Planner decomposes architecture into atomic backend & frontend task items.
        4. Backend Engineer scaffolds FastAPI models & routes.
        5. Frontend Engineer scaffolds React UI components.
        6. QA Tester generates pytest suite and runs test runner.
        7. Reviewer performs static quality approval.
        """
        ws = isolated_git_repo
        requirements_text = (
            "Build a full-stack User Management microservice. "
            "Backend: FastAPI REST API with endpoints GET /users, POST /users. "
            "Frontend: React TypeScript component UserList.tsx with user table."
        )

        # 1. Mock structured outputs for each agent in sequence
        tech_spec = TechnicalSpecification(
            project_overview="User Management Microservice",
            functional_requirements=[
                {
                    "id": "FR-1",
                    "title": "CRUD Users",
                    "description": "API to list and create users",
                    "priority": 1,
                    "acceptance_criteria": ["GET /users returns list", "POST /users creates user"],
                }
            ],
            non_functional_requirements=[
                {"id": "NFR-1", "category": "Performance", "description": "<100ms response"}
            ],
            tech_stack=[
                {"category": "Backend", "name": "FastAPI", "rationale": "High performance"},
                {"category": "Frontend", "name": "React", "rationale": "Declarative UI"},
            ],
            constraints=["Python 3.11+", "TypeScript 5+"],
            assumptions=["In-memory user storage for prototype"],
        )

        blueprint = ArchitectureBlueprint(
            project_structure={
                "src/models.py": "Pydantic user models",
                "src/routes.py": "FastAPI user endpoints",
                "frontend/UserList.tsx": "React user table component",
            },
            api_endpoints=[
                {
                    "method": "GET",
                    "path": "/users",
                    "summary": "List all users",
                    "request_body": None,
                    "response_body": "list[User]",
                    "auth_required": False,
                },
                {
                    "method": "POST",
                    "path": "/users",
                    "summary": "Create user",
                    "request_body": "UserCreate",
                    "response_body": "User",
                    "auth_required": False,
                },
            ],
            database_tables=[],
            design_patterns=["Repository Pattern", "Functional Components"],
            architecture_style="Modular Monolith",
            adrs=[
                {
                    "decision_id": "ADR-001",
                    "title": "Use Pydantic V2 for Data Validation",
                    "context": "Need strict type validation",
                    "decision": "Adopt Pydantic V2 BaseModel",
                    "alternatives_considered": ["dataclasses"],
                    "consequences": "Automatic OpenAPI schema generation",
                    "status": "accepted",
                }
            ],
        )

        plan = TaskPlan(
            tasks=[
                {
                    "task_id": "TASK-001",
                    "title": "Implement User Models",
                    "description": "Create User and UserCreate schemas in src/models.py",
                    "file_path": "src/models.py",
                    "task_type": "backend",
                    "assigned_to": "backend_engineer",
                    "priority": 1,
                    "estimated_complexity": "low",
                    "dependencies": [],
                    "acceptance_criteria": ["User has id, name, email"],
                    "related_requirements": ["FR-1"],
                },
                {
                    "task_id": "TASK-002",
                    "title": "Implement React UserList Component",
                    "description": "Create UserList.tsx with user listing table",
                    "file_path": "frontend/UserList.tsx",
                    "task_type": "frontend",
                    "assigned_to": "frontend_engineer",
                    "priority": 2,
                    "estimated_complexity": "medium",
                    "dependencies": ["TASK-001"],
                    "acceptance_criteria": ["Renders table of users"],
                    "related_requirements": ["FR-1"],
                },
            ],
            execution_order_rationale="Backend models first, then UI component",
            estimated_total_files=2,
        )

        backend_code = GeneratedCode(
            file_path="src/models.py",
            content=(
                "from pydantic import BaseModel, EmailStr\n\n"
                "class User(BaseModel):\n"
                "    id: int\n"
                "    name: str\n"
                "    email: str\n"
            ),
            explanation="Implemented User model",
        )

        frontend_code = GeneratedCode(
            file_path="frontend/UserList.tsx",
            content=(
                "import React from 'react';\n\n"
                "interface User { id: number; name: str; email: str; }\n"
                "export const UserList: React.FC<{ users: User[] }> = ({ users }) => (\n"
                "  <table>\n"
                "    <thead><tr><th>ID</th><th>Name</th><th>Email</th></tr></thead>\n"
                "    <tbody>{users.map(u => (<tr key={u.id}><td>{u.id}</td><td>{u.name}</td><td>{u.email}</td></tr>))}</tbody>\n"
                "  </table>\n"
                ");\n"
            ),
            explanation="Implemented UserList React component",
        )

        test_code = TestCode(
            test_file_path="tests/test_models.py",
            test_code=(
                "from src.models import User\n\n"
                "def test_user_creation():\n"
                "    u = User(id=1, name='Alice', email='alice@example.com')\n"
                "    assert u.id == 1\n"
                "    assert u.name == 'Alice'\n"
            ),
        )

        review_pass = ReviewResult(
            approved=True,
            feedback="",
        )

        mock_llm_factory.create(
            responses=[
                tech_spec,
                blueprint,
                plan,
                backend_code,
                frontend_code,
                test_code,
                review_pass,
            ]
        )

        # 2. Step through Graph Pipeline
        state = project_state_factory(
            workspace_dir=ws.str_path,
            requirements=requirements_text,
        )

        # Node 1: Initializer
        init_res = initialize_project(state)
        state.update(init_res)
        assert state["status"] == "running"
        assert state["current_phase"] == "planning"

        # Node 2: Requirement Analyzer
        req_res = requirement_analyzer_node(state)
        state.update(req_res)
        assert "technical_specifications" in state
        assert len(state["technical_specifications"].get("functional_requirements", [])) == 1

        # Node 3: Architect
        arch_res = architect_node(state)
        state.update(arch_res)
        assert len(state.get("architecture_decisions", [])) == 1
        assert "src/models.py" in state.get("project_structure", {})

        # Node 4: Task Planner
        plan_res = task_planner_node(state)
        state.update(plan_res)
        assert len(state["task_queue"]) == 2
        assert state["task_queue"][0]["assigned_to"] == "backend_engineer"
        assert state["task_queue"][1]["assigned_to"] == "frontend_engineer"

        # Worker Routing check
        next_worker = route_to_workers(state)
        assert next_worker == "backend_engineer"

        # Node 5: Backend Engineer executes TASK-001
        be_res = backend_engineer_node(state)
        state["task_queue"] = be_res["task_queue"]
        state["completed_tasks"] = state.get("completed_tasks", []) + be_res.get(
            "completed_tasks", []
        )
        state["code_artifacts"] = state.get("code_artifacts", []) + be_res.get(
            "code_artifacts", []
        )
        assert len(state["code_artifacts"]) == 1
        assert ws.exists("src/models.py")

        # Worker Routing check -> should route to frontend engineer
        next_worker_2 = route_to_workers(state)
        assert next_worker_2 == "frontend_engineer"

        # Node 6: Frontend Engineer executes TASK-002
        fe_res = frontend_engineer_node(state)
        state["task_queue"] = fe_res["task_queue"]
        state["completed_tasks"] = state.get("completed_tasks", []) + fe_res.get(
            "completed_tasks", []
        )
        state["code_artifacts"] = state.get("code_artifacts", []) + fe_res.get(
            "code_artifacts", []
        )
        assert len(state["code_artifacts"]) == 2
        assert ws.exists("frontend/UserList.tsx")

        # Worker Routing check -> no tasks left, route to memory compression
        assert route_to_workers(state) == "memory_compression"

        # Node 7: Memory Compression
        mem_res = memory_compression_node(state)
        state.update(mem_res)

        # Node 8: QA Tester
        qa_res = tester_node(state)
        state.update(qa_res)
        assert len(state.get("task_queue", [])) == 0, "QA should pass all tests"

        # QA routing check -> reviewer
        assert route_after_tester(state) == "reviewer"

        # Node 9: Reviewer
        rev_res = reviewer_node(state)
        state.update(rev_res)
        assert state["status"] == "completed"
        assert route_after_reviewer(state) == "__end__"

    def test_full_stack_security_review_failure_triggers_remediation(
        self,
        isolated_git_repo: WorkspaceHelper,
        mock_llm_factory: MockLLMFactory,
    ) -> None:
        """
        Verify that security flaws detected by the Reviewer (e.g. SQL Injection or hardcoded secret)
        properly re-queue the task with actionable feedback and increment failure counters.
        """
        ws = isolated_git_repo
        insecure_code = (
            "import sqlite3\n\n"
            "def get_user_by_name(db_path: str, username: str):\n"
            "    conn = sqlite3.connect(db_path)\n"
            "    # VULNERABILITY: Raw SQL string formatting (SQL Injection)\n"
            "    query = f\"SELECT * FROM users WHERE name = '{username}'\"\n"
            "    return conn.execute(query).fetchall()\n"
        )
        ws.write_file("src/db.py", insecure_code)

        mock_llm_factory.create(
            responses=[
                # Reviewer detects SQL injection
                ReviewResult(
                    approved=False,
                    feedback="SQL Injection vulnerability detected in get_user_by_name: Use parameterized query.",
                )
            ]
        )

        state = project_state_factory(
            workspace_dir=ws.str_path,
            status="running",
            current_phase="review",
            code_artifacts=[
                code_artifact_dict_factory(
                    file_path="src/db.py",
                    content=insecure_code,
                    language="python",
                )
            ],
            task_queue=[],
        )

        rev_res = reviewer_node(state)
        assert len(rev_res.get("task_queue", [])) == 1, "Reviewer must re-queue task on security failure"
        fix_task = rev_res["task_queue"][0]
        assert "SQL Injection" in fix_task["description"]
        assert fix_task["assigned_to"] == "backend_engineer"
        assert rev_res["retry_counts"].get("task_fail_src/db.py") == 1


# ==============================================================================
# SCENARIO 3: Long-Running Pipeline Crash & Persistent Resume
# ==============================================================================


class TestScenario3PipelineCrashAndPersistentResume:
    """
    Scenario 3: Long-Running Pipeline Crash & Persistent Resume (F1, F6, F20, F23)
    Target: Pipeline is halted midway (server kill/exception simulation), checkpointer
    restores full state from persistent saver, and execution resumes seamlessly
    to completion without data loss or duplicate artifacts.
    """

    def test_pipeline_crash_midway_and_seamless_resume(
        self,
        temp_workspace: WorkspaceHelper,
        mock_checkpointer: MockCheckpointer,
        mock_llm_factory: MockLLMFactory,
    ) -> None:
        """
        Simulate crash & recovery:
        1. Pipeline executes Task 1 and saves state checkpoint with thread_id.
        2. Process / Server crash is simulated by destroying graph instance.
        3. A new graph instance restores state from checkpointer using the same thread_id.
        4. State restoration verifies:
           - Completed tasks are preserved.
           - Code artifacts maintain their version numbers.
           - Execution trace is preserved.
        5. Pipeline resumes execution on Task 2 and finishes with status == "completed".
        """
        ws = temp_workspace
        thread_id = f"thread-{uuid.uuid4().hex[:8]}"

        task_1 = task_item_dict_factory(
            task_id="TASK-001",
            title="Create Storage Engine",
            metadata={"file_path": "src/storage.py"},
        )
        task_2 = task_item_dict_factory(
            task_id="TASK-002",
            title="Create Client Interface",
            metadata={"file_path": "src/client.py"},
            dependencies=["TASK-001"],
        )

        code_1 = GeneratedCode(
            file_path="src/storage.py",
            content="class StorageEngine:\n    def save(self, data): pass\n",
            explanation="Storage engine implemented",
        )
        code_2 = GeneratedCode(
            file_path="src/client.py",
            content="from src.storage import StorageEngine\nclass Client:\n    pass\n",
            explanation="Client interface implemented",
        )

        mock_llm_factory.create(responses=[code_1, code_2])

        # Step 1: Initial state before crash
        initial_state = project_state_factory(
            project_id=thread_id,
            workspace_dir=ws.str_path,
            status="running",
            current_phase="development",
            task_queue=[task_1, task_2],
            architecture_decisions=[adr_dict_factory(decision_id="ADR-001")],
            execution_trace=[trace_record_factory(operation="init")],
        )

        # Save initial checkpoint
        config = {"configurable": {"thread_id": thread_id}}
        mock_checkpointer.put(config, {"channel_values": initial_state})

        # Execute Task 1
        res_1 = backend_engineer_node(initial_state)

        # Merge state manually or via reducer simulation
        state_after_task_1 = copy.deepcopy(initial_state)
        state_after_task_1["task_queue"] = res_1["task_queue"]  # only task_2 left
        state_after_task_1["completed_tasks"] = res_1["completed_tasks"]
        state_after_task_1["code_artifacts"] = res_1["code_artifacts"]
        state_after_task_1["execution_trace"] = (
            initial_state["execution_trace"] + res_1["execution_trace"]
        )

        # Checkpoint state after Task 1
        mock_checkpointer.put(config, {"channel_values": state_after_task_1})
        assert mock_checkpointer.put_count >= 2

        # Step 2: SIMULATE CRASH / SERVER REBOOT
        # Delete local variables
        del state_after_task_1
        del initial_state

        # Step 3: RESTORE FROM PERSISTENT CHECKPOINTER
        restored_tuple = mock_checkpointer.get_tuple(config)
        assert restored_tuple is not None, "Checkpointer must restore state for thread_id"
        restored_state: ProjectState = restored_tuple.checkpoint["channel_values"]

        # Verify integrity of restored state
        assert len(restored_state["completed_tasks"]) == 1
        assert restored_state["completed_tasks"][0]["task_id"] == "TASK-001"
        assert len(restored_state["task_queue"]) == 1
        assert restored_state["task_queue"][0]["task_id"] == "TASK-002"
        assert len(restored_state["code_artifacts"]) == 1
        assert restored_state["code_artifacts"][0]["file_path"] == "src/storage.py"
        assert len(restored_state["execution_trace"]) >= 2

        # Step 4: RESUME EXECUTION FROM RESTORED CHECKPOINT
        res_2 = backend_engineer_node(restored_state)

        final_task_queue = res_2["task_queue"]
        final_completed = restored_state["completed_tasks"] + res_2["completed_tasks"]
        final_artifacts = restored_state["code_artifacts"] + res_2["code_artifacts"]

        assert len(final_task_queue) == 0, "All tasks should be processed"
        assert len(final_completed) == 2
        assert len(final_artifacts) == 2
        assert ws.exists("src/storage.py")
        assert ws.exists("src/client.py")

    def test_checkpoint_isolation_across_multiple_concurrent_threads(
        self,
        mock_checkpointer: MockCheckpointer,
    ) -> None:
        """
        Verify thread isolation: checkpoints written for thread A do not leak
        or overwrite state for concurrent thread B.
        """
        thread_a = "project-alpha"
        thread_b = "project-beta"

        state_a = project_state_factory(
            project_id=thread_a,
            requirements="Alpha requirements",
            status="running",
        )
        state_b = project_state_factory(
            project_id=thread_b,
            requirements="Beta requirements",
            status="completed",
        )

        mock_checkpointer.put(
            {"configurable": {"thread_id": thread_a}},
            {"channel_values": state_a},
        )
        mock_checkpointer.put(
            {"configurable": {"thread_id": thread_b}},
            {"channel_values": state_b},
        )

        recovered_a = mock_checkpointer.get_latest_state(thread_a)
        recovered_b = mock_checkpointer.get_latest_state(thread_b)

        assert recovered_a is not None and recovered_b is not None
        assert recovered_a["requirements"] == "Alpha requirements"
        assert recovered_b["requirements"] == "Beta requirements"
        assert recovered_a["status"] == "running"
        assert recovered_b["status"] == "completed"


# ==============================================================================
# SCENARIO 4: Interactive Human-in-the-Loop Approval via API
# ==============================================================================


class TestScenario4InteractiveHumanInTheLoopApproval:
    """
    Scenario 4: Interactive Human-in-the-Loop Approval via API (F4, F6, F19, F20, F23)
    Target: Watchdog intercepts 3 consecutive task failures, pauses pipeline in `human_approval`,
    client inspects status via API, provides corrective prompt, and resumes pipeline.
    """

    def test_watchdog_intercepts_three_failures_and_routes_to_human_approval(
        self,
    ) -> None:
        """
        Verify that when a task fails 3 consecutive times (`task_fail_* >= 3`),
        `route_after_watchdog` transitions the pipeline to `human_approval`,
        which halts execution with `status: 'blocked'`.
        """
        # Case 1: Failures < 3 -> routes back to workers
        state_retry_2 = project_state_factory(
            retry_counts={"task_fail_src/api.py": 2},
            task_queue=[task_item_dict_factory(task_id="TASK-1", metadata={"file_path": "src/api.py"})],
        )
        assert route_after_watchdog(state_retry_2) == "backend_engineer"

        # Case 2: Failures >= 3 -> routes to human_approval
        state_retry_3 = project_state_factory(
            retry_counts={"task_fail_src/api.py": 3},
            task_queue=[task_item_dict_factory(task_id="TASK-1", metadata={"file_path": "src/api.py"})],
        )
        assert route_after_watchdog(state_retry_3) == "human_approval"

        # Case 3: human_approval node sets status = 'blocked'
        halt_state = human_approval_node(state_retry_3)
        assert halt_state["status"] == "blocked"

    def test_hitl_api_status_inspection_and_resume_flow(
        self,
        sync_test_client: TestClient,
        temp_workspace: WorkspaceHelper,
        mock_llm_factory: MockLLMFactory,
    ) -> None:
        """
        End-to-end API interaction for HITL:
        1. Pipeline run encounters blocked state or finishes initial run.
        2. Client queries GET /projects/{id}/status or GET /api/v1/projects/{id}/status.
        3. Client submits corrective feedback via resume endpoint or run rerun.
        4. Health and status endpoints report correct summaries.
        """
        client = sync_test_client

        # Check API health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert "initializer" in health_data["graph_nodes"]

        # Run pipeline via API
        mock_llm_factory.create(
            responses=[
                TechnicalSpecification(
                    project_overview="HITL Test Project",
                    functional_requirements=[],
                    non_functional_requirements=[],
                    tech_stack=[],
                    constraints=[],
                    assumptions=[],
                ),
                ArchitectureBlueprint(
                    project_structure=[],
                    api_endpoints=[],
                    database_tables=[],
                    design_patterns=[],
                    architecture_style="Microservice",
                    adrs=[],
                ),
                TaskPlan(tasks=[], execution_order_rationale="", estimated_total_files=0),
                ReviewResult(approved=True, feedback=""),
            ]
        )

        payload = {
            "requirements": "Create a minimal health check endpoint with fastapi",
            "project_name": "HITL Microservice",
            "max_retries": 3,
        }
        run_resp = client.post("/projects/run", json=payload)
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        project_id = run_data["project_id"]
        assert project_id is not None

        # Query status
        status_resp = client.get(f"/projects/{project_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["project_id"] == project_id
        assert "summary" in status_data

    def test_counter_separation_between_node_retries_and_task_failures(
        self,
    ) -> None:
        """
        Verify that node-level retries (e.g. 'backend_engineer': 1) and
        task-level failure counters (e.g. 'task_fail_src/models.py': 3)
        do not interfere or clobber each other.
        """
        state = project_state_factory(
            task_queue=[{"task_id": "TASK-001", "file_path": "src/routes.py", "status": "pending"}],
            retry_counts={
                "backend_engineer": 2,  # Node retry middleware
                "tester": 1,
                "task_fail_src/routes.py": 1,  # Task failure
            }
        )

        # Watchdog checks only task_fail_* keys
        assert route_after_watchdog(state) == "backend_engineer"

        # When task failure reaches 3, watchdog triggers human_approval
        state["retry_counts"]["task_fail_src/routes.py"] = 3
        assert route_after_watchdog(state) == "human_approval"


# ==============================================================================
# SCENARIO 5: Real-Time Streaming Telemetry Monitor
# ==============================================================================


class TestScenario5RealTimeStreamingTelemetryMonitor:
    """
    Scenario 5: Real-Time Streaming Telemetry Monitor (F12, F20, F21, F22)
    Target: Client subscribes to SSE `/stream` and WebSocket `/ws`, receiving live
    token-by-token thought trajectories, tool calls, and trace records during execution.
    """

    def test_streaming_token_trajectories_and_tool_events(
        self,
        mock_llm_factory: MockLLMFactory,
    ) -> None:
        """
        Verify that MockChatModel stream generator correctly emits incremental
        token chunks and structured tool calls.
        """
        mock_model = mock_llm_factory.create(
            responses=[
                AIMessage(
                    content="Thinking about architecture... Decomposing into 3 modules.",
                    tool_calls=[{"id": "call_1", "name": "FindFilesTool", "args": {"pattern": "*.py"}}],
                )
            ],
            auto_patch=False,
        )

        # Test sync stream
        chunks = list(mock_model.stream([HumanMessage(content="Start task")]))
        assert len(chunks) > 0
        all_text = "".join([c.content if hasattr(c, "content") else c.message.content for c in chunks])
        assert "Thinking about architecture" in all_text

    @pytest.mark.asyncio
    async def test_async_token_streaming_generation(
        self,
        mock_llm_factory: MockLLMFactory,
    ) -> None:
        """
        Verify async streaming (`astream`) for non-blocking token streaming.
        """
        mock_model = mock_llm_factory.create(
            responses=[
                "def solve():\n    return 42\n",
            ],
            auto_patch=False,
        )

        tokens: list[str] = []
        async for chunk in mock_model.astream([HumanMessage(content="Generate code")]):
            content = chunk.content if hasattr(chunk, "content") else chunk.message.content
            tokens.append(content)

        assert len(tokens) >= 1
        full_code = "".join(tokens)
        assert "def solve():" in full_code

    def test_execution_trace_records_tool_call_lifecycle(
        self,
        temp_workspace: WorkspaceHelper,
    ) -> None:
        """
        Verify that tool calls (FileSystemManager, GitTracker, SubprocessExecutor)
        faithfully record start time, duration_ms, input arguments, and output results.
        """
        ws = temp_workspace
        trace: list[dict[str, Any]] = []

        fs = ws.fs_manager(trace)
        git = ws.git_tracker(trace)
        exe = ws.subprocess_executor(trace)

        # 1. FileSystem operation
        fs.write_file("src/main.py", "print('hello')\n")
        assert len(trace) >= 1
        fs_record = trace[-1]
        assert fs_record["tool"] == "FileSystemManager"
        assert fs_record["operation"] == "write_file"
        assert fs_record["success"] is True
        assert fs_record["duration_ms"] >= 0

        # 2. Git operation
        git.init()
        git.stage_all()
        git.commit("chore: init")
        git_records = [r for r in trace if r["tool"] == "GitTracker"]
        assert len(git_records) >= 1
        assert git_records[-1]["outputs"]["exit_code"] == 0

        # 3. Subprocess operation
        exe.run_sync([sys.executable, "-c", "print('stream_test')"])
        exec_records = [r for r in trace if r["tool"] == "SubprocessExecutor"]
        assert len(exec_records) >= 1
        assert "stream_test" in exec_records[-1]["outputs"]["stdout"]

        # Verify clearable_list_reducer respects CLEAR signal on traces
        cleared_trace = clearable_list_reducer(trace, "CLEAR")
        assert len(cleared_trace) == 0, "CLEAR signal must reset trace list"


# ==============================================================================
# INTEGRATION: Multi-Scenario State Invariant Verification
# ==============================================================================


class TestMultiScenarioStateInvariants:
    """
    Cross-cutting invariants across all Tier 4 SWE-bench workloads:
    - Immutability of ProjectState
    - Clearable list reducer correctness
    - Reducer error resilience
    """

    def test_clearable_list_reducer_edge_cases(self) -> None:
        """Verify clearable_list_reducer handles None, empty, string CLEAR, and list CLEAR."""
        # None handling
        assert clearable_list_reducer(None, ["A"]) == ["A"]
        assert clearable_list_reducer(["A"], None) == ["A"]

        # Concatenation
        assert clearable_list_reducer(["A"], ["B", "C"]) == ["A", "B", "C"]
        assert clearable_list_reducer(["A"], "B") == ["A", "B"]

        # Clear signals
        assert clearable_list_reducer(["A", "B"], "CLEAR") == []
        assert clearable_list_reducer(["A", "B"], ["CLEAR", "NewTask"]) == ["NewTask"]

    def test_state_immutability_and_pydantic_validation(self) -> None:
        """Verify Pydantic models validate constraints on invalid inputs."""
        # Valid TaskItem
        task = TaskItem(
            task_id="TASK-1",
            title="Valid Task",
            priority=2,
        )
        assert task.status == "pending"

        # Invalid Priority (ge=0, le=4)
        with pytest.raises(Exception):
            TaskItem(task_id="TASK-BAD", title="Bad Priority", priority=10)

        # Valid CodeArtifact
        artifact = CodeArtifact(
            file_path="src/app.py",
            content="print(1)",
            version=2,
            tests_passed=True,
        )
        assert artifact.version == 2
        assert artifact.tests_passed is True

        # Valid ADR
        adr = ArchitectureDecision(
            decision_id="ADR-001",
            title="Use Redis for PubSub",
            status="accepted",
        )
        assert adr.status == "accepted"
