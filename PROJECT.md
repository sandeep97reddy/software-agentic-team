# Project: Autonomous AI Software Engineering Multi-Agent Team

## Architecture
The system is an industry-grade, autonomous Software Engineering multi-agent team built on LangGraph, FastAPI, and sandboxed execution environments, conforming to SWE-bench / Devin / OpenHands benchmarks.

### Key Architectural Layers:
1. **State & Persistence Layer (`src/core/`)**:
   - Pure functional, immutable state updates over `ProjectState`.
   - Dedicated custom reducers (`artifact_reducer`, `task_queue_reducer`, `dict_merge_reducer`, `adr_reducer`, `clearable_list_reducer`).
   - Pluggable persistent LangGraph checkpointer (`PostgresSaver`, `RedisSaver`, `MemorySaver`).
2. **ReAct Engine & Navigation Tools Layer (`src/tools/`, `src/core/react_engine.py`)**:
   - Autonomous ReAct loop with Thought-Action-Observation trajectories, budget limits, and stagnation checks.
   - Codebase navigation tools: `FindFilesTool` (glob matching), `GrepSearchTool` (regex search), `ASTSymbolNavigator` (AST symbol extraction), `ViewFileTool` (slice line reading).
   - Surgical editing tools: `ReplaceContentTool` (diff/hunk replacement), `WriteFileTool`.
   - In-loop diagnostics: `RunLinterTool` (Ruff/Mypy/ESLint), `RunTestTool`.
3. **Sandboxed Execution Layer (`src/tools/executor.py`, `src/tools/sandbox.py`)**:
   - Containerized and process-isolated subprocess execution.
   - Secure environment variable sanitization (credential redaction, strict allowlist).
   - Async process tree termination and timeout enforcement.
   - Automatic `PYTHONPATH` and workspace module resolution.
4. **QA & Static Analysis Layer (`src/agents/tester.py`, `src/agents/reviewer.py`, `src/agents/watchdog.py`)**:
   - AST-informed dynamic test generation across languages (Python/pytest, TS/JS/Jest).
   - Deterministic multi-tool static analysis (Ruff, Mypy, Bandit, ESLint/TypeScript).
   - Continuous test tracking on `CodeArtifact.tests_passed`.
   - Watchdog loop guard with pause/resume human-in-the-loop support.
5. **Async Distributed API & Streaming Layer (`src/api/`, `src/app.py`)**:
   - Non-blocking FastAPI execution with background jobs.
   - Real-time Server-Sent Events (SSE) and WebSocket streams for agent thought trajectories.
   - Persistent checkpoint restoration and human approval resume endpoints.

---

## Feature Inventory

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Immutable State & Pydantic Boundaries | Ensure state updates are strictly immutable, preventing in-place mutations in worker nodes. | M1 | Survey |
| 2 | Artifact Version Deduplication Reducer | Deduplicate `code_artifacts` by `file_path`, auto-incrementing versions on edits. | M1 | Survey |
| 3 | Task Queue Delta Reducer | Merge tasks by `task_id` without destructive full-list clobbering. | M1 | Survey |
| 4 | Dictionary Merge Reducer & Counter Separation | Merge dictionary keys cleanly and separate `task_failures` from `retry_counts`. | M1 | Survey |
| 5 | Hardened Clearable List Reducer | Guard `clearable_list_reducer` against `None`, type mismatches, and signal collisions. | M1 | Survey |
| 6 | ADR Deduplication Reducer | Deduplicate Architecture Decision Records by `decision_id`. | M1 | Survey |
| 7 | Persistent Checkpointer Integration | Support Postgres, Redis, and MemorySaver checkpointers in `build_graph()` with thread isolation. | M1 | Survey |
| 8 | Glob File Search Tool | Implement `FindFilesTool` with glob patterns, depth limits, and exclusion filters. | M2 | Survey |
| 9 | Regex Grep Search Tool | Implement `GrepSearchTool` for fast text/pattern searching with line numbers. | M2 | Survey |
| 10 | AST Symbol Navigator | Implement `ASTSymbolNavigator` to extract classes, functions, and signatures without full reads. | M2 | Survey |
| 11 | Slice View File Tool | Implement `ViewFileTool` for line-numbered sliced file reads. | M2 | Survey |
| 12 | Surgical Diff / Content Replacer | Implement `ReplaceContentTool` for precise search-and-replace hunk editing. | M2 | Survey |
| 13 | Autonomous ReAct Runtime | Implement `ReActEngine` with Thought-Action-Observation loops, stagnation checks, and budgets. | M2 | Survey |
| 14 | ReAct Engineer Node Upgrades | Refactor Backend and Frontend engineers to use `ReActEngine` and navigation tools. | M2 | Survey |
| 15 | Subprocess Environment Sanitization | Whitelist execution environment variables, redacting host secrets and API keys. | M3 | Survey |
| 16 | Subprocess Process-Tree Termination | Implement robust process-tree termination on timeouts across platforms. | M3 | Survey |
| 17 | Automatic PYTHONPATH Injection | Ensure subprocess execution resolves workspace modules without `ModuleNotFoundError`. | M3 | Survey |
| 18 | Containerized Sandbox Adapter | Provide isolated/sandboxed execution interface (Docker / process sandbox). | M3 | Survey |
| 19 | Dynamic Multi-Language QA Generation | Enhance QA Tester with AST-informed test generation, nested directory mirroring, and status updates. | M4 | Survey |
| 20 | Multi-Tool Deterministic Static Analysis | Integrate Ruff, Mypy, Bandit, and ESLint static analysis into Code Reviewer. | M4 | Survey |
| 21 | HITL Pause & Resume Mechanism | Implement pause/resume workflow with LangGraph interrupts and status endpoints. | M4 | Survey |
| 22 | Async Background Execution API | Refactor `POST /projects/run` to execute in async background tasks with immediate job response. | M5 | Survey |
| 23 | Real-Time SSE Streaming Endpoint | Implement Server-Sent Events endpoint streaming agent thoughts, tool calls, and node outputs. | M5 | Survey |
| 24 | Real-Time WebSocket Streaming Endpoint | Implement WebSocket endpoint for interactive real-time trace streaming. | M5 | Survey |
| 25 | Checkpoint State Query & Resume Endpoints | Implement API endpoints to query saved checkpoints and resume paused executions. | M5 | Survey |
| 26 | Comprehensive E2E Test Suite | Deliver 4-tier E2E test suite covering 100% of inventoried features. | E2E | Survey |
| 27 | Adversarial Hardening (Tier 5) | White-box stress testing achieving >= 90% codebase test coverage. | Final | Survey |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | State Architecture & Reducer Remediation | R1: Custom reducers (`artifact_reducer`, `task_queue_reducer`, `dict_merge_reducer`, `adr_reducer`, `clearable_list_reducer`), immutable node returns, checkpointer factory (`PostgresSaver`, `RedisSaver`, `MemorySaver`), graph checkpointer integration. | none | IN_PROGRESS |
| M2 | Agentic ReAct Engine & Code Navigation | R2: `FindFilesTool`, `GrepSearchTool`, `ASTSymbolNavigator`, `ViewFileTool`, `ReplaceContentTool`, `RunLinterTool`, `ReActEngine` loop, and refactored `BackendEngineer` / `FrontendEngineer`. | M1 | PLANNED |
| M3 | Sandboxed Tool Execution Layer | R3: `SubprocessExecutor` enhancements, environment variable sanitization, process-tree termination, `PYTHONPATH` workspace resolution, Docker/sandbox execution adapter. | M1 | PLANNED |
| M4 | Automated Testing & Static Analysis | R4: Dynamic test generation with nested directory support, deterministic static analysis (Ruff, Mypy, Bandit, ESLint/tsc), `CodeArtifact` telemetry, HITL pause/resume routing. | M2, M3 | PLANNED |
| M5 | Async Distributed Execution & Streaming API | R5: Async FastAPI background tasks, SSE & WebSocket event streaming, checkpoint-backed resume endpoints, thread-aware state access. | M1, M4 | PLANNED |
| E2E | E2E Testing Suite (Tiers 1-4) | Requirements-driven opaque-box test suite (Tier 1 Feature Coverage, Tier 2 Boundary/Corner, Tier 3 Cross-Feature, Tier 4 Real-World Workloads). | none | IN_PROGRESS |
| Final | E2E Verification & Adversarial Hardening | 100% E2E test pass rate + Tier 5 Adversarial Coverage Hardening reaching >= 90% test coverage. | M1, M2, M3, M4, M5, E2E | PLANNED |

---

## Interface Contracts

### 1. State Layer (`src/core/state.py`, `src/core/reducers.py`)
- `artifact_reducer(existing: list | None, new: list | dict | None) -> list[dict]`
- `task_queue_reducer(existing: list | None, new: list | dict | None) -> list[dict]`
- `dict_merge_reducer(existing: dict | None, new: dict | None) -> dict`
- `adr_reducer(existing: list | None, new: list | dict | None) -> list[dict]`
- `clearable_list_reducer(existing: list | None, new: list | str | ClearSignal | None) -> list`
- `get_checkpointer(backend: str | None = None) -> BaseCheckpointSaver`

### 2. Navigation & ReAct Layer (`src/tools/navigation.py`, `src/tools/ast_navigator.py`, `src/tools/editor.py`, `src/core/react_engine.py`)
- `FindFilesTool`: `run(pattern: str, search_dir: str = ".", max_depth: int | None = None, exclude_patterns: list[str] | None = None) -> FindFilesOutput`
- `GrepSearchTool`: `run(query: str, path_pattern: str = "**/*", case_sensitive: bool = True, max_results: int = 50) -> GrepSearchOutput`
- `ASTSymbolNavigator`: `get_outline(file_path: str) -> FileOutlineOutput`, `find_symbol(symbol_name: str, file_pattern: str = "**/*.py") -> list[ASTSymbol]`
- `ViewFileTool`: `run(file_path: str, start_line: int = 1, end_line: int | None = None, show_line_numbers: bool = True) -> ViewFileOutput`
- `ReplaceContentTool`: `run(file_path: str, target_content: str, replacement_content: str, allow_multiple: bool = False) -> ReplaceContentOutput`
- `ReActEngine`: `run(task: dict[str, Any], context: dict[str, Any]) -> ReActResult`

### 3. Execution Sandbox Layer (`src/tools/executor.py`, `src/tools/sandbox.py`)
- `SubprocessExecutor`:
  - `run_async(command: list[str], cwd: str | None = None, timeout: int = 60, env_overrides: dict[str, str] | None = None) -> ExecutionResult`
  - `run_sync(...) -> ExecutionResult` (safe in/out event loops)
  - `run_pytest(...)`, `run_linter(...)`, `run_bandit(...)`
  - Automatic `PYTHONPATH` injection of `workspace_dir` and `src/`.
  - Whitelisted environment variables.

### 4. QA & Static Analysis Layer (`src/agents/tester.py`, `src/agents/reviewer.py`)
- `tester_node(state: ProjectState) -> dict[str, Any]`: Generates and executes tests, updating `task_queue`, `code_artifacts`, and `task_failures`.
- `reviewer_node(state: ProjectState) -> dict[str, Any]`: Executes deterministic linters and security checks before LLM review, updating `task_queue`, `retry_counts`, `status`.

### 5. API Layer (`src/api/routes.py`, `src/app.py`)
- `POST /api/v1/projects/run`: Returns `{ "project_id": str, "status": "running", "thread_id": str }` immediately.
- `GET /api/v1/projects/{project_id}/status`: Returns current status and checkpoint summary.
- `GET /api/v1/projects/{project_id}/stream`: SSE endpoint streaming `TraceRecord` events.
- `WS /api/v1/projects/{project_id}/ws`: WebSocket endpoint streaming events.
- `POST /api/v1/projects/{project_id}/resume`: Resumes execution with human feedback.

---

## Code Layout

```
software-agentic-team/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── architect.py
│   │   ├── backend_engineer.py
│   │   ├── frontend_engineer.py
│   │   ├── memory.py
│   │   ├── requirement_analyzer.py
│   │   ├── reviewer.py
│   │   ├── task_planner.py
│   │   ├── tester.py
│   │   └── watchdog.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── checkpointer.py          # NEW: Checkpointer factory (Postgres, Redis, Memory)
│   │   ├── config.py
│   │   ├── graph.py
│   │   ├── middleware.py
│   │   ├── observability.py
│   │   ├── react_engine.py          # NEW: ReAct autonomous loop engine
│   │   ├── reducers.py              # NEW: Hardened custom reducers
│   │   └── state.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── ast_navigator.py         # NEW: AST symbol search & extraction
│   │   ├── diagnostics.py           # NEW: Fast linter & test runner tools
│   │   ├── editor.py                # NEW: Surgical diff & replace tools
│   │   ├── executor.py              # Upgraded: Sandboxed SubprocessExecutor
│   │   ├── filesystem.py            # Upgraded: Safe FileSystemManager
│   │   ├── git_tracker.py
│   │   ├── navigation.py            # NEW: Glob & Grep search tools
│   │   └── sandbox.py               # NEW: Sandboxed execution interfaces
│   └── app.py
├── tests/
│   ├── e2e/                         # NEW: Requirement-driven E2E tests (Tiers 1-4)
│   ├── unit/                        # Unit tests for tools, reducers, agents
│   └── test_tools_integration.py
├── docker-compose.yml
├── pyproject.toml
└── PROJECT.md
```
