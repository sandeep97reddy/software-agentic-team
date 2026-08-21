# 🚀 Autonomous AI Software Engineering Team
> **An Enterprise-Grade, Multi-Agent SWE Orchestration System Powered by LangGraph, ReAct Code Navigation, Deterministic Sandboxing, and Persistent State Machines.**

[![CI / Test Suite](https://img.shields.io/badge/Tests-325%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![Architecture](https://img.shields.io/badge/Orchestrator-LangGraph%20StateGraph-orange.svg)](src/core/graph.py)
[![Security](https://img.shields.io/badge/Sandbox-Isolated%20Subprocess%20%2B%20FS%20Jail-red.svg)](src/tools/executor.py)
[![API](https://img.shields.io/badge/API-FastAPI%20Async%20%2B%20SSE%20%2F%20WS-teal.svg)](src/api/routes.py)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [Staff-Level Architectural Principles](#-staff-level-architectural-principles)
- [System Architecture & State Flow](#-system-architecture--state-flow)
- [Core Feature Highlights](#-core-feature-highlights)
  - [1. Conflict-Free State Reducers & Immutability](#1-conflict-free-state-reducers--immutability)
  - [2. Multi-Backend Persistent Checkpointer](#2-multi-backend-persistent-checkpointer)
  - [3. Autonomous ReAct Engine & AST Code Navigation](#3-autonomous-react-engine--ast-code-navigation)
  - [4. Sandboxed Tool Execution & Secret Sanitization](#4-sandboxed-tool-execution--secret-sanitization)
  - [5. Automated QA, Dynamic Test Generation & Static Security Analysis](#5-automated-qa-dynamic-test-generation--static-security-analysis)
  - [6. Infinite Loop Watchdog & Human-in-the-Loop (HITL)](#6-infinite-loop-watchdog--human-in-the-loop-hitl)
  - [7. Async Distributed Execution & Real-Time SSE/WebSocket Streaming](#7-async-distributed-execution--real-time-ssewebsocket-streaming)
- [Agent Profiles & Node Contracts](#-agent-profiles--node-contracts)
- [4-Tier Enterprise Verification Harness (325 Tests)](#-4-tier-enterprise-verification-harness-325-tests)
- [Quick Start Guide](#-quick-start-guide)
- [Configuration & Environment](#-configuration--environment)
- [Production Deployment](#-production-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏛 Executive Overview

The **Autonomous AI Software Engineering Team** is a production-grade multi-agent software engineering platform built to autonomously take high-level natural language requirements and deliver verified, tested, secure, and production-ready codebases.

Unlike monolithic single-prompt agent systems that suffer from token exhaustion, hallucination cascades, and catastrophic state overwrites, this platform employs a **strictly decoupled, pure-functional multi-agent topology** governed by LangGraph. Each specialized agent operates with **bounded autonomy**, interacting through typed state mutations, deterministic file-system sandboxes, AST navigation tools, and automated testing feedback loops.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 User Natural Language Req               │
                  └────────────────────────────┬────────────────────────────┘
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │               Initializer & Workspace Jail              │
                  └────────────────────────────┬────────────────────────────┘
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │             Requirement Analyzer (Product Mgr)          │
                  └────────────────────────────┬────────────────────────────┘
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │              Architect Agent (System Designer)          │
                  └────────────────────────────┬────────────────────────────┘
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │             Task Planner (Linear DAG Scheduler)         │
                  └────────────────────────────┬────────────────────────────┘
                                               ▼
                                  ┌─────────────────────────┐
                    ┌────────────►│  Conditional Task Router│◄────────────┐
                    │             └────────────┬────────────┘             │
                    │                          │                         │
            (Backend Task)             (Frontend Task)             (Remaining)
                    │                          │                         │
                    ▼                          ▼                         │
         ┌─────────────────────┐    ┌─────────────────────┐              │
         │  Backend Engineer   │    │  Frontend Engineer  │              │
         │ (FastAPI / Py / ReAct)│  │ (React / TS / ReAct)│              │
         └──────────┬──────────┘    └──────────┬──────────┘              │
                    │                          │                         │
                    └──────────────────────────┴─────────────────────────┘
                                               │
                                      (All Tasks Drained)
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │               Memory Compression Engine                 │
                  └────────────────────────────┬────────────────────────────┘
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │           QA Tester (Dynamic Pytest Runner)             │
                  └────────────────────────────┬────────────────────────────┘
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │          Code Reviewer (AST Security & Linters)         │
                  └──────────────┬───────────────────────────┬──────────────┘
                                 │                           │
                            (Failures)                    (Passed)
                                 ▼                           ▼
                  ┌──────────────────────────────┐    ┌──────────────┐
                  │ Watchdog & HITL Halt Guard   │    │  END / Done  │
                  └──────────────────────────────┘    └──────────────┘
```

---

## 💎 Staff-Level Architectural Principles

1. **Agents as Pure Stateless Functions**:
   Agents do not retain internal state across graph iterations. They receive a frozen snapshot of `ProjectState`, execute deterministic LLM invocations or sandboxed side-effects, and return atomic state mutation dictionaries.
2. **Conflict-Free State Aggregation**:
   State fields are annotated with mathematical reducers (CRDT-like state reconciliation), preventing parallel branch clobbering, duplicate artifact generation, and dictionary key collisions.
3. **Principle of Least Privilege & Sandbox Isolation**:
   Subprocess commands and file I/O operations are strictly bounded inside isolated temporary workspace roots with credential sanitization and path traversal denial.
4. **Deterministic Outer-Loop Verification**:
   No code is committed to version control without automated pytest verification, AST security auditing, and stagnation checks.
5. **Point-in-Time Recovery & Thread Isolation**:
   Every state transition is checkpointed with thread IDs to Redis, PostgreSQL, or Memory, enabling seamless resumption after pipeline crashes or human-in-the-loop pauses.

---

## ⚡ Core Feature Highlights

### 1. Conflict-Free State Reducers & Immutability
Located in [`src/core/reducers.py`](src/core/reducers.py), custom state reducers guarantee mathematically sound mutations:
- `artifact_reducer`: Deduplicates artifacts by canonicalized relative path (`src/utils.py` vs `src/./utils.py`), automatically increments version counters on content modifications, and resets test statuses on code updates.
- `task_queue_reducer`: Merges task updates atomically by `task_id`, maintaining FIFO execution order while preserving dependency graphs.
- `clearable_list_reducer`: Provides append-only logs while supporting atomic trace truncations when receiving `ClearSignal` or `"CLEAR"`.
- `dict_merge_reducer` & `adr_reducer`: Conflict-free merging for retry counters, failure trackers, and Architectural Decision Records.

### 2. Multi-Backend Persistent Checkpointer
Located in [`src/core/checkpointer.py`](src/core/checkpointer.py):
- **Dynamic Backend Selection**: Supports `MemorySaver`, `PostgresSaver`, and `RedisSaver` configured via environment variables.
- **Fail-Safe Fallback**: Gracefully degrades to high-performance in-memory checkpointing if external databases are temporarily unavailable.
- **Full State Replay**: Enables audit trails, point-in-time branch rewinds, and crash-resilient reboots without losing in-flight task progress.

### 3. Autonomous ReAct Engine & AST Code Navigation
Located in [`src/tools/`](src/tools/):
- **`ASTSymbolNavigator`**: Parses Python source trees via standard library `ast`, extracting class hierarchies, method signatures, parameters, docstrings, and line bounds without executing code.
- **`ReplaceContentTool`**: Performs surgical, collision-checked search-and-replace edits with diff previews.
- **`GrepSearchTool` & `FindFilesTool`**: Fast regex and glob searches with directory traversal guards.
- **ReAct Loop Integration**: Engineers iteratively explore, locate symbols, apply diffs, and inspect syntax errors before committing.

### 4. Sandboxed Tool Execution & Secret Sanitization
Located in [`src/tools/executor.py`](src/tools/executor.py) and [`src/tools/filesystem.py`](src/tools/filesystem.py):
- **Path Traversal Protection**: Rejects all directory escape attempts (`../../`, `/etc/passwd`, `C:\Windows`) with `PermissionError`.
- **Host Secret Sanitization**: Automatically strips host API keys (`*_API_KEY`, `*SECRET*`, `*PASS*`, `*TOKEN*`) from subprocess environments so untrusted code cannot exfiltrate host credentials.
- **UTF-8 & Tooling Setup**: Injects `PYTHONIOENCODING="utf-8"`, `PYTHONUTF8="1"`, and automatic workspace `PYTHONPATH` resolution.
- **Command Allowlisting & Timeout Governance**: Strictly limits executed binaries (`pytest`, `ruff`, `black`, `mypy`, `git`) with process tree termination on timeout.

### 5. Automated QA, Dynamic Test Generation & Static Security Analysis
Located in [`src/agents/tester.py`](src/agents/tester.py) and [`src/agents/reviewer.py`](src/agents/reviewer.py):
- **Dynamic Pytest Generation**: Automatically creates comprehensive test suites for untested Python artifacts.
- **Test Telemetry & Re-queueing**: Captures stdout/stderr from failed pytest runs, wraps the failure context into a `FIX-<filename>` task, and places it back in the task queue for engineer remediation.
- **Multi-Tool Static Auditing**: Scans artifacts for SQL injection, unsanitized subprocess execution, and cross-site scripting (XSS) patterns.

### 6. Infinite Loop Watchdog & Human-in-the-Loop (HITL)
Located in [`src/agents/watchdog.py`](src/agents/watchdog.py) and [`src/core/graph.py`](src/core/graph.py):
- **Stagnation Detection**: Identifies zero-byte diffs where an agent generates identical logic across consecutive retries.
- **Watchdog Counter**: Tracks isolated task failure counts. If any task fails $\ge 3$ times, the graph transitions to `human_approval` (`status="blocked"`).
- **HITL Resume API**: Allows human operators to inspect the blocked checkpoint, supply clarifying feedback or code overrides, and unblock the pipeline via `/api/v1/projects/{id}/resume`.

### 7. Async Distributed Execution & Real-Time SSE/WebSocket Streaming
Located in [`src/api/routes.py`](src/api/routes.py) and [`src/app.py`](src/app.py):
- **Non-Blocking Background Dispatch**: Dispatches project runs into asynchronous worker tasks without blocking FastAPI request threads.
- **Server-Sent Events (SSE)**: Streams incremental LLM thoughts, tool calls, and state diffs via `GET /api/v1/projects/{id}/stream`.
- **Bidirectional WebSockets**: Interactive WebSocket channels via `/api/v1/ws/{id}` for live telemetry streaming, breakpoint pausing, and manual input injection.

---

## 🤖 Agent Profiles & Node Contracts

| Agent Node | Core Role | Bounded Autonomy & Contract |
| :--- | :--- | :--- |
| **Initializer** | System Bootstrapper | Scaffolds sandboxed workspace directory, writes `.gitkeep`, initializes local git repo, and creates active branch. |
| **Requirement Analyzer** | Product Manager | Structures ambiguous user prompts into formal `TechnicalSpecification` with functional/non-functional specs. |
| **Architect Agent** | System Designer | Produces Architectural Decision Records (ADRs) and global `project_structure`. Strictly forbidden from generating code files. |
| **Task Planner** | Project Manager | Decomposes architecture into atomic, linear `TaskItem` queues with explicit dependency mapping. |
| **Backend Engineer** | Python/FastAPI Developer | Pops backend tasks from queue, leverages ReAct navigation tools, applies diffs, and commits code. |
| **Frontend Engineer** | React/TypeScript Developer | Pops UI tasks from queue, scaffolds React components and CSS, and commits code. Forbidden from modifying backend models. |
| **Memory Compression** | Context Optimizer | Summarizes long trace logs to keep LLM context slim, passing `ClearSignal` to truncate raw execution traces. |
| **QA Tester** | Validation Specialist | Dynamically writes and executes pytest suites with automatic workspace `PYTHONPATH` resolution. |
| **Code Reviewer** | Security & Quality Auditor | Performs AST static analysis to reject vulnerabilities (SQLi, shell injection) back to the task queue. |
| **Watchdog Node** | Infinite Loop Guard | Monitors per-task failure counts. Halts runaway retry loops at threshold ($\ge 3$) and diverts to `human_approval`. |
| **Human Approval** | Halt Waypoint | Safe halt state that pauses the LangGraph execution until human guidance or approval is submitted. |

---

## 🧪 4-Tier Enterprise Verification Harness (325 Tests)

The system includes a comprehensive, 4-tier test harness with **325 automated tests** executing in under **40 seconds**:

```
tests/
├── unit/                         # 51 Tests: Isolated Reducer & Checkpointer Unit Tests
│   ├── test_reducers.py         # ClearSignal, artifact_reducer, task_queue_reducer, etc.
│   └── test_checkpointer.py     # Memory, Postgres, and Redis checkpointer backends
└── e2e/                          # 274 Tests: 4-Tier Integration and E2E Harness
    ├── conftest.py               # MockLLMFactory, WorkspaceHelper, and test fixtures
    ├── test_tier1_features.py    # 115 Tests: Tier 1 Individual Feature Verifications (F1–F25)
    ├── test_tier2_boundaries.py  # 115 Tests: Tier 2 Edge Cases, Boundary Limits & Error Handling
    ├── test_tier3_combinations.py# 30 Tests: Tier 3 Cross-Feature & Multi-Agent Combinations
    └── test_tier4_workloads.py   # 14 Tests: Tier 4 SWE-bench & Full-Stack Workload Scenarios
```

### Verification Command & Output
```bash
python -m pytest tests/unit/ tests/e2e/ -v
```

```
======================= 325 passed, 1 warning in 36.68s =======================
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+ (Fully tested on Python 3.11, 3.12, 3.13, and 3.14)
- Git installed on host PATH
- (Optional) PostgreSQL / Redis for production persistence

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/sandeep97reddy/software-agentic-team.git
cd software-agentic-team
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 2. Install Package & Dependencies
```bash
pip install -e .
```

### 3. Configure Environment
```bash
cp .env.example .env
```
Edit `.env` to configure your LLM provider and settings:
```ini
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o
TEMPERATURE=0.2
CHECKPOINTER_BACKEND=memory
LANGSMITH_API_KEY=
LANGCHAIN_TRACING_V2=false
```

### 4. Run the Full Test Suite
```bash
python -m pytest tests/unit/ tests/e2e/ -v
```

### 5. Launch the API Server
```bash
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Execute a Project via REST API
```bash
curl -X POST http://localhost:8000/api/v1/projects/run \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": "Build a complete RESTful Note-taking API with SQLite and FastAPI, with full pytest test coverage.",
    "project_name": "NoteTakingService",
    "max_retries": 3
  }'
```

---

## 🔧 Production Deployment

### Docker Deployment
```bash
docker compose up -d --build
```
This starts:
- **FastAPI Orchestrator** on port `8000`
- **PostgreSQL Persistence Store** on port `5432`
- **Redis Event & Task Broker** on port `6379`

### Health & Monitoring Endpoints
- `GET /health` or `GET /api/v1/health` — System status and compiled graph node registry.
- `GET /api/v1/projects/{project_id}/status` — Current phase, task queue summary, and completed artifacts.
- `GET /api/v1/projects/{project_id}/stream` — Real-time Server-Sent Events (SSE) thought trajectory stream.
- `POST /api/v1/projects/{project_id}/resume` — Resume a paused or blocked pipeline with human feedback.

---

## 🤝 Contributing

We welcome contributions! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_STANDARDS.md](CODE_STANDARDS.md) before submitting pull requests.

All PRs must adhere to:
1. **Pure Stateless Node Contracts**: Agent nodes must be pure functions decorated with `@retry_middleware`.
2. **Reducer Annotations**: Any collection fields in `ProjectState` must use custom reducers.
3. **100% Test Pass Rate**: Run `python -m pytest tests/unit/ tests/e2e/ -v` to ensure all 325 tests pass.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
