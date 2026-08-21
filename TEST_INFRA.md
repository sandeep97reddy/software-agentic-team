# E2E Test Infra: Autonomous AI Software Engineering Team

## Test Philosophy
- Opaque-box, requirement-driven testing derived from `ORIGINAL_REQUEST.md`.
- No reliance on internal implementation details; exercise the public APIs, agent ReAct loops, tool execution layer, and StateGraph contracts.
- Systematic 4-tier test case design methodology:
  1. **Tier 1: Feature Coverage** (>=5 test cases per feature across all 5 requirement areas R1-R5).
  2. **Tier 2: Boundary & Corner Cases** (>=5 test cases per feature covering edge cases, timeouts, malformed inputs, process crashes, large payloads).
  3. **Tier 3: Cross-Feature Combinations** (Pairwise coverage of feature interactions: ReAct + Sandboxing, State deduplication + Checkpointing, Async Streaming + HITL pause/resume).
  4. **Tier 4: Real-World Application Scenarios** (Full multi-turn SWE-bench-style workloads: full-stack repository generation, bug reproduction and fix, security vulnerability remediation, streaming monitoring).

---

## Feature Inventory & Test Coverage Matrix

| # | Feature | Requirement | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|-------------|:------:|:------:|:------:|:------:|
| 1 | Immutable State & Pydantic Boundaries | R1 | 5 | 5 | ✓ | ✓ |
| 2 | Artifact Version Deduplication Reducer | R1 | 5 | 5 | ✓ | ✓ |
| 3 | Task Queue Delta Reducer | R1 | 5 | 5 | ✓ | ✓ |
| 4 | Dictionary Merge Reducer & Counter Separation | R1 | 5 | 5 | ✓ | ✓ |
| 5 | Hardened Clearable List Reducer | R1 | 5 | 5 | ✓ | ✓ |
| 6 | Persistent Checkpointer Integration | R1 | 5 | 5 | ✓ | ✓ |
| 7 | Glob File Search Tool | R2 | 5 | 5 | ✓ | ✓ |
| 8 | Regex Grep Search Tool | R2 | 5 | 5 | ✓ | ✓ |
| 9 | AST Symbol Navigator | R2 | 5 | 5 | ✓ | ✓ |
| 10 | Slice View File Tool | R2 | 5 | 5 | ✓ | ✓ |
| 11 | Surgical Diff / Content Replacer | R2 | 5 | 5 | ✓ | ✓ |
| 12 | Autonomous ReAct Runtime | R2 | 5 | 5 | ✓ | ✓ |
| 13 | Subprocess Environment Sanitization | R3 | 5 | 5 | ✓ | ✓ |
| 14 | Subprocess Process-Tree Termination | R3 | 5 | 5 | ✓ | ✓ |
| 15 | Automatic PYTHONPATH Injection | R3 | 5 | 5 | ✓ | ✓ |
| 16 | Sandboxed Execution Adapter | R3 | 5 | 5 | ✓ | ✓ |
| 17 | Dynamic Multi-Language QA Generation | R4 | 5 | 5 | ✓ | ✓ |
| 18 | Multi-Tool Deterministic Static Analysis | R4 | 5 | 5 | ✓ | ✓ |
| 19 | HITL Pause & Resume Mechanism | R4 | 5 | 5 | ✓ | ✓ |
| 20 | Async Background Execution API | R5 | 5 | 5 | ✓ | ✓ |
| 21 | Real-Time SSE Streaming Endpoint | R5 | 5 | 5 | ✓ | ✓ |
| 22 | Real-Time WebSocket Streaming Endpoint | R5 | 5 | 5 | ✓ | ✓ |
| 23 | Checkpoint State Query & Resume Endpoints | R5 | 5 | 5 | ✓ | ✓ |

---

## Test Architecture

### Test Runner
- **Command:** `pytest tests/ -v --tb=short --cov=src --cov-report=term-missing`
- **Pass/Fail Semantics:** 100% of test cases must pass with exit code 0.
- **Coverage Goal:** Total codebase coverage >= 90%.

### Directory Layout
```
tests/
├── e2e/
│   ├── test_tier1_features.py       # Tier 1: Feature coverage
│   ├── test_tier2_boundaries.py     # Tier 2: Boundary and corner cases
│   ├── test_tier3_combinations.py   # Tier 3: Pairwise cross-feature interactions
│   └── test_tier4_workloads.py      # Tier 4: Real-world SWE application workloads
├── unit/
│   ├── test_reducers.py
│   ├── test_checkpointer.py
│   ├── test_navigation_tools.py
│   ├── test_ast_navigator.py
│   ├── test_editor_tools.py
│   ├── test_react_engine.py
│   ├── test_executor_sandbox.py
│   ├── test_tester_agent.py
│   ├── test_reviewer_agent.py
│   ├── test_api_async_streaming.py
│   └── test_graph_execution.py
└── conftest.py                      # Shared fixtures (mock LLM, sandbox workspaces, checkpointers)
```

---

## Real-World Application Scenarios (Tier 4)

| # | Scenario | Features Exercised | Target Behavior |
|---|----------|--------------------|-----------------|
| 1 | **SWE-bench Bug Fix Lifecycle** | F1, F2, F8, F9, F11, F12, F15, F17, F18 | ReAct agent navigates workspace using grep/AST, identifies broken function, applies surgical diff, verifies with linter/pytest, and commits clean fix. |
| 2 | **Full-Stack REST Microservice Scaffolding** | F1, F3, F4, F6, F7, F10, F12, F19, F20 | System plans, scaffolds FastAPI backend, React frontend components, runs multi-tool static analysis (Ruff + Bandit + Mypy), generates tests, and passes QA. |
| 3 | **Long-Running Pipeline Crash & Persistent Resume** | F1, F6, F20, F23 | Pipeline is halted midway (server kill), checkpointer restores full state from Redis/Postgres/MemorySaver, and execution resumes seamlessly to completion. |
| 4 | **Interactive Human-in-the-Loop Approval via API** | F4, F6, F19, F20, F23 | Watchdog intercepts 3 consecutive task failures, pauses pipeline in `human_approval`, client inspects status via API, provides corrective prompt, and resumes pipeline. |
| 5 | **Real-Time Streaming Telemetry Monitor** | F12, F20, F21, F22 | Client subscribes to SSE `/stream` and WebSocket `/ws`, receiving live token-by-token thought trajectories, tool calls, and trace records during execution. |
