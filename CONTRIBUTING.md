# 🛠 Contributing Guidelines & Engineering Standards

Thank you for contributing to the **Autonomous AI Software Engineering Team**! This project sets the enterprise benchmark for LangGraph-driven multi-agent systems, ReAct code navigation, and deterministic software verification.

To maintain the rock-solid reliability, thread safety, and immutability of our orchestration pipeline, we enforce strict architectural standards for state schemas, agent nodes, tools, and test suites.

---

## 🏛 1. Core State Schema & Reducer Contracts

The global state in [`src/core/state.py`](src/core/state.py) is the single source of truth for the entire multi-agent graph.

### Strict Reducer Rule for State Collections
Never use raw lists for state fields that can be updated concurrently or across iterations. You **must** annotate collections with custom reducers defined in [`src/core/reducers.py`](src/core/reducers.py):

| State Key | Required Reducer | Rationale |
| :--- | :--- | :--- |
| `code_artifacts` | `artifact_reducer` | Canonicalizes file paths, prevents duplicate entries, and manages atomic version bumps upon content change. |
| `task_queue` | `task_queue_reducer` | Performs atomic task upserts by `task_id` while preserving execution order. |
| `completed_tasks` | `clearable_list_reducer` | Appends completed tasks while supporting atomic context resets via `ClearSignal` or `"CLEAR"`. |
| `execution_trace` | `clearable_list_reducer` | Accumulates telemetry and enables memory compression truncation. |
| `architecture_decisions` | `adr_reducer` | Deduplicates and updates ADRs by `decision_id`. |
| `retry_counts` | `dict_merge_reducer` | Merges retry counters without overwriting failure tracking keys. |

### Pydantic Sub-Model Typing
All complex payloads (e.g., `TaskItem`, `CodeArtifact`, `ADR`, `ErrorRecord`) must have a corresponding Pydantic v2 `BaseModel`. Untyped or generic dictionaries are strictly forbidden inside graph state lists.

---

## 🤖 2. Agent Node Design (Stateless Pure Functions)

Every LangGraph node represents an isolated agent worker and must adhere to pure functional semantics:

### Node Signature
```python
def my_agent_node(state: ProjectState) -> dict[str, Any]:
    """
    Pure function: receives frozen ProjectState, performs bounded
    LLM invocation or side-effect, and returns state mutation dict.
    """
    ...
```

### Mandatory `@retry_middleware` Wrapper
All agent nodes must be decorated with `@retry_middleware(max_retries=3)`.
- Do not write manual `try/except` loops inside node functions.
- The middleware automatically captures exceptions, records structured `ErrorRecord` objects in `error_log`, tracks attempts in `retry_counts`, and manages exponential backoff.

---

## 🔧 3. Tool Sandboxing & Execution Security

Tools in [`src/tools/`](src/tools/) perform sandboxed execution and must observe the following constraints:

1. **Workspace Jail & Path Containment**:
   All filesystem reads and writes must pass through `FileSystemManager._safe_resolve()`. Any path attempting to escape the workspace root (`../../`, `/etc/passwd`, `C:\Windows`) must immediately raise a `PermissionError`.
2. **Credential Sanitization**:
   Subprocess executors must never leak host secrets. Any environment variable containing `KEY`, `SECRET`, `PASS`, `TOKEN`, or `CREDENTIAL` must be filtered from child processes unless explicitly supplied.
3. **Structured Telemetry Ingestion**:
   All tools must accept the `trace: list[dict[str, Any]]` array by reference and append structured records (operation, duration, input, sha256 output, exit code).

---

## 🧪 4. 4-Tier Testing Requirement

Every pull request must maintain **100% test pass rate** across all 4 verification tiers:

- **Unit Tests (`tests/unit/`)**: Verify custom reducers, signal handling, and checkpointer persistence backends in complete isolation.
- **Tier 1 Feature Tests (`tests/e2e/test_tier1_features.py`)**: Individual feature assertions across F1–F25.
- **Tier 2 Boundary Tests (`tests/e2e/test_tier2_boundaries.py`)**: Edge cases, schema bounds, syntax errors, and fault injection.
- **Tier 3 Combination Tests (`tests/e2e/test_tier3_combinations.py`)**: Cross-feature and multi-agent interaction combinations.
- **Tier 4 Workloads (`tests/e2e/test_tier4_workloads.py`)**: Sustained SWE-bench bug-fix scenarios, server reboot state recovery, and HITL flows.

### Running Test Verification
```bash
# Execute entire test suite (325 tests)
python -m pytest tests/unit/ tests/e2e/ -v
```

---

## 📋 5. Pull Request Review Checklist

Before opening a PR, ensure that:
- [ ] All new state fields use appropriate custom reducers from `src/core/reducers.py`.
- [ ] Agent nodes are pure functions decorated with `@retry_middleware`.
- [ ] Pydantic models for structured LLM outputs have `__test__ = False` if class names start with `Test`.
- [ ] Subprocess executions resolve workspace paths safely and sanitize host secrets.
- [ ] Full test suite passes: `python -m pytest tests/unit/ tests/e2e/ -v` (325/325 tests).
- [ ] Code is formatted with `black` / `ruff` and strictly typed without linting warnings.
