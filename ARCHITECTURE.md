# Architecture & State Management

This document details the core architectural decisions, state management patterns, and edge routing logic powering our autonomous AI software engineering team.

## 🧠 LangGraph State Machine

The system is designed as a deterministic **StateGraph** using LangGraph. The pipeline is fundamentally a state machine that transitions through well-defined phases: `planning`, `execution`, and `validation`. 

Each node in the graph represents an isolated worker (or agent) that reads from a single, globally immutable-by-convention state, performs its task, and returns a dictionary of state updates.

### Phase 1: Planning
The pipeline always begins linearly to prevent execution before design is finalized.
`Initializer` → `Requirement Analyzer` → `Architect` → `Task Planner`

### Phase 2: Execution & Routing Loop
The execution phase uses conditional edges to dynamically route tasks to specialized engineers.
The `Task Planner` populates a queue of tasks. A router node evaluates `state["task_queue"]`. If tasks exist, it looks at the target file extensions:
- Exts like `.tsx`, `.js`, `.css` → routes to **Frontend Engineer**
- Other extensions/paths → routes to **Backend Engineer**

This loop continues until the task queue is empty.

### Phase 3: Validation & Memory
Once the queue is drained:
`Memory Compression` → `QA Tester`

The `QA Tester` runs deterministic static analysis and tests via the Subprocess Executor. 
- **Pass:** Routes to the **Reviewer**.
- **Fail:** Re-queues the failed task and routes to the **Watchdog**.

## 💾 State Management (ProjectState)

Our state is strictly typed using Python's `TypedDict` and augmented with `operator.add` reducers to enable safe parallel branching and append-only state mutations.

The single source of truth is the `ProjectState` (defined in `src/core/state.py`):

```python
class ProjectState(TypedDict, total=False):
    project_id: str
    requirements: str
    architecture_decisions: Annotated[list[dict[str, Any]], operator.add]
    task_queue: list[dict[str, Any]]
    completed_tasks: Annotated[list[dict[str, Any]], clearable_list_reducer]
    code_artifacts: Annotated[list[dict[str, Any]], operator.add]
    execution_trace: Annotated[list[dict[str, Any]], clearable_list_reducer]
    retry_counts: dict[str, int]
    current_phase: str
    status: str
```

### Pydantic Sub-Models
While the LangGraph global state must be a `TypedDict` to satisfy schema constraints, we enforce rigorous typing at the agent boundary using **Pydantic BaseModel**.

Key sub-models include:
- `TaskItem`: Defines bounded execution criteria for engineers (`task_id`, `dependencies`, `assigned_to`).
- `ArchitectureDecision`: An ADR (Architecture Decision Record) dictating the system design.
- `CodeArtifact`: Represents generated code, its path, language, and validation status.
- `ErrorRecord`: A structured log for tracking node failures and facilitating observability.

### Reducer Functions
To prevent destructive overrides during concurrent execution, we use custom reducers. For example, `clearable_list_reducer` appends lists naturally but allows nodes to pass `"CLEAR"` to reset the array. This is critical for the **Memory Compression** node to truncate logs dynamically while preserving long-term memory.

## 🔀 Conditional Edge Routing

LangGraph conditional edges (`add_conditional_edges`) dictate the flow dynamically based on state mutations.

1. **`route_to_workers`**: Evaluates the `task_queue`. If empty, returns `memory_compression`. If populated, analyzes the first task's file path to return either `frontend_engineer` or `backend_engineer`.
2. **`route_after_tester`**: Checks the QA results. If tests fail, returns `watchdog`. If pass, returns `reviewer`.
3. **`route_after_reviewer`**: Checks for security/style issues. If anomalies are found, returns `watchdog`, else returns `END`.
4. **`route_after_watchdog`**: Checks `retry_counts`. If any task fails >= 3 times, routes to `human_approval`. Otherwise, routes back to `route_to_workers`.

## 🛡 Retry Middleware

Instead of embedding try/catch loops in every agent, all LangGraph nodes are decorated with a `@retry_middleware`.
This wrapper intercepts execution. If a node crashes, it logs the exception to the `error_log`, increments `retry_counts[node_name]`, and re-invokes the node up to `max_retries`. This separates business logic from fault-tolerance logic.
