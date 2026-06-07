# Engineering Code Standards

To guarantee predictability in a non-deterministic AI orchestrator, our codebase strictly adheres to enterprise-grade software engineering standards. This document outlines the patterns and conventions required when modifying the pipeline or introducing new agents.

## 1. Type Safety & State Validation

Agents interact with a central state blob (`ProjectState`). To prevent schema drift and runtime key errors, strict typing boundaries are enforced:

### LangGraph Dicts vs Pydantic Models
- **LangGraph State:** The global `ProjectState` MUST be defined as a Python `TypedDict`. This fulfills LangGraph's routing requirements.
- **Agent Boundaries:** When data enters an agent, or is parsed out of an LLM generation, it MUST be validated via `Pydantic` `BaseModel` classes (e.g., `TaskItem`, `CodeArtifact`, `ArchitectureDecision`).
- **Structured LLM Outputs:** Always utilize `langchain.output_parsers.PydanticOutputParser` or `with_structured_output()` to coerce the LLM into returning valid Pydantic JSON. Never rely on raw regex scraping for critical state objects.

## 2. Deterministic State Updating (Reducers)

In an asynchronous, multi-agent environment (especially when parallel branching is enabled), direct state mutation leads to race conditions. 

- **Append-Only Operations:** We utilize `operator.add` through LangGraph's `Annotated` type hint. Nodes must never overwrite lists. Instead, returning `{"task_queue": [new_task]}` cleanly appends to the queue.
- **Custom Reducers:** When an array needs to be safely truncated (e.g., `completed_tasks` cleared by the `memory_compression` node), use a dedicated reducer function (e.g., `clearable_list_reducer`) that intercepts string signals like `"CLEAR"` safely.

## 3. Asynchronous & Sync Execution Patterns

- **Synchronous vs Async Nodes:** Currently, all core graph nodes are built as synchronous pure functions for deterministic debugging and predictable testing. If introducing I/O heavy operations (like external API calls), nodes can be migrated to `async def` signatures as supported natively by LangGraph.
- **Stateless Execution:** Nodes must NEVER store information in class-level or global variables. All execution context must be retrieved from the `ProjectState` argument, and all outputs must be returned in the resulting dictionary.

## 4. Code Quality & Observability Criteria

- **No Silent Failures:** All agent logic must be wrapped in our generic `@retry_middleware`. Exceptions inside LLM parsing or tool execution should raise standard Python errors. The middleware will catch them, log the stack trace to the `error_log`, increment the `retry_counts`, and attempt a clean restart.
- **Granular Tracing:** Every tool invocation must append a timestamped, structured dictionary to `state["execution_trace"]`.
- **LangSmith Tagging:** Ensure `get_run_config()` is passed alongside all `graph.invoke()` calls at the API route layer to guarantee accurate grouping and distributed trace tracking within the LangSmith UI.
- **Linting & Formatting:** All Python code must conform to strict `ruff` formatting, type-hinting checking (`mypy`), and `isort` configurations.
