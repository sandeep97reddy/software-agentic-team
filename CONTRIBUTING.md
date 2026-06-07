# Contributing Guidelines

Thank you for your interest in contributing to our autonomous AI software engineering team! This project aims to define the enterprise standard for LangGraph orchestration and multi-agent coordination.

To maintain the deterministic reliability of our orchestration pipeline, we enforce strict guidelines when contributing to the `ProjectState`, Agent Nodes, or the Tool/Skill layers.

## 1. Expanding the Core State Schema

The global state (`src/core/state.py`) is the lifeblood of the LangGraph pipeline.

- **Use TypedDict:** Any new top-level key must be defined in the `ProjectState` `TypedDict`. LangGraph utilizes this dict representation to manage node-to-node routing states.
- **Annotated Reducers for Collections:** If your new key represents a collection (like a list of items), you **must** use `Annotated[..., operator.add]` or a custom reducer.
  - *Why?* If parallel branches execute, failing to use a reducer will cause one branch to entirely overwrite the progress of another.
- **Pydantic Validation:** If the data structure is complex (e.g., an object rather than a primitive), define a `pydantic.BaseModel` for it. Do not rely on untyped dictionaries inside arrays.

## 2. Scaling Out New Nodes (Agents)

If you are adding a new agent (e.g., a `database_architect` or a `devops_engineer`), follow the architectural standard:

### Step 1: Define the Node Interface
All nodes must be pure functions with a strict signature:
```python
def my_new_node(state: ProjectState) -> dict[str, Any]:
    pass
```

### Step 2: Implement the `@retry_middleware`
You **must** decorate your node with `@retry_middleware()`. Do not write custom `try/except` loops to catch API timeouts or LLM hallucinations. Let the middleware handle it uniformly so the global `error_log` and `retry_counts` update deterministically.

### Step 3: Wire into the Graph
Open `src/core/graph.py` and register your node inside `build_graph()`:
```python
graph.add_node("my_new_node", my_new_node)
```
Update the relevant routing edge logic (e.g., modifying `route_to_workers`) to dynamically direct tasks to your new node when appropriate.

## 3. Adding New Skills (Tools)

Tools are defined in `src/tools/` and are strictly distinct from Agents. They are deterministic Python modules that interact with the host system.

- **Class-Based Encapsulation:** New capabilities (e.g., `DockerManager` or `AWSCliWrapper`) must be built as Python classes.
- **Trace Injection Requirement:** The `__init__` method of your new tool must accept the global `execution_trace` array by reference:
  ```python
  def __init__(self, trace: list[dict]):
      self.trace = trace
  ```
- **Strict Logging:** Every action the tool takes (e.g., spinning up a container, downloading an artifact) must append a structured event log to `self.trace`. This guarantees complete auditability of side effects.
- **Sandbox Isolation:** Ensure any file paths or executable inputs are strictly confined within the `workspace_dir` context passed from the `Initializer` node.

## 4. Pull Request Process

1. Create a descriptive feature branch (e.g., `feat/add-database-node`).
2. Implement your logic conforming to the `CODE_STANDARDS.md`.
3. Add or update tests in the `tests/` directory. If testing a new tool, append it to the `test_tools_integration.py` script.
4. Execute the pipeline locally using the `RUNBOOK.md` instructions and ensure the graph compiles and traces perfectly in LangSmith.
5. Submit your PR and request a review from a core maintainer.
