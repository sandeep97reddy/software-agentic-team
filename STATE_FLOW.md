# Global State Flow & Execution Sequence

This document traces the exact sequence of how a single natural language requirement traverses the LangGraph state machine, mutates the global `ProjectState`, and routes through the orchestration nodes until successful completion.

## The Execution Lifecycle

Every project begins when an HTTP POST request hits the `/api/v1/execute` endpoint. The FastAPI router instantiates the initial `ProjectState` payload and invokes the compiled LangGraph execution.

### 1. Initialization (Bootstrapping)
- **Node:** `initializer`
- **Action:** A sandboxed local workspace is created. A git repository is initialized.
- **State Mutation:** Updates `workspace_dir`, `status` to `running`, and injects the first records into the `execution_trace`.

### 2. Planning & Architecture Phase
- **Node:** `requirement_analyzer`
  - **Input:** Raw string `requirements`.
  - **State Mutation:** Translates the raw text into a structured Pydantic model representation assigned to `technical_specifications`.
- **Node:** `architect`
  - **Input:** `technical_specifications`.
  - **State Mutation:** Defines the software structural design. Generates and appends architectural blueprints to `architecture_decisions` and the JSON `project_structure`.
- **Node:** `task_planner`
  - **Input:** `architecture_decisions` and `project_structure`.
  - **State Mutation:** Decomposes the blueprints into granular `TaskItem` blocks, appending them into the `task_queue` list.

### 3. Execution & Routing Loop (The Workers)
The graph now hits a conditional edge (`route_to_workers`). It inspects `task_queue`.
- **Node Routing:** If the top task pertains to UI elements (`.tsx`, `.css`), the graph routes to the `frontend_engineer`. If backend, it routes to `backend_engineer`.
- **Action:** The engineer pops the task, reads the architectural context, utilizes the `FileSystemManager` to generate the file, and runs `GitTracker` to commit the code.
- **State Mutation:** The task is moved from `task_queue` to `completed_tasks`. The generated file context is appended to `code_artifacts`. `execution_trace` expands with Git and FS logs.
- **Loop:** The graph routes back to `route_to_workers`. This loop repeats until `task_queue` is completely drained.

### 4. Memory Optimization
- **Node:** `memory_compression`
- **Condition:** Triggers automatically once the worker queue is empty.
- **State Mutation:** Analyzes the massive lists in `completed_tasks` and `execution_trace`. It summarizes the content, and then utilizes the custom `clearable_list_reducer` (by passing the `"CLEAR"` string) to reset the raw logs while retaining the summary. This crucial step prevents the LLM from exceeding token limits during long runs.

### 5. Validation & Quality Assurance
- **Node:** `tester`
  - **Input:** `code_artifacts`.
  - **Action:** Dynamically utilizes the `SubprocessExecutor` to generate and run `pytest` files inside the sandbox.
  - **State Mutation:** If tests fail, it re-queues the associated file back into the `task_queue` with explicit error feedback, forcing the execution loop to restart.
- **Node:** `reviewer`
  - **Condition:** Reached only if the QA Tester passes.
  - **Action:** Performs a security/style analysis (e.g., detecting hallucinated package imports or XSS vectors).
  - **State Mutation:** Appends findings to `error_log` and can also force tasks back onto the `task_queue` if severe warnings are triggered.

### 6. Termination
When the task queue remains empty after the validation nodes, the graph safely routes to the terminal `END` node. The FastAPI endpoint captures the final, fully populated `ProjectState` dictionary and returns it to the caller.
