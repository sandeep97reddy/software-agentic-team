# Automated Testing & QA

Unlike traditional AI coding tools that blindly trust the output of an LLM generation, our framework mathematically guarantees execution paths by enforcing a containerized, deterministic QA phase.

## The QA Tester Node

The `tester` node acts as an automated Validation Specialist. It runs precisely when the execution loop has completely drained the `task_queue`, meaning the entire architecture for this iteration has been written to the sandbox.

### 1. Test Generation
When evaluating `code_artifacts`, the QA Tester detects components that lack corresponding `tests_passed` coverage flags.
Instead of relying on the original Engineer to test its own code, the QA Tester possesses the autonomy to independently generate `pytest` assertion files based on the global `architecture_decisions` and `technical_specifications`.

### 2. Isolated Execution
Tests are executed using the `SubprocessExecutor` tool. 

- **Security Constraint:** The executor operates exclusively within the bounds of the sandbox directory (`workspace_dir`).
- **Timeout Restrictions:** The executor enforces strict hardware-level timeouts. If a generated test initiates an infinite loop or a hanging network request, the executor kills the process, returning a non-zero exit code and capturing the trace.

### 3. Feedback Loop Routing
When `pytest` exits:
- **Success (Exit Code 0):** The `tester` flags the code artifacts as verified. The LangGraph routing function (`route_after_tester`) evaluates the state and forwards the graph to the `reviewer` node for static code analysis and security auditing.
- **Failure (Non-Zero Exit Code):** The `tester` captures the raw `stderr` and `stdout` traceback arrays. It resurrects the completed `TaskItem` responsible for that file, injects the stack trace into the task's `metadata`, and appends the task *back* onto the `task_queue`.

### 4. Graph Handoff
By pushing a task back onto the queue, the conditional edge router intercepts the state. Instead of terminating, the graph loops back to the `backend_engineer` or `frontend_engineer`. The Engineer receives the exact traceback of why its previous code failed in the Python interpreter or Node sandbox, allowing it to accurately debug its own generation in the subsequent iteration.
