# Agent Profiles & Node Contracts

This document serves as the official registry for all agents (nodes) operating within our LangGraph ecosystem. Each agent adheres to strict **bounded autonomy constraints** and specific input/output state update signatures.

Our system enforces the rule that **agents are stateless pure functions**. They receive `ProjectState`, perform a deterministic LLM invocation or programmatic side-effect, and return a dictionary of state mutations.

---

## 1. Initializer Node
**Role:** System Bootstrapper
- **Input Requirements:** Initial payload with `project_id`, `requirements`, and optionally `workspace_dir`.
- **Bounded Autonomy:** None. Purely deterministic python execution.
- **Responsibilities:** Creates the sandboxed filesystem workspace, writes `.gitkeep`, initializes a git repository, and checks out the active branch.
- **State Update Signature:** Returns initial setup data including `workspace_dir`, `status` ("running"), and initial `execution_trace`.

## 2. Requirement Analyzer
**Role:** Product Manager
- **Input Requirements:** Natural language `requirements`.
- **Bounded Autonomy:** Permitted to format and structure requirements into technical specs, but cannot invent features outside the user's prompt.
- **Responsibilities:** Translates ambiguous user prompts into structured `technical_specifications`.
- **State Update Signature:** `{"technical_specifications": dict}`

## 3. Architect Agent
**Role:** System Designer
- **Input Requirements:** `technical_specifications`.
- **Bounded Autonomy:** Authorized to define the project directory tree and API schema. Not permitted to write code files.
- **Responsibilities:** Creates Architectural Decision Records (ADRs) and the global project structure.
- **State Update Signature:** `{"architecture_decisions": list, "project_structure": dict}`

## 4. Task Planner
**Role:** Project Manager
- **Input Requirements:** `project_structure` and `architecture_decisions`.
- **Bounded Autonomy:** Can decompose architecture into tasks, but must assign them linearly with strict dependencies.
- **Responsibilities:** Generates a granular, atomic queue of `TaskItem` objects.
- **State Update Signature:** `{"task_queue": list[TaskItem]}`

## 5. Backend Engineer
**Role:** Python/FastAPI Developer
- **Input Requirements:** A popped `TaskItem` from the queue, `architecture_decisions`, and access to `SubprocessExecutor` & `FileSystemManager`.
- **Bounded Autonomy:** Confined to generating logic for the assigned `TaskItem`. Forbidden from touching frontend `.tsx`/`.js` files. Enforced by prompt boundaries and Stagnation Checks.
- **Responsibilities:** Generates, edits, and commits backend code.
- **State Update Signature:** `{"task_queue": list (remaining tasks), "code_artifacts": list[CodeArtifact], "execution_trace": list}`

## 6. Frontend Engineer
**Role:** React/UI Developer
- **Input Requirements:** A popped `TaskItem` from the queue.
- **Bounded Autonomy:** Confined to generating React/UI logic. Cannot alter backend API models.
- **Responsibilities:** Scaffolds components, hooks, and CSS files, then commits them.
- **State Update Signature:** `{"task_queue": list, "code_artifacts": list[CodeArtifact], "execution_trace": list}`

## 7. Memory Compression Node
**Role:** Context Optimizer
- **Input Requirements:** `completed_tasks` and `execution_trace`.
- **Bounded Autonomy:** Programmatic execution.
- **Responsibilities:** Summarizes massive logs to keep the LLM context window slim to avoid hitting token limits during long runs. Clears raw traces via the `clearable_list_reducer`.
- **State Update Signature:** `{"completed_tasks": ["CLEAR", summary_dict], "execution_trace": ["CLEAR"]}`

## 8. QA Tester
**Role:** Automated Validation Specialist
- **Input Requirements:** `code_artifacts`.
- **Bounded Autonomy:** Allowed to auto-generate `pytest` files for un-tested artifacts and execute them securely via `SubprocessExecutor`.
- **Responsibilities:** Verifies execution paths. If tests fail, it re-queues the failed task with feedback for the engineer.
- **State Update Signature:** `{"task_queue": list, "code_artifacts": list (updated test status)}`

## 9. Code Reviewer
**Role:** Security & Quality Auditor
- **Input Requirements:** Tested `code_artifacts`.
- **Bounded Autonomy:** Performs static analysis to detect SQLi, XSS, and "hallucinated" uninstalled dependencies.
- **Responsibilities:** Approves artifacts or rejects them back to the `task_queue` with severe warnings.
- **State Update Signature:** `{"task_queue": list, "status": "running" | "failed"}`

## 10. Watchdog Node
**Role:** Infinite Loop Guard
- **Input Requirements:** `retry_counts` map.
- **Bounded Autonomy:** Programmatic execution.
- **Responsibilities:** Acts as the routing waypoint for failures. Evaluates if any task retry limit has exceeded the threshold (3 times).
- **State Update Signature:** Typically `{}`, heavily relies on graph conditional edge routing.

## 11. Human Approval Node
**Role:** Fallback Halt State
- **Input Requirements:** Excessive failures intercepted by the Watchdog.
- **Bounded Autonomy:** Hard block.
- **Responsibilities:** Pauses execution and transitions the graph to a blocked state requiring manual intervention to prevent runaway token usage.
- **State Update Signature:** `{"status": "blocked"}`

---

## Tool Layer Managers (Capabilities)

While not active LangGraph nodes, these components act as capability agents bounded strictly to deterministic APIs:

### 12. FileSystemManager
- **Responsibilities:** Sandboxed read/write capabilities. Automatically blocks directory traversal attacks outside the workspace.
- **Mutation Pattern:** Logs all mutations strictly to the `execution_trace`.

### 13. GitTracker
- **Responsibilities:** Maintains project version history natively. Commits every successful agent output for point-in-time recovery.
- **Mutation Pattern:** Logs bash outputs and git hashes to the `execution_trace`.
