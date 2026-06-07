# Failure Recovery & Anti-Loop Guardrails

Autonomous agent pipelines are inherently prone to catastrophic "infinite loops", where an agent generates broken code, a testing node fails it, and the agent blindly regenerates the exact same broken code indefinitely, consuming massive amounts of API tokens and time. 

To prevent this, our orchestration relies on a multi-tiered failure recovery architecture.

## 1. Node-Level Retry Middleware

Every agent function (node) in our StateGraph is decorated with `@retry_middleware(max_retries=3)`. 
This is the first line of defense against transient failures (e.g., OpenAI API 502 errors, JSON parsing hallucinations).

- **Mechanism:** If an exception is raised inside a node, the middleware catches it before LangGraph sees a catastrophic graph crash.
- **State Mutation:** It logs the failure to the `error_log` array, increments the `retry_counts` map for that specific node, and attempts to re-run the node locally.
- **Failure:** If it breaches `max_retries`, the graph status is hard-set to `failed`, triggering conditional edges to abort the pipeline cleanly.

## 2. Stagnation Detection (Recursion Protection)

When an engineer agent fails a QA test, the task is re-queued. To prevent the agent from attempting the exact same failed solution:

- **Git Diff Evaluation:** Before an agent commits a "fix", it evaluates the `GitTracker.diff()`. 
- **Stagnation Trigger:** If the LLM produces a source code artifact that results in a zero-byte diff (meaning it didn't actually change the logic from the last failed attempt), the agent forcibly raises a `StagnationError`.
- **Resolution:** This triggers the retry middleware, forcing the LLM to process the previous failure and try a radically different approach.

## 3. The Watchdog & Human-in-the-loop (HITL)

If an engineer agent attempts to fix a failing test but continuously fails the QA/Review cycle, we rely on the graph-level **Watchdog Node**.

- **Routing Waypoint:** The conditional edges `route_after_tester` and `route_after_reviewer` do not send failures directly back to the engineers. They route to the `watchdog`.
- **Heuristic Check:** The Watchdog inspects the global `ProjectState["retry_counts"]`. If it detects that a specific task loop (e.g., `task_fail_TASK-001`) has breached the threshold limit (default: 3 iterations):
- **Escalation Path:** The Watchdog overrides the routing and sends the graph to the `human_approval` node.
- **Halt State:** The `human_approval` node acts as a hard breakpoint. It updates the graph status to `blocked`, terminating execution to save tokens. An engineer can then step in, review the context via the FastAPI `/status` endpoint, and manually intervene.

## 4. Token Budget Tracking & Memory Compression

Long-running projects accumulate massive `execution_trace` lists, eventually breaching the LLM's context window limit (e.g., 128k tokens), resulting in a hard crash.

- **Trigger:** The `memory_compression` node automatically runs whenever the `task_queue` is completely drained, prior to the QA phase.
- **Compression Logic:** It passes the dense execution traces to an LLM, generating a highly dense technical summary. 
- **State Pruning:** Utilizing LangGraph's custom reducers (`clearable_list_reducer`), it pushes a special `"CLEAR"` signal to safely flush the massive raw arrays from the State dictionary, replacing them with the condensed context blocks. This guarantees the pipeline can run indefinitely without hitting API context caps.
