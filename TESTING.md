# 🧪 Enterprise Verification & Automated QA Architecture

Unlike traditional AI coding assistants that blindly trust LLM-generated code, our framework mathematically guarantees code execution and system correctness by enforcing a multi-layered, sandboxed QA and testing architecture.

---

## 🏛 1. The 4-Tier Verification Hierarchy

Our verification suite comprises **325 automated tests** organized across 4 rigorous tiers:

```
tests/
├── unit/                         # 51 Unit Tests: Core State Reducers & Persistence Checkpointers
│   ├── test_reducers.py         # ClearSignal, artifact_reducer, task_queue_reducer, dict_merge_reducer, adr_reducer
│   └── test_checkpointer.py     # Memory, Postgres, and Redis checkpointer backends
│
└── e2e/                          # 274 Tests: 4-Tier Integration and E2E Harness
    ├── conftest.py               # MockLLMFactory, WorkspaceHelper, and test fixtures
    ├── test_tier1_features.py    # 115 Tests: Tier 1 Feature Verification (F1–F25)
    ├── test_tier2_boundaries.py  # 115 Tests: Tier 2 Edge Cases, Boundary Limits & Fault Injections
    ├── test_tier3_combinations.py# 30 Tests: Tier 3 Cross-Feature & Multi-Agent ReAct Combinations
    └── test_tier4_workloads.py   # 14 Tests: Tier 4 SWE-bench Bug-Fixes & Full-Stack Scenarios
```

### Verification Tier Breakdown

| Tier | Focus Area | Test Count | Key Invariants Tested |
| :--- | :--- | :---: | :--- |
| **Unit Tests** | Reducer & Storage Units | 51 | CRDT conflict-free state merging, path normalization, thread-isolated checkpoints. |
| **Tier 1 (Features)** | Individual Capabilities | 115 | AST symbol navigation, grep/glob search, surgical diffs, subprocess sandbox, SSE streaming. |
| **Tier 2 (Boundaries)**| Edge Cases & Faults | 115 | Malformed JSON recovery, path traversal denial, infinite loop detection, secret sanitization. |
| **Tier 3 (Combinations)**| Multi-Agent Interactions | 30 | ReAct loop navigation $\rightarrow$ diff patching $\rightarrow$ automated test generation $\rightarrow$ security review. |
| **Tier 4 (Workloads)** | End-to-End Scenarios | 14 | SWE-bench multi-turn bug fix lifecycle, full-stack microservice scaffolding, crash-recovery replay. |

---

## ⚙️ 2. Automated Dynamic QA & Test Generation

The `tester_node` ([`src/agents/tester.py`](src/agents/tester.py)) serves as an autonomous validation specialist:

1. **Automatic Test Generation**:
   When new code artifacts are detected, the QA tester generates targeted `pytest` assertions verifying interface adherence, error boundaries, and return contracts.
2. **Sandboxed Subprocess Execution**:
   Tests run inside an isolated workspace via `SubprocessExecutor` with automatic `PYTHONPATH` injection (`workspace_root` + `workspace_root/src`) and UTF-8 encoding.
3. **Trace Telemetry & Re-queueing**:
   If pytest fails, stdout/stderr stack traces are extracted, wrapped into a `FIX-<filename>` task with telemetry metadata, and re-queued for the engineer agent to debug.

---

## 🔒 3. Static Security & Quality Auditing

The `reviewer_node` ([`src/agents/reviewer.py`](src/agents/reviewer.py)) performs static analysis on generated artifacts:
- Detects SQL injection vulnerabilities (`cursor.execute(f"...")`).
- Flags unvalidated subprocess calls and shell injection vectors.
- Rejects vulnerable code back to the task queue before any commits are finalized.

---

## 🚀 4. Executing the Test Suite

```bash
# Run the complete test suite (325 tests)
python -m pytest tests/unit/ tests/e2e/ -v

# Run with short traceback on failure
python -m pytest tests/unit/ tests/e2e/ --tb=short

# Run specific tier
python -m pytest tests/e2e/test_tier4_workloads.py -v
```

### Execution Baseline
```
======================= 325 passed, 1 warning in 36.68s =======================
```
