# Skills Registry (Tool Capability Layer)

Our agents possess strictly bounded capabilities defined through the `tools` module. These capabilities, or "Skills", grant agents deterministic, isolated methods for producing side-effects safely.

By treating these as separate, object-oriented manager classes, we enforce a unified interface where all side-effects are aggressively logged to the `execution_trace` in the global `ProjectState`. This guarantees an audit trail for every action.

---

## 1. FileSystemManager

**File:** `src/tools/filesystem.py`

The `FileSystemManager` represents the agent's ability to mutate the physical sandbox structure. It acts as an abstraction over standard python OS operations with injected security barriers.

### Core Capabilities
- `write_file(path, content)`: Creates or overwrites files. Includes built-in directory scaffolding (i.e., `os.makedirs` equivalent).
- `read_file(path)`: Retrieves content for the LLM context.
- `list_dir(path)`: Explores tree structures for debugging context.
- `delete_file(path)`: Prunes unneeded or erroneously created assets.

### Safety & Constraints
- **Directory Traversal Protection:** All input paths are resolved securely against the `workspace_dir` base path. Attempts to navigate backwards (`../../etc/passwd`) immediately trigger a `PermissionError` and log a security violation to the trace.
- **Trace Mutability:** Every I/O operation appends a structured record (action type, file path, success boolean, timestamp) to the mutable `execution_trace` list passed during instantiation.

---

## 2. SubprocessExecutor (Isolated Execution)

**File:** `src/tools/executor.py`

Agents must test their code, install dependencies, and run linters. The `SubprocessExecutor` governs how these commands execute on the host or inside a constrained Docker/sandbox runtime.

### Core Capabilities
- `run_command(command, timeout)`: Fires shell commands securely. Captures `stdout`, `stderr`, and precise exit codes (0 for success).
- `run_pytest(target_path)`: Opinionated helper that forces structured output formats for the `QA Tester` agent.

### Safety & Constraints
- **Timeout Restrictions:** Prevents infinite loop hangs (e.g., a process waiting for user input) by enforcing strict timeouts (defaulting to e.g., 30s or 60s).
- **Allowlist Enforcements:** Can be configured to only permit pre-approved command binaries (e.g., `pytest`, `npm`, `pip`, `python`) to prevent arbitrary execution vulnerabilities.
- **Trace Mutability:** Command inputs, sanitized output streams, execution duration, and exit codes are logged to the `execution_trace`.

---

## 3. GitTracker

**File:** `src/tools/git_tracker.py`

Acts as the project's native version control mechanism. Agents do not merely overwrite text files; they construct a linear repository history. This allows developers using the final system to gracefully rewind and understand the autonomous agent's decision-making timeline.

### Core Capabilities
- `init(default_branch)`: Scaffolds the `.git` directory inside the workspace sandbox upon initialization.
- `commit(message)`: Takes an atomic snapshot of current `FileSystemManager` alterations.
- `ensure_branch(branch_name)`: Creates or checks out isolated branches.
- `log() / diff()`: Provides agents the ability to introspect exactly what they changed since the last step.

### Safety & Constraints
- **Automated Staging:** Abstracts away raw `git add` and `git commit` intricacies to ensure the graph execution never stalls on unresolved staging conflicts.
- **Trace Mutability:** Commits attach Git SHA-hashes into the `execution_trace` for point-in-time recovery tracking.
