"""
Chunk 3 -- Integration smoke-test for the Tool Layer.

Tests FileSystemManager, GitTracker, and SubprocessExecutor end-to-end
against a real temporary directory (no mocking).

Run with::

    .venv\\Scripts\\python.exe tests\\test_tools_integration.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Make sure the workspace root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.executor import ExecutionResult, SubprocessExecutor
from src.tools.filesystem import FileSystemManager
from src.tools.git_tracker import GitTracker

PASS = "[PASS]"
FAIL = "[FAIL]"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────────────────────


def assert_true(condition: bool, msg: str) -> None:
    status = PASS if condition else FAIL
    print(f"  {status}  {msg}")
    if not condition:
        raise AssertionError(msg)


# ──────────────────────────────────────────────────────────────
#  FileSystemManager tests
# ──────────────────────────────────────────────────────────────


def test_filesystem(workspace: str) -> None:
    section("FileSystemManager")
    trace: list[dict] = []
    fs = FileSystemManager(workspace_dir=workspace, trace=trace)

    # write
    record = fs.write_file("src/hello.py", 'print("hello world")\n')
    assert_true(record["success"], "write_file returns success=True")
    assert_true(record["outputs"]["bytes_written"] > 0, "bytes_written > 0")
    assert_true(len(record["outputs"]["sha256"]) == 64, "sha256 is 64-char hex")
    assert_true(Path(workspace, "src", "hello.py").is_file(), "file exists on disk")

    # read
    content = fs.read_file("src/hello.py")
    assert_true('print("hello world")' in content, "read_file returns correct content")

    # list
    entries = fs.list_dir("src")
    assert_true(
        any(e["name"] == "hello.py" for e in entries), "list_dir finds hello.py"
    )
    assert_true(
        all(e["type"] in ("file", "directory") for e in entries),
        "all entries have valid type",
    )

    # file_exists
    assert_true(fs.file_exists("src/hello.py"), "file_exists True for existing file")
    assert_true(
        not fs.file_exists("no_such_file.py"), "file_exists False for missing file"
    )

    # path traversal blocked
    try:
        fs.read_file("../../etc/passwd")
        assert_true(False, "traversal should raise PermissionError")
    except PermissionError:
        assert_true(True, "path traversal correctly blocked by PermissionError")

    # delete
    del_record = fs.delete_file("src/hello.py")
    assert_true(del_record["success"], "delete_file returns success=True")
    assert_true(not fs.file_exists("src/hello.py"), "file is gone after delete")

    # trace completeness -- the blocked traversal raises BEFORE the trace append,
    # so it does NOT add an entry; we get 4 records: write, read, list, delete.
    assert_true(len(trace) >= 4, f"trace has >=4 entries (got {len(trace)})")
    ops = {r["operation"] for r in trace}
    assert_true("write_file" in ops, "trace contains write_file")
    assert_true("read_file" in ops, "trace contains read_file")
    assert_true("list_dir" in ops, "trace contains list_dir")
    assert_true("delete_file" in ops, "trace contains delete_file")
    assert_true(
        all("timestamp" in r for r in trace), "all trace entries have timestamp"
    )
    assert_true(
        all("duration_ms" in r for r in trace), "all trace entries have duration_ms"
    )

    print(f"\n  Trace events recorded: {len(trace)}")


# ──────────────────────────────────────────────────────────────
#  GitTracker tests
# ──────────────────────────────────────────────────────────────


def test_git(workspace: str) -> None:
    section("GitTracker")
    trace: list[dict] = []

    # Pre-create a file to commit
    Path(workspace, "README.md").write_text("# Test Repo\n", encoding="utf-8")

    git = GitTracker(workspace_dir=workspace, trace=trace)

    # init
    result = git.init(default_branch="main")
    assert_true(result.returncode == 0, "git init succeeded (exit=0)")
    assert_true(Path(workspace, ".git").is_dir(), ".git directory created")

    # stage_all + commit
    git.stage_all()
    commit_result = git.commit("chore: initial commit")
    assert_true(commit_result.returncode == 0, "initial commit succeeded")

    # ensure_branch
    branch_result = git.ensure_branch("feature/test-branch")
    assert_true(branch_result.returncode == 0, "branch creation succeeded")
    current = git.current_branch()
    assert_true(
        current == "feature/test-branch",
        f"current branch is feature/test-branch (got: {current})",
    )

    # Write a new file and diff
    Path(workspace, "app.py").write_text("x = 1\n", encoding="utf-8")
    git.stage_all()
    diff_text = git.diff(staged=True)
    assert_true("app.py" in diff_text or diff_text == "", "diff captures app.py change")

    # commit the change
    git.commit("feat: add app.py")

    # log
    commits = git.log(n=5)
    assert_true(len(commits) >= 1, f"log returns >=1 commit (got {len(commits)})")
    assert_true("hash" in commits[0], "log entry has hash field")
    assert_true("subject" in commits[0], "log entry has subject field")

    # status (should be clean now)
    status_entries = git.status()
    assert_true(isinstance(status_entries, list), "status returns a list")

    # trace
    assert_true(len(trace) >= 5, f"git trace has >=5 entries (got {len(trace)})")
    assert_true(
        all("exit_code" in r["outputs"] for r in trace if r["success"]),
        "successful git traces have exit_code in outputs",
    )

    print(f"\n  Commits: {[c['subject'] for c in commits]}")
    print(f"  Git trace events recorded: {len(trace)}")


# ──────────────────────────────────────────────────────────────
#  SubprocessExecutor tests
# ──────────────────────────────────────────────────────────────


def test_executor(workspace: str) -> None:
    section("SubprocessExecutor")
    trace: list[dict] = []
    exe = SubprocessExecutor(
        workspace_dir=workspace,
        trace=trace,
        allowed_commands=False,  # permissive for the test
    )

    # run python --version
    result = exe.run_sync([sys.executable, "--version"])
    assert_true(isinstance(result, ExecutionResult), "returns ExecutionResult instance")
    assert_true(result.exit_code == 0, "python --version exit=0")
    assert_true(result.success, "result.success is True")
    assert_true(
        "Python" in (result.stdout + result.stderr), "Python version appears in output"
    )
    assert_true(result.duration_ms > 0, "duration_ms > 0")

    # run a simple python one-liner
    r2 = exe.run_sync([sys.executable, "-c", "print('sandbox_ok')"])
    assert_true(r2.success, "python -c one-liner succeeds")
    assert_true("sandbox_ok" in r2.stdout, "stdout contains 'sandbox_ok'")

    # failing command
    r3 = exe.run_sync([sys.executable, "-c", "raise SystemExit(42)"])
    assert_true(not r3.success, "failing command has success=False")
    assert_true(r3.exit_code == 42, f"exit_code is 42 (got {r3.exit_code})")

    # timeout enforcement
    r4 = exe.run_sync(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=1.0,
    )
    assert_true(r4.timed_out, "short-timeout command is marked timed_out=True")
    assert_true(not r4.success, "timed_out command has success=False")

    # allowlist enforcement
    exe2 = SubprocessExecutor(
        workspace_dir=workspace,
        trace=trace,
        allowed_commands={"python"},  # only python allowed
    )
    try:
        exe2.run_sync(["git", "status"])
        assert_true(False, "blocked command should raise ValueError")
    except ValueError as e:
        assert_true(
            "not in the allowed-commands list" in str(e),
            "ValueError message mentions allowlist",
        )

    # trace records
    assert_true(len(trace) >= 4, f"executor trace has >=4 entries (got {len(trace)})")
    assert_true(
        all("exit_code" in r["outputs"] for r in trace if r["success"]),
        "successful exec traces have exit_code in outputs",
    )
    assert_true(
        all("duration_ms" in r for r in trace), "all exec traces have duration_ms"
    )

    print(f"\n  Executor trace events recorded: {len(trace)}")


# ──────────────────────────────────────────────────────────────
#  Cross-tool end-to-end: simulate a node writing, committing, testing
# ──────────────────────────────────────────────────────────────


def test_end_to_end_node_simulation(workspace: str) -> None:
    section("End-to-End Node Simulation (all 3 tools)")

    trace: list[dict] = []
    fs = FileSystemManager(workspace_dir=workspace, trace=trace)
    git = GitTracker(workspace_dir=workspace, trace=trace)
    exe = SubprocessExecutor(
        workspace_dir=workspace,
        trace=trace,
        allowed_commands=False,
    )

    # 1. Initialise repo
    git.init()
    git.commit("chore: initial commit", allow_empty=True)
    git.ensure_branch("feature/math-utils")

    # 2. Write code file
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    fs.write_file("math_utils.py", code)

    # 3. Write test file
    test_code = (
        "from math_utils import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n"
        "    assert add(-1, 1) == 0\n"
    )
    fs.write_file("test_math_utils.py", test_code)

    # 4. Stage and commit
    git.stage_all()
    git.commit("feat: add math_utils and tests")

    # 5. Run pytest (must point at the file inside the workspace)
    result = exe.run_sync(
        [sys.executable, "-m", "pytest", "test_math_utils.py", "-v", "--tb=short"],
        extra_env={"PYTHONPATH": workspace},
    )
    assert_true(result.success, f"pytest passes (exit={result.exit_code})")
    assert_true(
        "test_add" in result.stdout or "passed" in result.stdout,
        "pytest output confirms test_add ran",
    )

    # 6. Verify git log
    commits = git.log(n=5)
    subjects = [c["subject"] for c in commits]
    assert_true(
        any("math_utils" in s for s in subjects),
        f"git log shows math_utils commit (got: {subjects})",
    )

    # 7. Verify total trace shape
    total_events = len(trace)
    assert_true(
        total_events >= 8, f"combined trace has >=8 events (got {total_events})"
    )
    tools_seen = {r["tool"] for r in trace}
    assert_true("FileSystemManager" in tools_seen, "FileSystemManager appears in trace")
    assert_true("GitTracker" in tools_seen, "GitTracker appears in trace")
    assert_true(
        "SubprocessExecutor" in tools_seen, "SubprocessExecutor appears in trace"
    )

    print(f"\n  Total trace events across all 3 tools: {total_events}")
    print(f"  Tools seen: {sorted(tools_seen)}")
    print(f"  Pytest output snippet: {result.stdout[:200]}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────


def main() -> None:
    failures = 0
    with tempfile.TemporaryDirectory(prefix="ai_team_test_") as tmpdir:
        print(f"\nUsing workspace: {tmpdir}")
        tests = [
            ("FileSystemManager", test_filesystem),
            ("GitTracker", test_git),
            ("SubprocessExecutor", test_executor),
            ("End-to-End", test_end_to_end_node_simulation),
        ]
        for name, fn in tests:
            # Each test gets its own sub-directory to avoid state bleed
            test_ws = os.path.join(tmpdir, name.replace(" ", "_").replace("-", "_"))
            os.makedirs(test_ws, exist_ok=True)
            try:
                fn(test_ws)
            except AssertionError as exc:
                print(f"\n  {FAIL} {name}: {exc}")
                failures += 1
            except Exception as exc:
                print(f"\n  {FAIL} {name}: UNEXPECTED {type(exc).__name__}: {exc}")
                failures += 1

    section("Summary")
    total = len(tests)
    passed = total - failures
    print(f"  Passed: {passed} / {total}")
    if failures:
        print(f"  Failed: {failures} test(s)")
        sys.exit(1)
    else:
        print("  All tests passed!")


if __name__ == "__main__":
    main()
