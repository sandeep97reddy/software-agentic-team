"""
GitTracker -- Chunk 3 Tool Layer
=================================

Manages a local Git repository inside the workspace directory.

Operations: init, ensure_branch, stage_all, commit, diff, status, log.

Every public method appends a TraceRecord to the shared ``trace`` list,
which is then merged into ``state["execution_trace"]`` via operator.add.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from src.tools.filesystem import _make_trace

logger = logging.getLogger(__name__)


class GitTracker:
    """
    Git operations scoped to a single workspace directory.

    Parameters
    ----------
    workspace_dir : str | Path
        Absolute path of the project workspace.
    trace : list[dict]
        Shared mutable trace list appended to ``execution_trace``.
    git_executable : str
        Path to the ``git`` binary (defaults to ``"git"`` on PATH).
    """

    TOOL_NAME = "GitTracker"

    _GIT_ENV: dict[str, str] = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",  # never prompt for credentials
        "GIT_ASKPASS": "echo",  # return empty string for any password prompt
    }

    def __init__(
        self,
        workspace_dir: str | Path,
        trace: list[dict[str, Any]],
        git_executable: str = "git",
    ) -> None:
        self._root = Path(workspace_dir).resolve()
        self._trace = trace
        self._git = git_executable

    # ----------------------------------------------------------
    #  Internal runner
    # ----------------------------------------------------------

    def _run(
        self,
        *args: str,
        operation: str,
        extra_inputs: dict[str, Any] | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a git sub-command, record output, append trace."""
        cmd = [self._git, *args]
        inputs: dict[str, Any] = {
            "command": " ".join(cmd),
            "cwd": str(self._root),
            **(extra_inputs or {}),
        }
        start = time.perf_counter()

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._GIT_ENV,
                check=check,
            )
            elapsed_ms = (time.perf_counter() - start) * 1_000
            success = result.returncode == 0
            outputs = {
                "exit_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
            record = _make_trace(
                self.TOOL_NAME,
                operation,
                inputs,
                outputs,
                success=success,
                error=result.stderr.strip() if not success else None,
                duration_ms=round(elapsed_ms, 3),
            )
            log_fn = logger.info if success else logger.warning
            log_fn("[GIT] %s exit=%d", operation, result.returncode)

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1_000
            record = _make_trace(
                self.TOOL_NAME,
                operation,
                inputs,
                {},
                success=False,
                error=str(exc),
                duration_ms=round(elapsed_ms, 3),
            )
            logger.error("[GIT] %s EXCEPTION: %s", operation, exc)
            self._trace.append(record)
            raise

        self._trace.append(record)
        return result

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------

    def init(self, default_branch: str = "main") -> subprocess.CompletedProcess[str]:
        """
        Initialise a git repository in the workspace (idempotent).

        Sets a minimal local user identity so commits can be created
        without a system-wide git config.
        """
        result = self._run(
            "init",
            "-b",
            default_branch,
            operation="init",
            extra_inputs={"default_branch": default_branch},
        )
        # Configure local identity for headless operation
        self._run(
            "config", "user.email", "agent@ai-team.local", operation="config_email"
        )
        self._run(
            "config",
            "user.name",
            "AI Software Engineering Team",
            operation="config_name",
        )
        return result

    def ensure_branch(self, branch_name: str) -> subprocess.CompletedProcess[str]:
        """
        Create *branch_name* if it does not exist and check it out.

        Uses ``git switch -C`` (create-or-reset) for idempotency.
        Driven by ``state["active_branch"]``.
        """
        logger.info("[GIT] Switching to branch: %s", branch_name)
        return self._run(
            "switch",
            "-C",
            branch_name,
            operation="ensure_branch",
            extra_inputs={"branch_name": branch_name},
        )

    def stage_all(self) -> subprocess.CompletedProcess[str]:
        """Stage all modified and untracked files (``git add -A``)."""
        return self._run("add", "-A", operation="stage_all")

    def commit(
        self,
        message: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """
        Commit staged changes.

        Parameters
        ----------
        message : str
            Commit message.
        allow_empty : bool
            If True, passes ``--allow-empty`` (useful for initial commits).
        """
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        return self._run(
            *args,
            operation="commit",
            extra_inputs={"message": message, "allow_empty": allow_empty},
        )

    def diff(self, staged: bool = False, max_bytes: int = 64_000) -> str:
        """
        Capture the current diff as a string.

        Parameters
        ----------
        staged : bool
            Diff against the staged index (``--cached``) instead of HEAD.
        max_bytes : int
            Truncate the diff at this many bytes.

        Returns
        -------
        str  Raw unified diff text.
        """
        args = ["diff"]
        if staged:
            args.append("--cached")
        args.append("HEAD")
        result = self._run(*args, operation="diff", extra_inputs={"staged": staged})
        raw = result.stdout
        if len(raw) > max_bytes:
            raw = raw[:max_bytes] + f"\n... [truncated at {max_bytes} bytes]"
        return raw

    def status(self) -> list[dict[str, str]]:
        """
        Return a parsed list of changed files from ``git status --porcelain``.

        Each entry: ``{"status": "M", "path": "src/main.py"}``.
        """
        result = self._run("status", "--porcelain", operation="status")
        entries: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            if len(line) >= 3:
                entries.append(
                    {
                        "status": line[:2].strip(),
                        "path": line[3:].strip(),
                    }
                )
        return entries

    def log(self, n: int = 10) -> list[dict[str, str]]:
        """
        Return the last *n* commits as structured records.

        Each record: ``{"hash": ..., "author": ..., "date": ..., "subject": ...}``.
        """
        fmt = "%H|%an|%ad|%s"
        result = self._run(
            "log",
            f"-{n}",
            f"--pretty=format:{fmt}",
            "--date=iso-strict",
            operation="log",
            extra_inputs={"n": n},
        )
        entries: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entries.append(
                    {
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "subject": parts[3],
                    }
                )
        return entries

    def current_branch(self) -> str:
        """Return the name of the currently checked-out branch."""
        result = self._run(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
            operation="current_branch",
        )
        return result.stdout.strip()

    @property
    def root(self) -> Path:
        """The absolute Path of the workspace sandbox root."""
        return self._root
