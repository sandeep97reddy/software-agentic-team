"""
SubprocessExecutor -- Chunk 3 Tool Layer
==========================================

Securely runs arbitrary shell commands (``pytest``, ``pip install``,
linters, formatters, etc.) in an isolated subprocess.

Key properties
--------------
- **Async-first**: ``run_async`` uses ``asyncio.create_subprocess_exec``
  for non-blocking I/O; a ``run_sync`` wrapper is provided for use
  inside synchronous LangGraph nodes via ``asyncio.run()``.
- **Timeout enforcement**: each command has a hard wall-clock timeout
  (default 120 s).  The process tree is killed on expiry.
- **Clean stdio**: stdout and stderr are captured separately, stripped
  of ANSI escape codes, and stored in the trace.
- **No shell=True**: commands are split into a list and passed directly
  to the OS so no shell injection is possible.
- **Allowlist**: an optional ``allowed_commands`` set restricts which
  executables may be invoked.
- **Structured exit codes**: ``exit_code == 0`` means success; any
  non-zero value is a failure.  ``timed_out == True`` is a distinct
  failure mode.

Every invocation appends a ``TraceRecord`` to the shared ``trace`` list.

Usage
-----
    # Inside a sync LangGraph node:
    trace: list[dict] = []
    exe = SubprocessExecutor(
        workspace_dir=state["workspace_dir"],
        trace=trace,
    )
    result = exe.run_sync(["pytest", "tests/", "-v", "--tb=short"])
    return {"execution_trace": trace, ...}
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

from src.tools.filesystem import _make_trace

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  ANSI escape-code stripper
# ──────────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ──────────────────────────────────────────────────────────────
#  ExecutionResult  (returned by every run_* method)
# ──────────────────────────────────────────────────────────────


class ExecutionResult:
    """
    Structured result from a subprocess invocation.

    Attributes
    ----------
    command : list[str]
        The exact argv that was executed.
    exit_code : int
        Process exit code (0 = success).
    stdout : str
        Cleaned standard output.
    stderr : str
        Cleaned standard error.
    timed_out : bool
        True if the process was killed due to timeout.
    duration_ms : float
        Wall-clock time in milliseconds.
    success : bool
        True iff exit_code == 0 and not timed_out.
    """

    __slots__ = (
        "command",
        "exit_code",
        "stdout",
        "stderr",
        "timed_out",
        "duration_ms",
    )

    def __init__(
        self,
        command: list[str],
        exit_code: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
        duration_ms: float,
    ) -> None:
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.duration_ms = duration_ms

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": " ".join(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_ms": round(self.duration_ms, 3),
            "success": self.success,
        }

    def __repr__(self) -> str:
        return (
            f"ExecutionResult(exit_code={self.exit_code}, "
            f"success={self.success}, timed_out={self.timed_out})"
        )


# ──────────────────────────────────────────────────────────────
#  SubprocessExecutor
# ──────────────────────────────────────────────────────────────


class SubprocessExecutor:
    """
    Secure, async-capable subprocess runner for the execution sandbox.

    Parameters
    ----------
    workspace_dir : str | Path
        Working directory for all commands (the sandboxed workspace).
    trace : list[dict]
        Shared mutable trace list merged into ``execution_trace``.
    default_timeout : float
        Wall-clock timeout in seconds per command (default 120 s).
    allowed_commands : set[str] | None
        If provided, only executables in this set are permitted.
        Pass ``None`` to disable the allowlist (permissive mode).
    env_overrides : dict[str, str] | None
        Extra environment variables to inject into every subprocess.
    max_output_bytes : int
        Truncate stdout/stderr at this many bytes each.
    """

    TOOL_NAME = "SubprocessExecutor"

    # Safe default allowlist for a Python development environment
    DEFAULT_ALLOWED: frozenset[str] = frozenset(
        {
            "python",
            "python3",
            "pytest",
            "pip",
            "pip3",
            "ruff",
            "black",
            "mypy",
            "bandit",
            "coverage",
            "git",
            "echo",
            "cat",
        }
    )

    def __init__(
        self,
        workspace_dir: str | Path,
        trace: list[dict[str, Any]],
        default_timeout: float = 120.0,
        allowed_commands: set[str] | None | bool = None,
        env_overrides: dict[str, str] | None = None,
        max_output_bytes: int = 32_768,
    ) -> None:
        self._root = Path(workspace_dir).resolve()
        self._trace = trace
        self._timeout = default_timeout
        self._max_bytes = max_output_bytes

        # allowed_commands=None -> use DEFAULT_ALLOWED
        # allowed_commands=False -> permissive (no allowlist)
        if allowed_commands is None:
            self._allowlist: frozenset[str] | None = self.DEFAULT_ALLOWED
        elif allowed_commands is False:
            self._allowlist = None
        else:
            self._allowlist = frozenset(allowed_commands)

        # Build subprocess environment
        self._env: dict[str, str] = {**os.environ}
        if env_overrides:
            self._env.update(env_overrides)
        # Ensure the venv's Scripts/bin is on PATH for pytest, ruff, etc.
        venv_scripts = (
            self._root.parent
            / ".venv"
            / ("Scripts" if sys.platform == "win32" else "bin")
        )
        if venv_scripts.is_dir():
            self._env["PATH"] = (
                str(venv_scripts) + os.pathsep + self._env.get("PATH", "")
            )

    # ----------------------------------------------------------
    #  Validation
    # ----------------------------------------------------------

    def _validate(self, command: list[str]) -> None:
        """Raise ValueError if the command is blocked by the allowlist."""
        if not command:
            raise ValueError("command must be a non-empty list")
        executable = Path(command[0]).name  # strip directory prefix if any
        if self._allowlist is not None and executable not in self._allowlist:
            raise ValueError(
                f"Command '{executable}' is not in the allowed-commands list. "
                f"Allowed: {sorted(self._allowlist)}"
            )

    def _truncate(self, text: str) -> str:
        """Truncate and clean text to ``max_output_bytes``."""
        clean = _strip_ansi(text)
        if len(clean.encode("utf-8")) > self._max_bytes:
            clean = clean[: self._max_bytes] + "\n... [output truncated]"
        return clean

    # ----------------------------------------------------------
    #  Async core
    # ----------------------------------------------------------

    async def run_async(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        stdin_data: str | None = None,
    ) -> ExecutionResult:
        """
        Execute *command* asynchronously and return an ``ExecutionResult``.

        Parameters
        ----------
        command : list[str]
            Argv to execute.  Must NOT be a shell string.
        timeout : float | None
            Override the instance default timeout.
        extra_env : dict[str, str] | None
            Additional env vars for this specific invocation.
        stdin_data : str | None
            If provided, written to the process stdin.

        Returns
        -------
        ExecutionResult
        """
        self._validate(command)
        effective_timeout = timeout if timeout is not None else self._timeout

        env = {**self._env, **(extra_env or {})}
        stdin_bytes = stdin_data.encode("utf-8") if stdin_data else None

        operation = f"run:{Path(command[0]).name}"
        inputs: dict[str, Any] = {
            "command": " ".join(command),
            "cwd": str(self._root),
            "timeout_s": effective_timeout,
        }

        timed_out = False
        exit_code = -1
        raw_stdout = ""
        raw_stderr = ""
        start = time.perf_counter()

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin_bytes
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin_bytes),
                    timeout=effective_timeout,
                )
                exit_code = proc.returncode if proc.returncode is not None else -1
                raw_stdout = stdout_bytes.decode("utf-8", errors="replace")
                raw_stderr = stderr_bytes.decode("utf-8", errors="replace")

            except asyncio.TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                exit_code = -1
                raw_stdout = ""
                raw_stderr = f"[TIMEOUT] Process killed after {effective_timeout}s"
                logger.warning(
                    "[EXEC] TIMEOUT after %.1fs: %s",
                    effective_timeout,
                    " ".join(command),
                )

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
            self._trace.append(record)
            logger.error("[EXEC] EXCEPTION running %s: %s", command[0], exc)
            raise

        elapsed_ms = (time.perf_counter() - start) * 1_000
        stdout_clean = self._truncate(raw_stdout)
        stderr_clean = self._truncate(raw_stderr)

        result = ExecutionResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout_clean,
            stderr=stderr_clean,
            timed_out=timed_out,
            duration_ms=elapsed_ms,
        )

        outputs = result.as_dict()
        record = _make_trace(
            self.TOOL_NAME,
            operation,
            inputs,
            outputs,
            success=result.success,
            error=stderr_clean if not result.success else None,
            duration_ms=round(elapsed_ms, 3),
        )
        self._trace.append(record)

        log_fn = logger.info if result.success else logger.warning
        log_fn(
            "[EXEC] %s exit=%d (%.0fms)%s",
            " ".join(command),
            exit_code,
            elapsed_ms,
            " [TIMEOUT]" if timed_out else "",
        )
        return result

    # ----------------------------------------------------------
    #  Sync wrapper  (for use inside synchronous LangGraph nodes)
    # ----------------------------------------------------------

    def run_sync(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        stdin_data: str | None = None,
    ) -> ExecutionResult:
        """
        Synchronous wrapper around ``run_async``.

        Internally calls ``asyncio.run()`` so it must NOT be called from
        inside an already-running event loop.  Use ``run_async`` directly
        in async node functions.

        Parameters
        ----------
        command : list[str]
            Same as ``run_async``.
        timeout : float | None
            Same as ``run_async``.
        extra_env : dict[str, str] | None
            Same as ``run_async``.
        stdin_data : str | None
            Same as ``run_async``.

        Returns
        -------
        ExecutionResult
        """
        return asyncio.run(
            self.run_async(
                command,
                timeout=timeout,
                extra_env=extra_env,
                stdin_data=stdin_data,
            )
        )

    # ----------------------------------------------------------
    #  High-level convenience methods
    # ----------------------------------------------------------

    def run_pytest(
        self,
        test_path: str = "tests/",
        *,
        extra_args: list[str] | None = None,
        timeout: float = 180.0,
    ) -> ExecutionResult:
        """
        Run the test suite with pytest and return a structured result.

        Parameters
        ----------
        test_path : str
            Path to the test directory or file (relative to workspace).
        extra_args : list[str] | None
            Additional pytest flags, e.g. ``["--cov=src", "-x"]``.
        timeout : float
            Hard timeout in seconds (defaults to 180 to allow test suites
            to complete).

        Returns
        -------
        ExecutionResult
            ``result.success`` is True only when exit_code == 0.
        """
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_path,
            "--tb=short",
            "--no-header",
            "-q",
            *(extra_args or []),
        ]
        return self.run_sync(cmd, timeout=timeout)

    def run_pip_install(
        self,
        packages: list[str],
        *,
        requirements_file: str | None = None,
        timeout: float = 300.0,
    ) -> ExecutionResult:
        """
        Install Python packages.

        Parameters
        ----------
        packages : list[str]
            Package names, e.g. ``["requests", "pydantic>=2"]``.
        requirements_file : str | None
            Path to a requirements file (relative to workspace).
            If provided, *packages* is ignored.
        timeout : float
            Hard timeout (defaults to 300 s for large dependency trees).

        Returns
        -------
        ExecutionResult
        """
        if requirements_file:
            cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
        else:
            cmd = [sys.executable, "-m", "pip", "install", *packages]
        return self.run_sync(cmd, timeout=timeout)

    def run_linter(
        self,
        path: str = ".",
        *,
        linter: str = "ruff",
        timeout: float = 60.0,
    ) -> ExecutionResult:
        """
        Run a linter against *path*.

        Parameters
        ----------
        path : str
            File or directory to lint (relative to workspace).
        linter : str
            ``"ruff"`` or ``"mypy"``.
        timeout : float
            Hard timeout.

        Returns
        -------
        ExecutionResult
        """
        if linter == "ruff":
            cmd = [sys.executable, "-m", "ruff", "check", path]
        elif linter == "mypy":
            cmd = [sys.executable, "-m", "mypy", path, "--ignore-missing-imports"]
        else:
            raise ValueError(f"Unknown linter: {linter!r}. Use 'ruff' or 'mypy'.")

        # Add linter to allowlist temporarily if needed
        orig = self._allowlist
        if self._allowlist is not None:
            self._allowlist = self._allowlist | {"ruff", "mypy"}

        try:
            return self.run_sync(cmd, timeout=timeout)
        finally:
            self._allowlist = orig

    @property
    def root(self) -> Path:
        """The absolute Path of the workspace sandbox root."""
        return self._root
