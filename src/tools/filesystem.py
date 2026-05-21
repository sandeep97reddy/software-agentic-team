"""
FileSystemManager -- Chunk 3 Tool Layer
========================================

Provides sandboxed read / write / list operations on the designated
workspace directory.  All paths are *canonicalised* and validated to
stay inside the workspace root, preventing path-traversal attacks.

Every method appends a structured ``TraceRecord`` dict to the caller's
running ``execution_trace`` list, which is then merged into the
``ProjectState`` via ``operator.add``.

Usage inside a LangGraph node
------------------------------
    trace: list[dict] = []
    fs = FileSystemManager(workspace_dir=state["workspace_dir"], trace=trace)

    fs.write_file("src/main.py", "print('hello')")
    content = fs.read_file("src/main.py")
    entries  = fs.list_dir("src")

    # return trace to the state
    return {"execution_trace": trace, ...}
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Trace-record builder (shared with all tool classes)
# ──────────────────────────────────────────────────────────────


def _make_trace(
    tool: str,
    operation: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    success: bool,
    error: str | None = None,
    duration_ms: float = 0.0,
) -> dict[str, Any]:
    """Return a fully-typed trace-record dict."""
    return {
        "tool": tool,
        "operation": operation,
        "inputs": inputs,
        "outputs": outputs,
        "success": success,
        "error": error,
        "duration_ms": round(duration_ms, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────
#  FileSystemManager
# ──────────────────────────────────────────────────────────────


class FileSystemManager:
    """
    Safe file-system operations confined to a single workspace directory.

    Parameters
    ----------
    workspace_dir : str | Path
        Absolute path of the sandboxed root directory.  Created on first
        use if it does not exist.
    trace : list[dict]
        Mutable list that each method appends a ``TraceRecord`` to.
        Pass the same list to all tools in a node so every tool call is
        captured in a single execution_trace batch.
    """

    TOOL_NAME = "FileSystemManager"

    def __init__(
        self,
        workspace_dir: str | Path,
        trace: list[dict[str, Any]],
    ) -> None:
        self._root = Path(workspace_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._trace = trace
        logger.info("[FS] Workspace root: %s", self._root)

    # ----------------------------------------------------------
    #  Internal helpers
    # ----------------------------------------------------------

    def _safe_resolve(self, relative_path: str) -> Path:
        """
        Resolve *relative_path* against the workspace root and verify it
        does not escape the sandbox.

        Raises
        ------
        PermissionError
            If the resolved path escapes the workspace root.
        """
        # Strip leading slashes / backslashes to force relative resolution
        cleaned = relative_path.lstrip("/\\")
        resolved = (self._root / cleaned).resolve()

        # Strict containment check
        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise PermissionError(
                f"Path traversal denied: '{relative_path}' resolves outside "
                f"workspace root '{self._root}'"
            )
        return resolved

    def _sha256(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _elapsed(self, start: float) -> float:
        import time

        return (time.perf_counter() - start) * 1_000

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------

    def write_file(
        self,
        relative_path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """
        Write *content* to *relative_path* inside the workspace.

        Creates parent directories automatically.  Overwrites any
        existing content.

        Returns
        -------
        dict
            Trace record with ``bytes_written`` and ``sha256`` of content.
        """
        import time

        start = time.perf_counter()
        inputs = {
            "relative_path": relative_path,
            "bytes": len(content.encode(encoding)),
        }

        try:
            target = self._safe_resolve(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding=encoding)
            checksum = self._sha256(content)
            outputs = {
                "absolute_path": str(target),
                "bytes_written": len(content.encode(encoding)),
                "sha256": checksum,
            }
            record = _make_trace(
                self.TOOL_NAME,
                "write_file",
                inputs,
                outputs,
                success=True,
                duration_ms=self._elapsed(start),
            )
            logger.info("[FS] WRITE %s (%d bytes)", target, outputs["bytes_written"])
        except Exception as exc:
            outputs = {}
            record = _make_trace(
                self.TOOL_NAME,
                "write_file",
                inputs,
                outputs,
                success=False,
                error=str(exc),
                duration_ms=self._elapsed(start),
            )
            logger.error("[FS] WRITE FAILED %s: %s", relative_path, exc)
            raise

        self._trace.append(record)
        return record

    def read_file(
        self,
        relative_path: str,
        encoding: str = "utf-8",
    ) -> str:
        """
        Read and return the content of *relative_path*.

        Raises
        ------
        FileNotFoundError
            If the file does not exist inside the workspace.

        Returns
        -------
        str
            Full file content.
        """
        import time

        start = time.perf_counter()
        inputs = {"relative_path": relative_path}
        content: str = ""

        try:
            target = self._safe_resolve(relative_path)
            if not target.is_file():
                raise FileNotFoundError(
                    f"No such file: '{relative_path}' (resolved: {target})"
                )
            content = target.read_text(encoding=encoding)
            outputs = {
                "absolute_path": str(target),
                "bytes_read": len(content.encode(encoding)),
                "sha256": self._sha256(content),
                # Return first 120 chars of content as a preview only
                "preview": content[:120],
            }
            record = _make_trace(
                self.TOOL_NAME,
                "read_file",
                inputs,
                outputs,
                success=True,
                duration_ms=self._elapsed(start),
            )
            logger.info("[FS] READ %s (%d bytes)", target, outputs["bytes_read"])
        except Exception as exc:
            outputs = {}
            record = _make_trace(
                self.TOOL_NAME,
                "read_file",
                inputs,
                outputs,
                success=False,
                error=str(exc),
                duration_ms=self._elapsed(start),
            )
            logger.error("[FS] READ FAILED %s: %s", relative_path, exc)
            raise

        self._trace.append(record)
        return content

    def list_dir(
        self,
        relative_path: str = ".",
        *,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List the contents of a directory inside the workspace.

        Parameters
        ----------
        relative_path : str
            Directory to list (relative to workspace root).
        recursive : bool
            If True, walk all subdirectories.

        Returns
        -------
        list[dict]
            Each entry has ``name``, ``type`` ("file" | "directory"),
            ``size_bytes`` (0 for directories), ``relative_path``.
        """
        import time

        start = time.perf_counter()
        inputs = {"relative_path": relative_path, "recursive": recursive}
        entries: list[dict[str, Any]] = []

        try:
            target = self._safe_resolve(relative_path)
            if not target.is_dir():
                raise NotADirectoryError(f"Not a directory: '{relative_path}'")

            iterator = target.rglob("*") if recursive else target.iterdir()
            for item in sorted(iterator):
                rel = item.relative_to(self._root)
                entries.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size_bytes": item.stat().st_size if item.is_file() else 0,
                        "relative_path": rel.as_posix(),
                    }
                )

            outputs = {"entry_count": len(entries), "entries": entries}
            record = _make_trace(
                self.TOOL_NAME,
                "list_dir",
                inputs,
                outputs,
                success=True,
                duration_ms=self._elapsed(start),
            )
            logger.info("[FS] LIST %s -> %d entries", target, len(entries))
        except Exception as exc:
            outputs = {}
            record = _make_trace(
                self.TOOL_NAME,
                "list_dir",
                inputs,
                outputs,
                success=False,
                error=str(exc),
                duration_ms=self._elapsed(start),
            )
            logger.error("[FS] LIST FAILED %s: %s", relative_path, exc)
            raise

        self._trace.append(record)
        return entries

    def delete_file(self, relative_path: str) -> dict[str, Any]:
        """
        Delete a single file from the workspace.

        Returns
        -------
        dict
            Trace record confirming deletion.
        """
        import time

        start = time.perf_counter()
        inputs = {"relative_path": relative_path}

        try:
            target = self._safe_resolve(relative_path)
            if not target.is_file():
                raise FileNotFoundError(f"No such file: '{relative_path}'")
            size = target.stat().st_size
            target.unlink()
            outputs = {"absolute_path": str(target), "bytes_freed": size}
            record = _make_trace(
                self.TOOL_NAME,
                "delete_file",
                inputs,
                outputs,
                success=True,
                duration_ms=self._elapsed(start),
            )
            logger.info("[FS] DELETE %s", target)
        except Exception as exc:
            outputs = {}
            record = _make_trace(
                self.TOOL_NAME,
                "delete_file",
                inputs,
                outputs,
                success=False,
                error=str(exc),
                duration_ms=self._elapsed(start),
            )
            logger.error("[FS] DELETE FAILED %s: %s", relative_path, exc)
            raise

        self._trace.append(record)
        return record

    def file_exists(self, relative_path: str) -> bool:
        """Return True if *relative_path* is a regular file inside the workspace."""
        try:
            return self._safe_resolve(relative_path).is_file()
        except PermissionError:
            return False

    @property
    def root(self) -> Path:
        """The absolute Path of the workspace sandbox root."""
        return self._root
