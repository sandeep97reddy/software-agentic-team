"""
src/core/reducers.py -- Custom State Reducers for LangGraph StateGraph.

Provides specialized reducer functions for merging state updates:
- ClearSignal & clearable_list_reducer: Append lists with clear support.
- artifact_reducer: Deduplicate CodeArtifacts by file_path, auto-increment version on change.
- task_queue_reducer: Upsert TaskItems by task_id, preserve order, merge metadata.
- dict_merge_reducer: Recursive dictionary merge without key clobbering.
- adr_reducer: Deduplicate ADRs by decision_id, preserve order.
"""

from __future__ import annotations

from typing import Any, Sequence


class ClearSignal:
    """
    Sentinel class used to clear a list without string collisions.
    Supports equality comparison with string 'CLEAR' for backward compatibility.
    """

    def __repr__(self) -> str:
        return "<CLEAR>"

    def __str__(self) -> str:
        return "<CLEAR>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClearSignal) or other == "CLEAR"

    def __hash__(self) -> int:
        return hash("CLEAR")


CLEAR = ClearSignal()


def clearable_list_reducer(
    existing: list[Any] | None,
    new: list[Any] | ClearSignal | str | Any | None,
) -> list[Any]:
    """
    Safely appends or clears list state.

    Rules:
    - If existing is None: initializes to [].
    - If new is None: returns a shallow copy of existing.
    - If new is ClearSignal or 'CLEAR': returns [].
    - If new is a list starting with ClearSignal or 'CLEAR': resets list to new[1:] (filtering None).
    - If new is a list: appends all non-None elements to existing copy.
    - If new is a single item (not None, not ClearSignal): appends item to existing copy.
    """
    if existing is None:
        existing = []
    if new is None:
        return list(existing)
    if isinstance(new, ClearSignal) or new == "CLEAR":
        return []
    if isinstance(new, (list, tuple)):
        if not new:
            return list(existing)
        if isinstance(new[0], ClearSignal) or new[0] == "CLEAR":
            return [x for x in new[1:] if x is not None]
        return list(existing) + [x for x in new if x is not None]
    return list(existing) + [new]


import posixpath


def _normalize_file_path(p: str) -> str:
    """Normalize file paths to unix-style relative paths without leading slashes/dots."""
    cleaned = p.replace("\\", "/")
    norm = posixpath.normpath(cleaned).lstrip("./").lstrip("/")
    return "" if norm == "." else norm


def artifact_reducer(
    existing: list[dict[str, Any] | Any] | None,
    new: list[dict[str, Any] | Any] | dict[str, Any] | Any | ClearSignal | str | None,
) -> list[dict[str, Any]]:
    """
    Deduplicates code artifacts by `file_path`.
    Automatically manages version increments when content changes.
    Preserves insertion order and safely updates metadata/test results.
    """
    if existing is None:
        existing = []
    if new is None:
        return [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in existing
            if isinstance(item, dict) or hasattr(item, "model_dump")
        ]
    if isinstance(new, ClearSignal) or new == "CLEAR":
        return []

    updates: list[Any]
    if isinstance(new, (list, tuple)):
        updates = list(new)
    else:
        updates = [new]

    artifact_map: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    if updates and (isinstance(updates[0], ClearSignal) or updates[0] == "CLEAR"):
        updates = updates[1:]
    else:
        for item in existing:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = dict(item)
            else:
                continue

            raw_fp = item_dict.get("file_path")
            if raw_fp and isinstance(raw_fp, str):
                fp = _normalize_file_path(raw_fp)
                item_dict["file_path"] = fp
                artifact_map[fp] = item_dict
                if fp not in order:
                    order.append(fp)

    for item in updates:
        if item is None:
            continue
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        elif isinstance(item, dict):
            item_dict = dict(item)
        else:
            continue

        raw_fp = item_dict.get("file_path")
        if not raw_fp or not isinstance(raw_fp, str):
            continue

        fp = _normalize_file_path(raw_fp)
        item_dict["file_path"] = fp

        if fp not in artifact_map:
            entry = dict(item_dict)
            if "version" not in entry or entry["version"] is None:
                entry["version"] = 1
            if "language" not in entry or entry["language"] is None:
                entry["language"] = "python"
            artifact_map[fp] = entry
            order.append(fp)
        else:
            prev = artifact_map[fp]
            prev_ver = prev.get("version") or 1
            prev_content = prev.get("content", "")
            new_content = item_dict.get("content", prev_content)

            # Check explicit version bump vs content change
            if item_dict.get("version") is not None and item_dict["version"] > prev_ver:
                new_ver = item_dict["version"]
            elif new_content != prev_content:
                new_ver = prev_ver + 1
            else:
                new_ver = prev_ver

            merged = {**prev, **item_dict, "file_path": fp, "version": new_ver}
            # If content changed and tests_passed not explicitly provided in update, reset tests_passed
            if new_content != prev_content and "tests_passed" not in item_dict:
                merged["tests_passed"] = None

            artifact_map[fp] = merged

    return [artifact_map[fp] for fp in order]


def task_queue_reducer(
    existing: list[dict[str, Any] | Any] | None,
    new: list[dict[str, Any] | Any] | dict[str, Any] | Any | ClearSignal | str | None,
) -> list[dict[str, Any]]:
    """
    Upserts task items by `task_id`.
    Preserves task sequence and atomically updates status/metadata.
    Supports ClearSignal and list resetting.
    """
    if existing is None:
        existing = []
    if new is None:
        return [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in existing
            if isinstance(item, dict) or hasattr(item, "model_dump")
        ]
    if isinstance(new, ClearSignal) or new == "CLEAR":
        return []

    updates: list[Any]
    if isinstance(new, (list, tuple)):
        updates = list(new)
    else:
        updates = [new]

    task_map: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    if updates and (isinstance(updates[0], ClearSignal) or updates[0] == "CLEAR"):
        updates = updates[1:]
    else:
        for item in existing:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = dict(item)
            else:
                continue

            tid = item_dict.get("task_id")
            if tid and isinstance(tid, str):
                task_map[tid] = item_dict
                if tid not in order:
                    order.append(tid)

    for item in updates:
        if item is None:
            continue
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        elif isinstance(item, dict):
            item_dict = dict(item)
        else:
            continue

        tid = item_dict.get("task_id")
        if not tid or not isinstance(tid, str):
            continue

        if tid not in task_map:
            task_map[tid] = dict(item_dict)
            order.append(tid)
        else:
            prev = task_map[tid]
            merged = {**prev, **item_dict}
            # Deep merge metadata if both contain it
            if isinstance(prev.get("metadata"), dict) and isinstance(item_dict.get("metadata"), dict):
                merged["metadata"] = {**prev["metadata"], **item_dict["metadata"]}
            task_map[tid] = merged

    return [task_map[tid] for tid in order]


def dict_merge_reducer(
    existing: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Safely merges dictionary updates without clobbering existing keys.
    Performs recursive merge for nested dictionaries.
    """
    if existing is None:
        existing = {}
    if new is None or not isinstance(new, dict):
        return dict(existing)

    result = dict(existing)
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = dict_merge_reducer(result[key], value)
        else:
            result[key] = value
    return result


def adr_reducer(
    existing: list[dict[str, Any] | Any] | None,
    new: list[dict[str, Any] | Any] | dict[str, Any] | Any | ClearSignal | str | None,
) -> list[dict[str, Any]]:
    """
    Deduplicates ADRs by `decision_id`.
    Preserves creation order while allowing updates.
    """
    if existing is None:
        existing = []
    if new is None:
        return [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in existing
            if isinstance(item, dict) or hasattr(item, "model_dump")
        ]
    if isinstance(new, ClearSignal) or new == "CLEAR":
        return []

    updates: list[Any]
    if isinstance(new, (list, tuple)):
        updates = list(new)
    else:
        updates = [new]

    adr_map: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    if updates and (isinstance(updates[0], ClearSignal) or updates[0] == "CLEAR"):
        updates = updates[1:]
    else:
        for item in existing:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = dict(item)
            else:
                continue

            did = item_dict.get("decision_id")
            if did and isinstance(did, str):
                adr_map[did] = item_dict
                if did not in order:
                    order.append(did)

    for item in updates:
        if item is None:
            continue
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        elif isinstance(item, dict):
            item_dict = dict(item)
        else:
            continue

        did = item_dict.get("decision_id")
        if not did or not isinstance(did, str):
            continue

        if did not in adr_map:
            adr_map[did] = dict(item_dict)
            order.append(did)
        else:
            adr_map[did] = {**adr_map[did], **item_dict}

    return [adr_map[did] for did in order]
