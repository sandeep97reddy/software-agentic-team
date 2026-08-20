"""
Unit tests for LangGraph State Reducers (Milestone M1).

Tests all custom reducers across normal, boundary, and corner cases:
- clearable_list_reducer & ClearSignal sentinel
- artifact_reducer (version bump on edit, deduplication by file_path)
- task_queue_reducer (delta updates, queue order preservation, metadata merge)
- dict_merge_reducer (key preservation, recursive merge, counter separation)
- adr_reducer (deduplication by decision_id)
- ProjectState schema integration and Pydantic boundary models
"""

import copy
from typing import Any, get_type_hints
import pytest

from src.core.reducers import (
    CLEAR,
    ClearSignal,
    adr_reducer,
    artifact_reducer,
    clearable_list_reducer,
    dict_merge_reducer,
    task_queue_reducer,
)
from src.core.state import (
    ADR,
    ArchitectureDecision,
    CodeArtifact,
    ErrorRecord,
    ExecutionTraceItem,
    ProjectState,
    TaskItem,
)


# =====================================================================
# 1. ClearSignal & clearable_list_reducer Tests
# =====================================================================

class TestClearableListReducer:
    def test_none_inputs(self):
        """None input handling should never raise or append [None]."""
        assert clearable_list_reducer(None, None) == []
        assert clearable_list_reducer(["a", "b"], None) == ["a", "b"]
        assert clearable_list_reducer(None, ["a", "b"]) == ["a", "b"]

    def test_normal_append_list_and_scalar(self):
        """Appending lists and single scalar elements."""
        assert clearable_list_reducer([1, 2], [3, 4]) == [1, 2, 3, 4]
        assert clearable_list_reducer([1, 2], 3) == [1, 2, 3]
        assert clearable_list_reducer([], []) == []

    def test_clear_signal_sentinel(self):
        """ClearSignal clears existing items and handles payload items."""
        # Standalone CLEAR sentinel
        assert clearable_list_reducer([1, 2, 3], ClearSignal()) == []
        assert clearable_list_reducer([1, 2, 3], CLEAR) == []
        assert clearable_list_reducer([1, 2, 3], [CLEAR]) == []

        # CLEAR followed by new items
        result = clearable_list_reducer([1, 2, 3], [CLEAR, 4, 5])
        assert result == [4, 5]

    def test_legacy_string_clear(self):
        """String 'CLEAR' clears list for backward compatibility."""
        assert clearable_list_reducer(["a", "b"], "CLEAR") == []
        assert clearable_list_reducer(["a", "b"], ["CLEAR"]) == []
        assert clearable_list_reducer(["a", "b"], ["CLEAR", "c", "d"]) == ["c", "d"]

    def test_immutability(self):
        """Reducer should return fresh list without mutating inputs."""
        existing = [1, 2]
        new_items = [3, 4]
        res = clearable_list_reducer(existing, new_items)
        res.append(99)
        assert existing == [1, 2]
        assert new_items == [3, 4]

    def test_list_with_none_elements_filtered(self):
        """None elements within list payload should be cleanly filtered."""
        existing = ["x"]
        new_items = ["y", None, "z"]
        assert clearable_list_reducer(existing, new_items) == ["x", "y", "z"]

    def test_clearsignal_properties(self):
        """Verify ClearSignal representation, equality and hash."""
        sig = ClearSignal()
        assert repr(sig) == "<CLEAR>"
        assert str(sig) == "<CLEAR>"
        assert sig == "CLEAR"
        assert sig == ClearSignal()
        assert hash(sig) == hash("CLEAR")


# =====================================================================
# 2. artifact_reducer Tests
# =====================================================================

class TestArtifactReducer:
    def test_initial_and_none(self):
        """Initial artifact population and None safety."""
        assert artifact_reducer(None, None) == []
        assert artifact_reducer([{"file_path": "a.py"}], None) == [{"file_path": "a.py"}]
        res = artifact_reducer(None, [{"file_path": "a.py", "content": "x = 1"}])
        assert len(res) == 1
        assert res[0]["file_path"] == "a.py"
        assert res[0]["version"] == 1
        assert res[0]["language"] == "python"

    def test_disjoint_files(self):
        """Adding different files preserves all artifacts and order."""
        existing = [{"file_path": "a.py", "content": "x = 1", "version": 1}]
        new_art = [{"file_path": "b.py", "content": "y = 2"}]
        res = artifact_reducer(existing, new_art)
        assert len(res) == 2
        assert res[0]["file_path"] == "a.py"
        assert res[1]["file_path"] == "b.py"
        assert res[1]["version"] == 1

    def test_content_change_increments_version(self):
        """When file content changes, version is automatically incremented."""
        existing = [{"file_path": "models.py", "content": "class User: pass", "version": 1}]
        new_art = [{"file_path": "models.py", "content": "class User:\n    id: int", "version": 1}]
        res = artifact_reducer(existing, new_art)
        assert len(res) == 1
        assert res[0]["file_path"] == "models.py"
        assert res[0]["version"] == 2
        assert "id: int" in res[0]["content"]

    def test_same_content_preserves_version(self):
        """Updating metadata with identical content does not increment version."""
        existing = [
            {"file_path": "utils.py", "content": "def add(): pass", "version": 2, "tests_passed": None}
        ]
        new_art = [{"file_path": "utils.py", "content": "def add(): pass", "tests_passed": True}]
        res = artifact_reducer(existing, new_art)
        assert len(res) == 1
        assert res[0]["version"] == 2
        assert res[0]["tests_passed"] is True

    def test_pydantic_code_artifact_support(self):
        """Supports CodeArtifact Pydantic model inputs."""
        existing = [CodeArtifact(file_path="app.py", content="x = 1", version=1)]
        new_art = CodeArtifact(file_path="app.py", content="x = 2")
        res = artifact_reducer(existing, new_art)
        assert len(res) == 1
        item = res[0]
        fp = item["file_path"]
        ver = item["version"]
        assert fp == "app.py"
        assert ver == 2

    def test_batch_updates_and_order(self):
        """Multiple updates in single batch process sequentially."""
        existing = [{"file_path": "a.py", "content": "v1", "version": 1}]
        updates = [
            {"file_path": "a.py", "content": "v2"},
            {"file_path": "b.py", "content": "b1"},
            {"file_path": "a.py", "content": "v3"},
        ]
        res = artifact_reducer(existing, updates)
        assert len(res) == 2
        assert res[0]["file_path"] == "a.py"
        assert res[0]["version"] == 3
        assert res[0]["content"] == "v3"
        assert res[1]["file_path"] == "b.py"
        assert res[1]["version"] == 1

    def test_clear_signal_resets_artifacts(self):
        """ClearSignal clears artifact list or resets with payload."""
        existing = [{"file_path": "a.py", "content": "v1", "version": 1}]
        assert artifact_reducer(existing, CLEAR) == []
        assert artifact_reducer(existing, "CLEAR") == []
        res = artifact_reducer(existing, [CLEAR, {"file_path": "fresh.py", "content": "new"}])
        assert len(res) == 1
        assert res[0]["file_path"] == "fresh.py"

    def test_explicit_version_override(self):
        """Explicitly passed higher version is honored."""
        existing = [{"file_path": "a.py", "content": "v1", "version": 1}]
        res = artifact_reducer(existing, [{"file_path": "a.py", "content": "v1", "version": 5}])
        assert res[0]["version"] == 5

    def test_content_change_resets_tests_passed(self):
        """Modifying content resets tests_passed to None unless explicitly provided."""
        existing = [{"file_path": "a.py", "content": "v1", "version": 1, "tests_passed": True}]
        res = artifact_reducer(existing, [{"file_path": "a.py", "content": "v2"}])
        assert res[0]["tests_passed"] is None

        # Explicit tests_passed is preserved
        res2 = artifact_reducer(existing, [{"file_path": "a.py", "content": "v3", "tests_passed": True}])
        assert res2[0]["tests_passed"] is True

    def test_malformed_and_none_entries_filtered(self):
        """Non-dict / None / invalid entries are safely ignored."""
        existing = [{"file_path": "a.py", "content": "v1"}]
        res = artifact_reducer(existing, [None, "invalid_str", {"no_path": 1}, {"file_path": "b.py"}])
        assert len(res) == 2
        assert res[1]["file_path"] == "b.py"


# =====================================================================
# 3. task_queue_reducer Tests
# =====================================================================

class TestTaskQueueReducer:
    def test_none_and_empty(self):
        """None and empty handling for task queue."""
        assert task_queue_reducer(None, None) == []
        assert task_queue_reducer([{"task_id": "T1"}], None) == [{"task_id": "T1"}]

    def test_delta_status_update_preserves_order(self):
        """Updating task status does not wipe queue or alter task order."""
        existing = [
            {"task_id": "T1", "title": "Init", "status": "pending", "priority": 1},
            {"task_id": "T2", "title": "Build", "status": "pending", "priority": 2},
            {"task_id": "T3", "title": "Test", "status": "pending", "priority": 3},
        ]
        update = [{"task_id": "T2", "status": "in_progress"}]
        res = task_queue_reducer(existing, update)
        assert len(res) == 3
        assert [t["task_id"] for t in res] == ["T1", "T2", "T3"]
        assert res[1]["status"] == "in_progress"
        assert res[1]["title"] == "Build"
        assert res[0]["status"] == "pending"

    def test_metadata_merge(self):
        """Task metadata updates merge dictionary keys."""
        existing = [{"task_id": "T1", "metadata": {"retries": 1, "owner": "backend"}}]
        update = [{"task_id": "T1", "metadata": {"retries": 2, "error": "timeout"}}]
        res = task_queue_reducer(existing, update)
        assert len(res) == 1
        assert res[0]["metadata"] == {"retries": 2, "owner": "backend", "error": "timeout"}

    def test_append_new_tasks(self):
        """Disjoint task_ids are appended cleanly to queue."""
        existing = [{"task_id": "T1", "status": "completed"}]
        new_tasks = [{"task_id": "T2", "status": "pending"}]
        res = task_queue_reducer(existing, new_tasks)
        assert len(res) == 2
        assert [t["task_id"] for t in res] == ["T1", "T2"]

    def test_pydantic_task_item_support(self):
        """Supports TaskItem Pydantic models as input."""
        existing = [TaskItem(task_id="T1", title="Task 1", status="pending")]
        update = TaskItem(task_id="T1", title="Task 1", status="completed")
        res = task_queue_reducer(existing, update)
        assert len(res) == 1
        status = res[0]["status"]
        assert status == "completed"

    def test_clear_signal_resets_queue(self):
        """ClearSignal resets task queue."""
        existing = [{"task_id": "T1", "status": "completed"}]
        assert task_queue_reducer(existing, CLEAR) == []
        assert task_queue_reducer(existing, "CLEAR") == []
        res = task_queue_reducer(existing, [CLEAR, {"task_id": "T_NEW", "title": "New"}])
        assert len(res) == 1
        assert res[0]["task_id"] == "T_NEW"

    def test_immutability(self):
        """Task queue reducer returns new objects without mutating existing inputs."""
        orig_existing = [{"task_id": "T1", "title": "Original"}]
        res = task_queue_reducer(orig_existing, [{"task_id": "T1", "title": "Modified"}])
        assert orig_existing[0]["title"] == "Original"
        assert res[0]["title"] == "Modified"


# =====================================================================
# 4. dict_merge_reducer Tests
# =====================================================================

class TestDictMergeReducer:
    def test_basic_merge_and_overwrites(self):
        """Merges disjoint keys and updates overlapping keys."""
        existing = {"a": 1, "b": 2}
        new_dict = {"b": 20, "c": 30}
        res = dict_merge_reducer(existing, new_dict)
        assert res == {"a": 1, "b": 20, "c": 30}

    def test_none_safety(self):
        """Handles None inputs gracefully."""
        assert dict_merge_reducer(None, None) == {}
        assert dict_merge_reducer({"a": 1}, None) == {"a": 1}
        assert dict_merge_reducer(None, {"b": 2}) == {"b": 2}

    def test_nested_dict_recursive_merge(self):
        """Recursively merges nested dictionaries."""
        existing = {"level1": {"a": 1, "b": 2}, "top": 10}
        new_dict = {"level1": {"b": 20, "c": 30}, "extra": 99}
        res = dict_merge_reducer(existing, new_dict)
        assert res == {
            "level1": {"a": 1, "b": 20, "c": 30},
            "top": 10,
            "extra": 99,
        }

    def test_counter_separation(self):
        """Separates retry counts from task failure counters."""
        retry_counts = dict_merge_reducer({"backend_engineer": 1}, {"tester": 0})
        task_failures = dict_merge_reducer({"src/app.py": 1}, {"src/models.py": 2})
        assert retry_counts == {"backend_engineer": 1, "tester": 0}
        assert task_failures == {"src/app.py": 1, "src/models.py": 2}


# =====================================================================
# 5. adr_reducer Tests
# =====================================================================

class TestAdrReducer:
    def test_deduplication_and_order(self):
        """ADRs are deduplicated by decision_id preserving insertion order."""
        existing = [{"decision_id": "ADR-001", "title": "FastAPI", "status": "proposed"}]
        new_adr = [{"decision_id": "ADR-002", "title": "Postgres", "status": "proposed"}]
        res = adr_reducer(existing, new_adr)
        assert len(res) == 2
        assert [a["decision_id"] for a in res] == ["ADR-001", "ADR-002"]

    def test_update_existing_adr(self):
        """Re-running architect updates existing ADRs without duplicating them."""
        existing = [{"decision_id": "ADR-001", "title": "FastAPI", "status": "proposed"}]
        update = [{"decision_id": "ADR-001", "status": "accepted", "consequences": "Fast"}]
        res = adr_reducer(existing, update)
        assert len(res) == 1
        assert res[0]["decision_id"] == "ADR-001"
        assert res[0]["status"] == "accepted"
        assert res[0]["consequences"] == "Fast"

    def test_pydantic_adr_model_support(self):
        """Supports ArchitectureDecision Pydantic models."""
        existing = [ArchitectureDecision(decision_id="ADR-01", title="ADR 1", status="proposed")]
        update = ArchitectureDecision(decision_id="ADR-01", title="ADR 1", status="accepted")
        res = adr_reducer(existing, update)
        assert len(res) == 1
        assert res[0]["decision_id"] == "ADR-01"
        assert res[0]["status"] == "accepted"

    def test_clear_signal_resets_adrs(self):
        """ClearSignal clears ADRs."""
        existing = [{"decision_id": "ADR-001", "title": "FastAPI"}]
        assert adr_reducer(existing, CLEAR) == []
        assert adr_reducer(existing, "CLEAR") == []


# =====================================================================
# 6. ProjectState & Pydantic Boundary Model Tests
# =====================================================================

class TestProjectStateAndBoundaryModels:
    def test_pydantic_models_validation(self):
        """Verify boundary Pydantic models validate and serialize properly."""
        task = TaskItem(
            task_id="TASK-100",
            title="Setup DB",
            file_path="src/db.py",
            acceptance_criteria=["Table exists", "Indexes created"],
            related_requirements=["FR-1"],
        )
        assert task.task_id == "TASK-100"
        dump = task.model_dump()
        assert dump["file_path"] == "src/db.py"

        artifact = CodeArtifact(
            file_path="src/main.py",
            content="print(1)",
        )
        assert artifact.version == 1
        assert artifact.tests_passed is None

        adr = ADR(
            decision_id="ADR-01",
            title="Use Redis",
            alternatives_considered=["Memcached"],
        )
        assert adr.status == "proposed"

        err = ErrorRecord(
            node_name="tester",
            error_type="AssertionError",
            error_message="Test failed",
        )
        assert err.resolved is False

        trace_item = ExecutionTraceItem(
            tool="filesystem",
            operation="write_file",
            inputs={"path": "a.txt"},
        )
        assert trace_item.success is True

    def test_project_state_type_annotations(self):
        """Ensure ProjectState has all expected field annotations."""
        hints = get_type_hints(ProjectState, include_extras=True)
        assert "task_failures" in hints
        assert "retry_counts" in hints
        assert "task_queue" in hints
        assert "code_artifacts" in hints
        assert "completed_tasks" in hints
        assert "execution_trace" in hints
        assert "error_log" in hints
        assert "architecture_decisions" in hints
