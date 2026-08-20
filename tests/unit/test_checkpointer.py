"""
Unit tests for Persistent Checkpointer & State Restoration (Milestone M1).

Tests:
- get_checkpointer() factory (memory, postgres fallback, redis fallback, strict mode)
- MemorySaver state snapshot persistence across StateGraph steps
- Thread isolation between different thread_id sessions
- State recovery and resumption across independent graph instances
- State history retrieval (get_state_history)
- Out-of-band state mutation injection (update_state)
- Graph compilation checkpointer configuration
"""

import os
from typing import Any
import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.core.checkpointer import get_checkpointer
from src.core.graph import build_graph
from src.core.observability import get_run_config
from src.core.state import ProjectState


# =====================================================================
# 1. get_checkpointer Factory Tests
# =====================================================================

class TestCheckpointerFactory:
    def test_default_checkpointer_returns_memory_saver(self):
        """Default checkpointer returns MemorySaver instance."""
        saver = get_checkpointer()
        assert isinstance(saver, BaseCheckpointSaver)
        assert isinstance(saver, MemorySaver)

    def test_explicit_memory_backend(self):
        """Explicit 'memory' backend returns MemorySaver."""
        saver = get_checkpointer(backend="memory")
        assert isinstance(saver, MemorySaver)

    def test_env_var_backend_selection(self, monkeypatch):
        """Honors CHECKPOINTER_BACKEND environment variable."""
        monkeypatch.setenv("CHECKPOINTER_BACKEND", "memory")
        saver = get_checkpointer()
        assert isinstance(saver, MemorySaver)

    def test_unknown_backend_falls_back_to_memory(self):
        """Invalid backend string falls back to MemorySaver without crashing."""
        saver = get_checkpointer(backend="unsupported_custom_backend")
        assert isinstance(saver, MemorySaver)

    def test_postgres_fallback_on_unreachable_host(self, monkeypatch):
        """Postgres backend falls back gracefully to MemorySaver on error."""
        monkeypatch.setenv("CHECKPOINTER_BACKEND", "postgres")
        monkeypatch.setenv(
            "POSTGRES_URL",
            "postgresql://invalid_user:invalid_pass@127.0.0.1:59999/nonexistent",
        )
        saver = get_checkpointer()
        assert isinstance(saver, BaseCheckpointSaver)
        assert isinstance(saver, MemorySaver)

    def test_redis_fallback_on_unreachable_host(self, monkeypatch):
        """Redis backend falls back gracefully to MemorySaver on error."""
        monkeypatch.setenv("CHECKPOINTER_BACKEND", "redis")
        monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:59999/0")
        saver = get_checkpointer()
        assert isinstance(saver, BaseCheckpointSaver)
        assert isinstance(saver, MemorySaver)

    def test_postgres_strict_mode_raises_when_fallback_disabled(self, monkeypatch):
        """When fallback_to_memory=False, connection/driver failure raises."""
        monkeypatch.setenv(
            "POSTGRES_URL",
            "postgresql://invalid_user:invalid_pass@127.0.0.1:59999/nonexistent",
        )
        with pytest.raises(Exception):
            get_checkpointer("postgres", fallback_to_memory=False)

    def test_redis_strict_mode_raises_when_fallback_disabled(self, monkeypatch):
        """When fallback_to_memory=False for redis, error raises."""
        monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:59999/0")
        with pytest.raises(Exception):
            get_checkpointer("redis", fallback_to_memory=False)


# =====================================================================
# 2. Graph Compilation & Run Config Tests
# =====================================================================

class TestGraphCheckpointerIntegration:
    def test_build_graph_with_custom_checkpointer(self):
        """Passing custom checkpointer to build_graph is honored."""
        custom_saver = MemorySaver()
        graph = build_graph(checkpointer=custom_saver)
        assert hasattr(graph, "checkpointer")
        assert graph.checkpointer is custom_saver

    def test_build_graph_stateless_mode(self):
        """checkpointer=False disables checkpointer (stateless)."""
        graph = build_graph(checkpointer=False)
        assert getattr(graph, "checkpointer", None) is None

    def test_build_graph_default_checkpointer(self):
        """build_graph() with None/default checkpointer compiles with MemorySaver."""
        graph = build_graph()
        assert hasattr(graph, "checkpointer")
        assert isinstance(graph.checkpointer, MemorySaver)

    def test_get_run_config_populates_thread_id(self):
        """get_run_config correctly populates configurable.thread_id."""
        config = get_run_config(project_id="proj-12345", node_name="pipeline")
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "proj-12345"
        assert "metadata" in config
        assert config["metadata"]["project_id"] == "proj-12345"
        assert "tags" in config


# =====================================================================
# 3. StateGraph Checkpointer Persistence & Resumption Tests
# =====================================================================

class TestStateGraphCheckpointing:
    @pytest.fixture
    def sample_graph(self):
        """Builds a test 2-step pipeline with checkpointer support."""
        def step_one(state: ProjectState) -> dict[str, Any]:
            it = state.get("iteration", 0) + 1
            return {
                "iteration": it,
                "current_phase": "phase_one",
                "execution_trace": [{"step": 1, "action": "initialize"}],
            }

        def step_two(state: ProjectState) -> dict[str, Any]:
            it = state.get("iteration", 0) + 1
            return {
                "iteration": it,
                "current_phase": "completed",
                "status": "completed",
                "execution_trace": [{"step": 2, "action": "finalize"}],
            }

        def create_compiled_graph(checkpointer: BaseCheckpointSaver):
            builder = StateGraph(ProjectState)
            builder.add_node("step_one", step_one)
            builder.add_node("step_two", step_two)
            builder.set_entry_point("step_one")
            builder.add_edge("step_one", "step_two")
            builder.add_edge("step_two", END)
            return builder.compile(checkpointer=checkpointer)

        return create_compiled_graph

    def test_single_thread_persistence_and_snapshot(self, sample_graph):
        """State is persisted and queryable via get_state with thread_id."""
        saver = MemorySaver()
        graph = sample_graph(saver)
        config = {"configurable": {"thread_id": "thread-001"}}

        initial_state = {
            "project_id": "thread-001",
            "project_name": "Test Project",
            "iteration": 0,
            "status": "running",
        }

        final_state = graph.invoke(initial_state, config=config)
        assert final_state["status"] == "completed"
        assert final_state["iteration"] == 2
        assert len(final_state["execution_trace"]) == 2

        # Verify state snapshot retrieval
        snapshot = graph.get_state(config)
        assert snapshot is not None
        assert snapshot.values["project_id"] == "thread-001"
        assert snapshot.values["status"] == "completed"
        assert snapshot.values["iteration"] == 2

    def test_thread_isolation(self, sample_graph):
        """Different thread_ids must maintain isolated state snapshots."""
        saver = MemorySaver()
        graph = sample_graph(saver)

        config_a = {"configurable": {"thread_id": "thread-A"}}
        config_b = {"configurable": {"thread_id": "thread-B"}}

        graph.invoke({"project_id": "thread-A", "requirements": "Req A", "iteration": 0}, config=config_a)
        graph.invoke({"project_id": "thread-B", "requirements": "Req B", "iteration": 10}, config=config_b)

        snap_a = graph.get_state(config_a)
        snap_b = graph.get_state(config_b)

        assert snap_a.values["requirements"] == "Req A"
        assert snap_a.values["iteration"] == 2  # 0 + 1 + 1

        assert snap_b.values["requirements"] == "Req B"
        assert snap_b.values["iteration"] == 12  # 10 + 1 + 1

    def test_crash_recovery_across_graph_instances(self, sample_graph):
        """A new graph instance sharing checkpointer recovers saved state."""
        shared_saver = MemorySaver()
        graph_instance_1 = sample_graph(shared_saver)
        config = {"configurable": {"thread_id": "resume-thread-99"}}

        # Run instance 1
        graph_instance_1.invoke({"project_id": "resume-thread-99", "iteration": 0}, config=config)

        # Simulate fresh process / restart with instance 2 sharing checkpointer
        graph_instance_2 = sample_graph(shared_saver)
        recovered_snapshot = graph_instance_2.get_state(config)

        assert recovered_snapshot is not None
        assert recovered_snapshot.values["project_id"] == "resume-thread-99"
        assert recovered_snapshot.values["status"] == "completed"
        assert recovered_snapshot.values["iteration"] == 2

    def test_get_state_history(self, sample_graph):
        """get_state_history returns all checkpoint snapshots across steps."""
        saver = MemorySaver()
        graph = sample_graph(saver)
        config = {"configurable": {"thread_id": "history-thread-1"}}

        graph.invoke({"project_id": "history-thread-1", "iteration": 0}, config=config)

        history = list(graph.get_state_history(config))
        assert len(history) >= 3  # Initial + Step 1 + Step 2
        phases = [snap.values.get("current_phase") for snap in history if "current_phase" in snap.values]
        assert "phase_one" in phases
        assert "completed" in phases

    def test_update_state_injection(self, sample_graph):
        """update_state injects state modifications into saved checkpoint."""
        saver = MemorySaver()
        graph = sample_graph(saver)
        config = {"configurable": {"thread_id": "hitl-thread-1"}}

        graph.invoke({"project_id": "hitl-thread-1", "iteration": 0}, config=config)

        # Inject manual state update (e.g. human intervention)
        graph.update_state(config, {"status": "paused", "requirements": "Updated requirement"})

        updated_snap = graph.get_state(config)
        assert updated_snap.values["status"] == "paused"
        assert updated_snap.values["requirements"] == "Updated requirement"
