"""
Tests for core.graph.GraphState

Tests GraphState immutability, capacity mode, and core operations.
"""
import pytest
import jax.numpy as jnp

from core.graph import GraphState, create_padded_state, CapacityExceededError


class TestGraphStateBasics:
    """Test basic GraphState creation and properties."""

    def test_create_simple_state(self):
        """Can create a simple GraphState."""
        state = GraphState(
            node_types=jnp.array([0, 0, 0]),
            node_attrs={"resources": jnp.array([100.0, 100.0, 100.0])},
            adj_matrices={"connections": jnp.zeros((3, 3))},
            global_attrs={"round": 0}
        )
        assert state.num_nodes == 3
        assert "resources" in state.node_attrs
        assert state.global_attrs["round"] == 0

    def test_immutability(self):
        """GraphState.replace creates new instance, doesn't mutate."""
        state1 = GraphState(
            node_types=jnp.array([0, 0]),
            node_attrs={"resources": jnp.array([100.0, 100.0])},
            adj_matrices={},
            global_attrs={"round": 0}
        )

        # Create new state with updated global_attrs
        state2 = state1.replace(global_attrs={"round": 1})

        # Original unchanged
        assert state1.global_attrs["round"] == 0
        # New state has update
        assert state2.global_attrs["round"] == 1
        # They are different objects
        assert state1 is not state2

    def test_update_node_attrs(self):
        """update_node_attrs creates new state with updated values."""
        state = GraphState(
            node_types=jnp.array([0, 0]),
            node_attrs={"resources": jnp.array([100.0, 100.0])},
            adj_matrices={},
            global_attrs={}
        )

        new_resources = jnp.array([50.0, 150.0])
        new_state = state.update_node_attrs("resources", new_resources)

        # Original unchanged
        assert float(state.node_attrs["resources"][0]) == 100.0
        # New state has update
        assert float(new_state.node_attrs["resources"][0]) == 50.0


class TestCapacityMode:
    """Test capacity mode with padded matrices."""

    def test_create_padded_state(self):
        """create_padded_state creates properly sized arrays."""
        state = create_padded_state(
            capacity=10,
            initial_active=3,
            node_attrs_init={"resources": jnp.array([100.0, 100.0, 100.0])},
            adj_matrices_init={"connections": jnp.zeros((3, 3))},
            global_attrs={"round": 0}
        )

        # Arrays sized to capacity
        assert state.node_types.shape == (10,)
        assert state.node_attrs["resources"].shape == (10,)
        assert state.adj_matrices["connections"].shape == (10, 10)

        # But only 3 active nodes
        assert state.num_nodes == 3
        assert len(state.get_active_indices()) == 3

    def test_inactive_nodes_marked(self):
        """Inactive slots have node_type=-1."""
        state = create_padded_state(
            capacity=5,
            initial_active=2,
            node_attrs_init={"x": jnp.array([1.0, 1.0])}
        )

        # First 2 are active (type 0)
        assert int(state.node_types[0]) == 0
        assert int(state.node_types[1]) == 0

        # Rest are inactive (type -1)
        assert int(state.node_types[2]) == -1
        assert int(state.node_types[3]) == -1
        assert int(state.node_types[4]) == -1

    def test_get_active_indices(self):
        """get_active_indices returns only active node IDs."""
        state = create_padded_state(
            capacity=10,
            initial_active=3,
            node_attrs_init={"x": jnp.array([1.0, 1.0, 1.0])}
        )

        active = state.get_active_indices()
        assert len(active) == 3
        assert list(active) == [0, 1, 2]

    def test_get_active_mask(self):
        """get_active_mask returns boolean mask for active nodes."""
        state = create_padded_state(
            capacity=5,
            initial_active=2,
            node_attrs_init={"x": jnp.array([1.0, 1.0])}
        )

        mask = state.get_active_mask()
        assert mask.shape == (5,)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is True
        assert bool(mask[2]) is False

    def test_is_capacity_mode(self):
        """is_capacity_mode property correctly identifies mode."""
        # Capacity mode
        padded_state = create_padded_state(
            capacity=5,
            initial_active=2,
            node_attrs_init={"x": jnp.array([1.0, 1.0])}
        )
        assert padded_state.is_capacity_mode is True

        # Dynamic mode
        dynamic_state = GraphState(
            node_types=jnp.array([0, 0]),
            node_attrs={"x": jnp.array([1.0, 1.0])},
            adj_matrices={},
            capacity=None
        )
        assert dynamic_state.is_capacity_mode is False


class TestNodeOperations:
    """Test node-level operations."""

    def test_get_node_state(self):
        """get_node_state returns dict with type, attrs, and edges."""
        state = GraphState(
            node_types=jnp.array([0, 1]),
            node_attrs={
                "resources": jnp.array([100.0, 200.0]),
                "score": jnp.array([0.5, 0.8])
            },
            adj_matrices={},
            global_attrs={}
        )

        node_state = state.get_node_state(1)

        # Check structure: type, attrs, edges
        assert "type" in node_state
        assert "attrs" in node_state
        assert "edges" in node_state

        # Check values
        assert node_state["type"] == 1
        assert "resources" in node_state["attrs"]
        assert "score" in node_state["attrs"]
        assert float(node_state["attrs"]["resources"]) == 200.0
        assert abs(float(node_state["attrs"]["score"]) - 0.8) < 1e-5


class TestSumConservation:
    """Test that operations can preserve resource totals."""

    def test_total_resources_preserved_with_mask(self):
        """Active mask correctly excludes inactive nodes from sums."""
        state = create_padded_state(
            capacity=10,
            initial_active=3,
            node_attrs_init={"resources": jnp.array([100.0, 100.0, 100.0])}
        )

        # Total across active nodes only
        mask = state.get_active_mask()
        total = float(jnp.sum(state.node_attrs["resources"] * mask))

        assert total == 300.0  # 3 nodes x 100 each
