"""
Tests for core.category - Transform composition operators.

Tests compose, sequential, identity, and property attachment.
"""
import pytest
import jax.numpy as jnp

from core.graph import GraphState
from core.category import compose, sequential, identity, attach_properties


def create_test_state() -> GraphState:
    """Create a simple test state."""
    return GraphState(
        node_types=jnp.array([0, 0, 0]),
        node_attrs={"value": jnp.array([1.0, 2.0, 3.0])},
        adj_matrices={},
        global_attrs={"step": 0}
    )


class TestCompose:
    """Test compose function for chaining transforms."""

    def test_compose_order(self):
        """compose(f, g) applies f first, then g."""
        def add_one(state: GraphState) -> GraphState:
            new_vals = state.node_attrs["value"] + 1
            return state.replace(node_attrs={"value": new_vals})

        def double(state: GraphState) -> GraphState:
            new_vals = state.node_attrs["value"] * 2
            return state.replace(node_attrs={"value": new_vals})

        state = create_test_state()  # [1, 2, 3]

        # compose(add_one, double): first add 1, then double
        composed = compose(add_one, double)
        result = composed(state)

        # [1, 2, 3] + 1 = [2, 3, 4], then * 2 = [4, 6, 8]
        expected = jnp.array([4.0, 6.0, 8.0])
        assert jnp.allclose(result.node_attrs["value"], expected)

    def test_compose_preserves_intersected_properties(self):
        """Composed transform preserves intersection of properties."""
        def f(state: GraphState) -> GraphState:
            return state

        def g(state: GraphState) -> GraphState:
            return state

        f.preserves = {"prop_a", "prop_b"}
        g.preserves = {"prop_b", "prop_c"}

        composed = compose(f, g)

        # Only prop_b is in both
        assert composed.preserves == {"prop_b"}


class TestSequential:
    """Test sequential composition of multiple transforms."""

    def test_sequential_empty_returns_identity(self):
        """sequential() with no args returns identity."""
        result = sequential()
        state = create_test_state()

        # Identity doesn't change state
        new_state = result(state)
        assert jnp.allclose(new_state.node_attrs["value"], state.node_attrs["value"])

    def test_sequential_single_returns_same(self):
        """sequential(f) returns f."""
        def f(state: GraphState) -> GraphState:
            return state.replace(global_attrs={"step": 1})

        result = sequential(f)
        state = create_test_state()
        new_state = result(state)

        assert new_state.global_attrs["step"] == 1

    def test_sequential_chains_multiple(self):
        """sequential(f, g, h) chains all in order."""
        def step1(state: GraphState) -> GraphState:
            new_attrs = dict(state.global_attrs)
            new_attrs["step1"] = True
            return state.replace(global_attrs=new_attrs)

        def step2(state: GraphState) -> GraphState:
            new_attrs = dict(state.global_attrs)
            new_attrs["step2"] = True
            return state.replace(global_attrs=new_attrs)

        def step3(state: GraphState) -> GraphState:
            new_attrs = dict(state.global_attrs)
            new_attrs["step3"] = True
            return state.replace(global_attrs=new_attrs)

        pipeline = sequential(step1, step2, step3)
        state = create_test_state()
        result = pipeline(state)

        # All steps executed
        assert result.global_attrs.get("step1") is True
        assert result.global_attrs.get("step2") is True
        assert result.global_attrs.get("step3") is True

    def test_sequential_order_matters(self):
        """sequential applies transforms in order."""
        def set_a(state: GraphState) -> GraphState:
            new_attrs = dict(state.global_attrs)
            new_attrs["last"] = "a"
            return state.replace(global_attrs=new_attrs)

        def set_b(state: GraphState) -> GraphState:
            new_attrs = dict(state.global_attrs)
            new_attrs["last"] = "b"
            return state.replace(global_attrs=new_attrs)

        # a then b -> last = "b"
        result1 = sequential(set_a, set_b)(create_test_state())
        assert result1.global_attrs["last"] == "b"

        # b then a -> last = "a"
        result2 = sequential(set_b, set_a)(create_test_state())
        assert result2.global_attrs["last"] == "a"


class TestIdentity:
    """Test identity transform."""

    def test_identity_returns_unchanged_state(self):
        """Identity transform returns state unchanged."""
        state = create_test_state()
        id_fn = identity()
        result = id_fn(state)

        # Same values
        assert jnp.allclose(result.node_attrs["value"], state.node_attrs["value"])
        assert result.global_attrs == state.global_attrs

    def test_identity_preserves_all_properties(self):
        """Identity has special 'ALL_PROPERTIES' marker."""
        id_fn = identity()
        assert id_fn.preserves == "ALL_PROPERTIES"


class TestAttachProperties:
    """Test property attachment to transforms."""

    def test_attach_properties_sets_preserves_attr(self):
        """attach_properties sets the preserves attribute."""
        def my_transform(state: GraphState) -> GraphState:
            return state

        props = {"conservation", "monotonicity"}
        result = attach_properties(my_transform, props)

        assert result.preserves == props
        # Same function object
        assert result is my_transform

    def test_attach_empty_properties(self):
        """Can attach empty property set."""
        def my_transform(state: GraphState) -> GraphState:
            return state

        result = attach_properties(my_transform, set())
        assert result.preserves == set()


class TestTransformPurity:
    """Test that transforms follow pure function semantics."""

    def test_transform_does_not_mutate_input(self):
        """Transform should not modify input state."""
        state = create_test_state()
        original_value = float(state.node_attrs["value"][0])

        def double_transform(s: GraphState) -> GraphState:
            new_vals = s.node_attrs["value"] * 2
            return s.replace(node_attrs={"value": new_vals})

        _ = double_transform(state)

        # Original state unchanged
        assert float(state.node_attrs["value"][0]) == original_value
