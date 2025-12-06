"""
Tests for core.property - Property system for invariant checking.

Tests ConservesSum binding, checking, and property composition.
"""
import pytest
import jax.numpy as jnp

from core.graph import GraphState
from core.property import Property, ConservesSum, ConjunctiveProperty, DisjunctiveProperty, NegatedProperty


def create_test_state() -> GraphState:
    """Create a simple test state with resources."""
    return GraphState(
        node_types=jnp.array([0, 0, 0]),
        node_attrs={"resources": jnp.array([10.0, 20.0, 30.0])},
        adj_matrices={},
        global_attrs={}
    )


class TestConservesSum:
    """Test ConservesSum property for conservation checking."""

    def test_unbound_property_raises_on_check(self):
        """Unbound ConservesSum raises ValueError when checking."""
        prop = ConservesSum("resources")
        state = create_test_state()

        with pytest.raises(ValueError, match="not bound"):
            prop.check(state)

    def test_bind_creates_new_property(self):
        """bind() returns a new property with reference sum."""
        prop = ConservesSum("resources")
        state = create_test_state()

        bound = prop.bind(state)

        # Original is unchanged
        assert prop.reference_sum is None
        # New property has the sum
        assert bound.reference_sum == 60.0  # 10 + 20 + 30

    def test_bind_missing_attribute_raises(self):
        """bind() raises ValueError for missing attribute."""
        prop = ConservesSum("nonexistent")
        state = create_test_state()

        with pytest.raises(ValueError, match="not found"):
            prop.bind(state)

    def test_check_conserved_sum(self):
        """check() returns True when sum is conserved."""
        state = create_test_state()
        prop = ConservesSum("resources").bind(state)

        # Same state should pass
        assert prop.check(state) is True

        # Redistributed (same total) should pass
        redistributed = state.replace(
            node_attrs={"resources": jnp.array([20.0, 20.0, 20.0])}
        )
        assert prop.check(redistributed) is True

    def test_check_violated_sum(self):
        """check() returns False when sum changes."""
        state = create_test_state()
        prop = ConservesSum("resources").bind(state)

        # Increased total should fail
        increased = state.replace(
            node_attrs={"resources": jnp.array([20.0, 30.0, 40.0])}
        )
        assert prop.check(increased) is False

        # Decreased total should fail
        decreased = state.replace(
            node_attrs={"resources": jnp.array([5.0, 10.0, 15.0])}
        )
        assert prop.check(decreased) is False

    def test_check_missing_attribute_returns_false(self):
        """check() returns False if attribute is missing in state."""
        state = create_test_state()
        prop = ConservesSum("resources").bind(state)

        # State without the attribute
        empty_state = state.replace(node_attrs={})
        assert prop.check(empty_state) is False

    def test_tolerance_allows_small_differences(self):
        """Tolerance parameter allows floating point imprecision."""
        state = create_test_state()
        prop = ConservesSum("resources", tolerance=0.01).bind(state)

        # Tiny difference within tolerance
        slightly_off = state.replace(
            node_attrs={"resources": jnp.array([10.001, 20.0, 29.999])}
        )
        assert prop.check(slightly_off) is True


class TestConservesSumCapacityMode:
    """Test ConservesSum with capacity mode (active nodes only)."""

    def test_bind_respects_capacity_mode(self):
        """bind() only sums active nodes in capacity mode."""
        state = GraphState(
            node_types=jnp.array([0, 0, 0, -1, -1]),  # 3 active, 2 inactive
            node_attrs={"resources": jnp.array([10.0, 20.0, 30.0, 100.0, 200.0])},
            adj_matrices={},
            global_attrs={},
            capacity=5
        )

        prop = ConservesSum("resources").bind(state)

        # Should only sum active nodes: 10 + 20 + 30 = 60
        assert prop.reference_sum == 60.0

    def test_check_respects_capacity_mode(self):
        """check() only sums active nodes in capacity mode."""
        state = GraphState(
            node_types=jnp.array([0, 0, 0, -1, -1]),
            node_attrs={"resources": jnp.array([10.0, 20.0, 30.0, 100.0, 200.0])},
            adj_matrices={},
            global_attrs={},
            capacity=5
        )

        prop = ConservesSum("resources").bind(state)

        # Changing inactive nodes shouldn't matter
        changed_inactive = state.replace(
            node_attrs={"resources": jnp.array([10.0, 20.0, 30.0, 0.0, 0.0])}
        )
        assert prop.check(changed_inactive) is True


class TestPropertyComposition:
    """Test property composition operators (&, |, ~)."""

    def test_conjunction_requires_both(self):
        """Conjunctive property requires both properties to hold."""
        state = GraphState(
            node_types=jnp.array([0, 0]),
            node_attrs={
                "a": jnp.array([10.0, 10.0]),
                "b": jnp.array([5.0, 5.0])
            },
            adj_matrices={},
            global_attrs={}
        )

        prop_a = ConservesSum("a").bind(state)
        prop_b = ConservesSum("b").bind(state)
        combined = prop_a & prop_b

        # Both conserved - should pass
        assert combined.check(state) is True

        # Only a conserved - should fail
        only_a = state.replace(node_attrs={
            "a": jnp.array([10.0, 10.0]),
            "b": jnp.array([20.0, 20.0])  # Changed
        })
        assert combined.check(only_a) is False

    def test_disjunction_requires_either(self):
        """Disjunctive property requires at least one property to hold."""
        state = GraphState(
            node_types=jnp.array([0, 0]),
            node_attrs={
                "a": jnp.array([10.0, 10.0]),
                "b": jnp.array([5.0, 5.0])
            },
            adj_matrices={},
            global_attrs={}
        )

        prop_a = ConservesSum("a").bind(state)
        prop_b = ConservesSum("b").bind(state)
        combined = prop_a | prop_b

        # Only a conserved - should pass
        only_a = state.replace(node_attrs={
            "a": jnp.array([10.0, 10.0]),
            "b": jnp.array([20.0, 20.0])
        })
        assert combined.check(only_a) is True

        # Only b conserved - should pass
        only_b = state.replace(node_attrs={
            "a": jnp.array([100.0, 100.0]),
            "b": jnp.array([5.0, 5.0])
        })
        assert combined.check(only_b) is True

        # Neither conserved - should fail
        neither = state.replace(node_attrs={
            "a": jnp.array([100.0, 100.0]),
            "b": jnp.array([50.0, 50.0])
        })
        assert combined.check(neither) is False

    def test_negation_inverts(self):
        """Negated property inverts the check result."""
        state = create_test_state()
        prop = ConservesSum("resources").bind(state)
        negated = ~prop

        # Original conserved, so negation should fail
        assert negated.check(state) is False

        # Original violated, so negation should pass
        changed = state.replace(
            node_attrs={"resources": jnp.array([100.0, 100.0, 100.0])}
        )
        assert negated.check(changed) is True


class TestPropertyNaming:
    """Test property naming and description."""

    def test_conserves_sum_name(self):
        """ConservesSum has descriptive name."""
        prop = ConservesSum("voting_power")
        assert "voting_power" in prop.name

    def test_conjunction_name_includes_both(self):
        """Conjunction name includes both property names."""
        a = ConservesSum("a")
        b = ConservesSum("b")
        combined = a & b

        assert "a" in combined.name
        assert "b" in combined.name
        assert "AND" in combined.name

    def test_disjunction_name_includes_both(self):
        """Disjunction name includes both property names."""
        a = ConservesSum("a")
        b = ConservesSum("b")
        combined = a | b

        assert "a" in combined.name
        assert "b" in combined.name
        assert "OR" in combined.name

    def test_negation_name_includes_not(self):
        """Negation name includes NOT."""
        prop = ConservesSum("resources")
        negated = ~prop

        assert "NOT" in negated.name
