"""
Test new GraphState helper methods.

Verify get_node_state() and update_node_state() work correctly.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import jax.numpy as jnp
from engine.environments.farmers_market import create_simple_farmers_market


def test_get_node_state():
    """Test getting all attributes for a specific node."""
    print("Testing get_node_state()...")

    state = create_simple_farmers_market(num_farmers=5, seed=42)

    # Get state for node 0
    node_state = state.get_node_state(0)

    print(f"  Node 0 state: {node_state}")

    # Verify structure
    assert "type" in node_state
    assert "attrs" in node_state
    assert "edges" in node_state

    # Verify attributes match
    assert node_state["attrs"]["resources_apples"] == float(state.node_attrs["resources_apples"][0])
    assert node_state["attrs"]["resources_wheat"] == float(state.node_attrs["resources_wheat"][0])

    print("  [OK] get_node_state() works correctly")


def test_update_node_state():
    """Test updating multiple attributes at once."""
    print("\nTesting update_node_state()...")

    state = create_simple_farmers_market(num_farmers=5, seed=42)

    # Get initial values
    initial_apples = float(state.node_attrs["resources_apples"][0])
    initial_wheat = float(state.node_attrs["resources_wheat"][0])

    # Update multiple attributes
    new_state = state.update_node_state(0, {
        "resources_apples": 999.0,
        "resources_wheat": 888.0
    })

    # Verify updates
    assert float(new_state.node_attrs["resources_apples"][0]) == 999.0
    assert float(new_state.node_attrs["resources_wheat"][0]) == 888.0

    # Verify other nodes unchanged
    assert float(new_state.node_attrs["resources_apples"][1]) == float(state.node_attrs["resources_apples"][1])

    # Verify original state unchanged (immutability)
    assert float(state.node_attrs["resources_apples"][0]) == initial_apples
    assert float(state.node_attrs["resources_wheat"][0]) == initial_wheat

    print("  [OK] update_node_state() works correctly")
    print("  [OK] Immutability preserved")


def test_round_trip():
    """Test get and update together."""
    print("\nTesting get/update round-trip...")

    state = create_simple_farmers_market(num_farmers=5, seed=42)

    # Get node state
    node_state = state.get_node_state(0)
    original_apples = node_state["attrs"]["resources_apples"]

    # Modify and update
    new_apples = original_apples * 2
    new_state = state.update_node_state(0, {"resources_apples": new_apples})

    # Verify change
    updated_node_state = new_state.get_node_state(0)
    assert updated_node_state["attrs"]["resources_apples"] == new_apples

    print(f"  Original: {original_apples:.2f}")
    print(f"  Updated:  {new_apples:.2f}")
    print("  [OK] Round-trip works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("GRAPHSTATE HELPER METHODS TESTS")
    print("=" * 60)

    test_get_node_state()
    test_update_node_state()
    test_round_trip()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
