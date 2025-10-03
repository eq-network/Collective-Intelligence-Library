# Example for tests/transformations/top_down/democratic_transforms/test_delegation.py
import unittest
import jax.numpy as jnp
# ... (add imports similar to test_power_flow.py) ...
from transformations.top_down.democratic_transforms.delegation import create_delegation_transform
import sys
from pathlib import Path

# Add project root to sys.path to allow direct import of project modules
# This assumes your 'tests' directory is at the root of your 'Mycorrhiza' project
# Adjust the number of .parent calls if your structure is different
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.graph import GraphState
class TestDelegationTransform(unittest.TestCase):
    def test_delegation_choices_to_matrix(self):
        num_agents = 4
        # Agent 0 votes directly (target -1)
        # Agent 1 delegates to Agent 0
        # Agent 2 delegates to Agent 0
        # Agent 3 delegates to Agent 2
        delegation_targets = jnp.array([-1, 0, 0, 2], dtype=jnp.int32)

        initial_state = GraphState(
            node_attrs={"delegation_target": delegation_targets}, # Assuming this is the attribute name used
            adj_matrices={},
            global_attrs={}
        )
        # In your delegation.py, it reads "delegation_choices".
        # Ensure the attribute name is consistent with what mechanism_factory.py produces.
        # Let's assume for this test, we create "delegation_choices" as per your delegation.py
        # initial_state = GraphState(
        #     node_attrs={"delegation_choices": delegation_targets}, 
        #     adj_matrices={},
        #     global_attrs={}
        # )


        transform = create_delegation_transform()
        final_state = transform(initial_state)

        self.assertIn("delegation", final_state.adj_matrices)
        
        expected_matrix = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)
        expected_matrix = expected_matrix.at[1, 0].set(1.0)
        expected_matrix = expected_matrix.at[2, 0].set(1.0)
        expected_matrix = expected_matrix.at[3, 2].set(1.0)

        self.assertTrue(jnp.array_equal(final_state.adj_matrices["delegation"], expected_matrix),
                        f"Expected {expected_matrix}, got {final_state.adj_matrices['delegation']}")

    def test_delegation_self_delegation_invalid(self):
        num_agents = 2
        # Agent 0 tries to delegate to self (invalid)
        # Agent 1 delegates to Agent 0
        delegation_targets = jnp.array([0, 0], dtype=jnp.int32) 
        initial_state = GraphState(
            node_attrs={"delegation_target": delegation_targets},
            adj_matrices={},
            global_attrs={}
        )
        transform = create_delegation_transform()
        final_state = transform(initial_state)
        
        expected_matrix = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)
        expected_matrix = expected_matrix.at[1, 0].set(1.0) # Only A1 -> A0 is valid

        self.assertTrue(jnp.array_equal(final_state.adj_matrices["delegation"], expected_matrix))

if __name__ == '__main__':
    unittest.main()