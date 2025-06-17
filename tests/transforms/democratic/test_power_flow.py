# tests/transformations/top_down/democratic_transforms/test_power_flow.py
import unittest
import jax.numpy as jnp
import sys
from pathlib import Path

# Add project root to sys.path to allow direct import of project modules
# This assumes your 'tests' directory is at the root of your 'Mycorrhiza' project
# Adjust the number of .parent calls if your structure is different
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from algebra.graph import GraphState
from transformations.top_down.democratic_transforms.power_flow import create_power_flow_transform

class TestPowerFlowTransform(unittest.TestCase):

    def test_no_delegation(self):
        """Test when no agents delegate, everyone should have base power."""
        num_agents = 5
        # No delegation: all zeros matrix
        delegation_matrix = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)

        initial_state = GraphState(
            node_attrs={}, # Power flow should initialize voting_power
            adj_matrices={"delegation": delegation_matrix},
            global_attrs={}
        )

        # Create the transform. The default base_power in your create_power_flow_transform is 1.0
        # If you want to test with a different base_power, pass it in config:
        # power_flow_config = {"base_voting_power": 1.0}
        # transform = create_power_flow_transform(config=power_flow_config)
        transform = create_power_flow_transform() # Uses default base_power of 1.0

        final_state = transform(initial_state)

        self.assertIn("voting_power", final_state.node_attrs)
        expected_power = jnp.ones(num_agents, dtype=jnp.float32) # Everyone has 1.0 power
        
        # Using jnp.array_equal for JAX arrays
        self.assertTrue(jnp.array_equal(final_state.node_attrs["voting_power"], expected_power),
                        f"Expected {expected_power}, got {final_state.node_attrs['voting_power']}")

    def test_single_level_delegation_one_recipient(self):
        """Test simple delegation: Agents 1 and 2 delegate to Agent 0."""
        num_agents = 5
        # Agent 1 -> Agent 0
        # Agent 2 -> Agent 0
        delegation_matrix = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)
        delegation_matrix = delegation_matrix.at[1, 0].set(1.0)
        delegation_matrix = delegation_matrix.at[2, 0].set(1.0)

        initial_state = GraphState(
            node_attrs={},
            adj_matrices={"delegation": delegation_matrix},
            global_attrs={}
        )
        transform = create_power_flow_transform()
        final_state = transform(initial_state)

        self.assertIn("voting_power", final_state.node_attrs)
        # Expected:
        # Agent 0: 1 (own) + 1 (from A1) + 1 (from A2) = 3.0
        # Agent 1: 0 (delegated)
        # Agent 2: 0 (delegated)
        # Agent 3: 1 (own)
        # Agent 4: 1 (own)
        expected_power = jnp.array([3.0, 0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        
        self.assertTrue(jnp.array_equal(final_state.node_attrs["voting_power"], expected_power),
                        f"Expected {expected_power}, got {final_state.node_attrs['voting_power']}")
    def test_multi_level_delegation_chain(self):
            """Test transitive delegation: Agent 2 -> Agent 1 -> Agent 0."""
            num_agents = 3
            # Agent 2 -> Agent 1
            # Agent 1 -> Agent 0
            delegation_matrix = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)
            delegation_matrix = delegation_matrix.at[2, 1].set(1.0) # A2 delegates to A1
            delegation_matrix = delegation_matrix.at[1, 0].set(1.0) # A1 delegates to A0

            initial_state = GraphState(
                node_attrs={},
                adj_matrices={"delegation": delegation_matrix},
                global_attrs={}
            )
            transform = create_power_flow_transform()
            final_state = transform(initial_state)

            self.assertIn("voting_power", final_state.node_attrs)
            # Expected with correct transitive flow:
            # Agent 0: 1 (own) + 1 (from A1 directly) + 1 (from A2 via A1) = 3.0
            # Agent 1: 0 (delegated all its received and own power to A0)
            # Agent 2: 0 (delegated its power to A1)
            expected_power = jnp.array([3.0, 0.0, 0.0], dtype=jnp.float32)
            
            self.assertTrue(jnp.array_equal(final_state.node_attrs["voting_power"], expected_power),
                            f"Expected {expected_power}, got {final_state.node_attrs['voting_power']}")

# To run this test file from your project root (e.g., Mycorrhiza/):
# python -m unittest tests.transformations.top_down.democratic_transforms.test_power_flow
if __name__ == '__main__':
    unittest.main()