# tests/transformations/top_down/democratic_transforms/test_voting.py
import unittest
import jax.numpy as jnp
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.graph import GraphState
# The _portfolio_vote_aggregator is likely defined within mechanism_factory.py
# or passed into create_voting_transform.
# For this test, we'll assume you can import/access it.
# If it's defined inside create_portfolio_mechanism_pipeline,
# you might need to extract it or test create_voting_transform directly.

# Let's assume _portfolio_vote_aggregator is accessible or you test create_voting_transform
# For simplicity, let's test create_voting_transform with the aggregator.
from transformations.top_down.democratic_transforms.voting import create_voting_transform
from environments.random.mechanism_factory import _portfolio_vote_aggregator # Assuming it's importable

class TestVotingTransform(unittest.TestCase):

    def test_pld_vote_aggregation_with_voting_power(self):
        """Test that PLD correctly aggregates votes weighted by voting_power."""
        num_agents = 3
        num_portfolios = 2

        # Agent 0: high power, votes for portfolio 0
        # Agent 1: low power, votes for portfolio 1
        # Agent 2: medium power, votes for portfolio 0
        agent_portfolio_votes = jnp.array([
            [1, 0],  # Agent 0 votes for P0
            [0, 1],  # Agent 1 votes for P1
            [1, 0]   # Agent 2 votes for P0
        ], dtype=jnp.int32)

        voting_power = jnp.array([3.0, 0.5, 1.5], dtype=jnp.float32) # Example voting powers

        initial_state = GraphState(
            node_attrs={
                "agent_portfolio_votes": agent_portfolio_votes,
                "voting_power": voting_power
            },
            adj_matrices={},
            global_attrs={}
        )

        # Configuration for the voting transform for PLD
        voting_config_pld = {"mechanism_type": "liquid", "output_attr_name": "current_decision"}
        
        # Create the voting transform using your _portfolio_vote_aggregator
        pld_voting_transform = create_voting_transform(
            vote_aggregator=_portfolio_vote_aggregator,
            config=voting_config_pld
        )

        final_state = pld_voting_transform(initial_state)

        self.assertIn("current_decision", final_state.global_attrs)
        self.assertIn("vote_distribution", final_state.global_attrs)

        # Expected weighted votes:
        # Portfolio 0: (1 * 3.0) + (0 * 0.5) + (1 * 1.5) = 3.0 + 1.5 = 4.5
        # Portfolio 1: (0 * 3.0) + (1 * 0.5) + (0 * 1.5) = 0.5
        expected_vote_distribution = jnp.array([4.5, 0.5], dtype=jnp.float32)
        
        # The decision should be portfolio 0 (index 0) as it has more weighted votes
        expected_decision = 0 

        self.assertTrue(jnp.allclose(final_state.global_attrs["vote_distribution"], expected_vote_distribution),
                        f"Expected distribution {expected_vote_distribution}, got {final_state.global_attrs['vote_distribution']}")
        self.assertEqual(final_state.global_attrs["current_decision"], expected_decision,
                         f"Expected decision {expected_decision}, got {final_state.global_attrs['current_decision']}")

    def test_pdd_vote_aggregation_no_weighting(self):
        """Test that PDD (direct) correctly aggregates votes without special weighting."""
        num_agents = 3
        num_portfolios = 2

        agent_portfolio_votes = jnp.array([
            [1, 0],  # Agent 0 votes for P0
            [0, 1],  # Agent 1 votes for P1
            [1, 0]   # Agent 2 votes for P0
        ], dtype=jnp.int32)

        # For PDD, voting_power attribute might not be used by the aggregator, or it's all 1s
        # The _portfolio_vote_aggregator's "direct" path just sums votes.
        initial_state = GraphState(
            node_attrs={"agent_portfolio_votes": agent_portfolio_votes},
            adj_matrices={},
            global_attrs={}
        )

        voting_config_pdd = {"mechanism_type": "direct", "output_attr_name": "current_decision"}
        pdd_voting_transform = create_voting_transform(
            vote_aggregator=_portfolio_vote_aggregator,
            config=voting_config_pdd
        )
        final_state = pdd_voting_transform(initial_state)

        # Expected votes (simple sum):
        # Portfolio 0: 1 + 0 + 1 = 2
        # Portfolio 1: 0 + 1 + 0 = 1
        expected_vote_distribution = jnp.array([2, 1], dtype=jnp.int32) # PDD sums integers
        expected_decision = 0

        self.assertTrue(jnp.array_equal(final_state.global_attrs["vote_distribution"], expected_vote_distribution),
                        f"Expected distribution {expected_vote_distribution}, got {final_state.global_attrs['vote_distribution']}")
        self.assertEqual(final_state.global_attrs["current_decision"], expected_decision)


# To run: python -m unittest tests.transformations.top_down.democratic_transforms.test_voting
if __name__ == '__main__':
    unittest.main()