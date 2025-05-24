import unittest
import jax
import jax.numpy as jnp
import jax.random as jr
from dataclasses import replace as dataclass_replace # Import the replace function

from environments.democracy.configuration import create_thesis_baseline_config
from environments.democracy.initialization import initialize_portfolio_democracy_graph_state
from transformations.top_down.democratic_transforms.election import create_election_transform

class TestPRDMechanism(unittest.TestCase):

    def setUp(self):
        # Start with the baseline config
        base_config = create_thesis_baseline_config(
            mechanism="PRD",
            seed=201,
        )

        # Modify it for 10 agents and 2 representatives for this specific test suite
        num_test_agents = 10
        # Baseline is 6 delegates for 15 agents (~40%). For 10 agents, 4 delegates.
        num_test_delegates = 4 
        # The tests expect 2 representatives to be elected.
        num_representatives_to_elect_for_test = 2

        self.config = dataclass_replace(base_config, # Use the imported replace function
            num_agents=num_test_agents,
            num_delegates=num_test_delegates,
            prd_num_representatives_to_elect=num_representatives_to_elect_for_test
        )
        
        key = jr.PRNGKey(0)
        # Initialize the state with the 10-agent, 2-reps config
        self.initial_state = initialize_portfolio_democracy_graph_state(key, self.config)
        
        # Now, self.initial_state.node_attrs will have all arrays of length 10.
        # Set the specific values for cognitive_resources and is_delegate for test logic.
        node_attrs_copy = self.initial_state.node_attrs.copy()
        cog_res_test = jnp.array([10, 80, 30, 90, 20, 5, 15, 60, 25, 50], dtype=jnp.int32) 
        node_attrs_copy["cognitive_resources"] = cog_res_test
        # is_delegate for 10 agents, with num_test_delegates (4) being True.
        node_attrs_copy["is_delegate"] = jnp.array([True, True, True, True, False, False, False, False, False, False], dtype=jnp.bool_)
        self.initial_state = self.initial_state.replace(node_attrs=node_attrs_copy)


    def test_election_selects_correct_number_of_representatives(self):
        election_tf = create_election_transform(election_logic="random_approval")
        
        # Ensure rounds_until_next_election_prd is 0 to trigger election
        state_for_election = self.initial_state.replace(
            global_attrs={**self.initial_state.global_attrs, "rounds_until_next_election_prd": 0}
        )
        
        next_state = election_tf(state_for_election)
        
        num_elected = jnp.sum(next_state.node_attrs["is_elected_representative"])
        self.assertEqual(num_elected, self.config.prd_num_representatives_to_elect)

    def test_representatives_have_term_set(self):
        election_tf = create_election_transform(election_logic="random_approval")
        state_for_election = self.initial_state.replace(
            global_attrs={**self.initial_state.global_attrs, "rounds_until_next_election_prd": 0}
        )
        next_state = election_tf(state_for_election)

        elected_mask = next_state.node_attrs["is_elected_representative"]
        term_lengths = next_state.node_attrs["representative_term_remaining"]
        
        for i in range(self.config.num_agents):
            if elected_mask[i]:
                self.assertEqual(term_lengths[i], self.config.prd_election_term_length)
            else:
                # Non-elected might have had their terms decremented if they were previously reps
                # For a fresh election, they should be 0 if not elected
                self.assertTrue(term_lengths[i] <= self.config.prd_election_term_length -1 or term_lengths[i] == 0)


    def test_election_highest_cog_resource_logic(self):
        # Test with the "highest_cog_resource" logic
        election_tf = create_election_transform(election_logic="highest_cog_resource")
        state_for_election = self.initial_state.replace(
            global_attrs={**self.initial_state.global_attrs, "rounds_until_next_election_prd": 0}
        )
        
        # Candidates are agents 0, 1, 2, 3 with cog_res: 10, 80, 30, 90
        # We want to elect 2. Expected elected: Agent 3 (90), Agent 1 (80)
        expected_elected_ids = jnp.array([1, 3]) # Sorted by ID for comparison if needed
        
        next_state = election_tf(state_for_election)
        
        elected_agent_ids = jnp.where(next_state.node_attrs["is_elected_representative"])[0]
        
        self.assertEqual(len(elected_agent_ids), 2)
        # Sort for comparison as order from where might not be guaranteed depending on tie-breaking
        self.assertCountEqual(elected_agent_ids.tolist(), expected_elected_ids.tolist())


    def test_term_countdown_and_re_election(self):
        election_tf = create_election_transform(election_logic="random_approval")
        
        state = self.initial_state.replace(
            global_attrs={**self.initial_state.global_attrs, "rounds_until_next_election_prd": 0}
        )
        
        # First election
        state = election_tf(state)
        initial_elected_mask = state.node_attrs["is_elected_representative"].copy()
        self.assertEqual(state.global_attrs["rounds_until_next_election_prd"], self.config.prd_election_term_length - 1)

        # Simulate rounds passing without an election
        for r in range(1, self.config.prd_election_term_length):
            # Update global_attrs to reflect round progression for the election_tf
            current_round_num = state.global_attrs.get("round_num",0) + 1
            state = state.replace(global_attrs={**state.global_attrs, "round_num": current_round_num})
            
            state_before_tf = state # Save state before election_tf potentially modifies countdown
            state = election_tf(state) # This will just decrement countdowns
            self.assertTrue(jnp.array_equal(state.node_attrs["is_elected_representative"], initial_elected_mask))
            self.assertEqual(state_before_tf.global_attrs["rounds_until_next_election_prd"] -1, state.global_attrs["rounds_until_next_election_prd"])
            self.assertEqual(state.node_attrs["representative_term_remaining"][initial_elected_mask][0], self.config.prd_election_term_length - r)


        # Next round should trigger a new election
        current_round_num = state.global_attrs.get("round_num",0) + 1
        state = state.replace(global_attrs={**state.global_attrs, "round_num": current_round_num})
        self.assertEqual(state.global_attrs["rounds_until_next_election_prd"], 0)
        state_after_re_election = election_tf(state)
        
        # Check if new representatives might have been elected (could be same, but process ran)
        self.assertTrue(jnp.sum(state_after_re_election.node_attrs["is_elected_representative"]) == self.config.prd_num_representatives_to_elect)
        self.assertEqual(state_after_re_election.global_attrs["rounds_until_next_election_prd"], self.config.prd_election_term_length - 1)

if __name__ == '__main__':
    unittest.main()