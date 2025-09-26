# tests/test_fruit_trading.py
import unittest
import numpy as np
import jax.numpy as jnp

from graph_transformation.core.graph import GraphState
from models.fruit_trading import (
    FruitTradingConfig, generate_preferences,
    generate_initial_endowment, calculate_utility
)
from transformations.trading import (
    TradeOpportunitySearch, TradeMatching,
    TradeNegotiation, UtilityCalculation
)
from simulations.fruit_trading import FruitTradingSimulation


class TestFruitTrading(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FruitTradingConfig(
            num_agents=5,
            num_fruits=3,
            num_rounds=3,
            fruits=["Apple", "Banana", "Cherry"],
            initial_endowment_range=(5, 10)
        )
    
    def test_graph_state_initialization(self):
        """Test creation of initial graph state."""
        # Create preferences and endowments
        preferences = generate_preferences(self.config.num_agents, self.config.num_fruits)
        endowments = generate_initial_endowment(
            self.config.num_agents,
            self.config.num_fruits,
            self.config.initial_endowment_range
        )
        utilities = calculate_utility(preferences, endowments)
        
        # Create graph state
        state = GraphState(
            node_attrs={
                "fruit_preferences": jnp.array(preferences),
                "fruit_endowments": jnp.array(endowments),
                "utility": jnp.array(utilities)
            },
            adj_matrices={},
            global_attrs={"fruit_names": self.config.fruits}
        )
        
        # Check state integrity
        self.assertEqual(state.node_attrs["fruit_preferences"].shape, 
                        (self.config.num_agents, self.config.num_fruits))
        self.assertEqual(state.node_attrs["fruit_endowments"].shape, 
                        (self.config.num_agents, self.config.num_fruits))
        self.assertEqual(state.global_attrs["fruit_names"], self.config.fruits)
    
    def test_utility_calculation(self):
        """Test that utility is correctly calculated."""
        # Create preferences and endowments with known values
        preferences = np.array([
            [5.0, 2.0, 8.0],  # Agent 0
            [3.0, 7.0, 1.0],  # Agent 1
            [2.0, 9.0, 4.0],  # Agent 2
            [7.0, 3.0, 5.0],  # Agent 3
            [1.0, 6.0, 3.0]   # Agent 4
        ])
        
        endowments = np.array([
            [10, 5, 2],  # Agent 0
            [3, 8, 5],   # Agent 1
            [6, 4, 7],   # Agent 2
            [8, 2, 9],   # Agent 3
            [4, 10, 3]   # Agent 4
        ])
        
        # Calculate expected utilities manually
        expected_utilities = [
            5.0*10 + 2.0*5 + 8.0*2,    # Agent 0: 5*10 + 2*5 + 8*2 = 76
            3.0*3 + 7.0*8 + 1.0*5,     # Agent 1: 3*3 + 7*8 + 1*5 = 70
            2.0*6 + 9.0*4 + 4.0*7,     # Agent 2: 2*6 + 9*4 + 4*7 = 76
            7.0*8 + 3.0*2 + 5.0*9,     # Agent 3: 7*8 + 3*2 + 5*9 = 107
            1.0*4 + 6.0*10 + 3.0*3     # Agent 4: 1*4 + 6*10 + 3*3 = 73
        ]
        
        # Calculate with our function
        actual_utilities = calculate_utility(preferences, endowments)
        
        # Check values match
        np.testing.assert_array_almost_equal(actual_utilities, expected_utilities)
    
    def test_trade_opportunity_search(self):
        """Test that trade opportunities are correctly identified."""
        # Create state with known preferences
        preferences = np.array([
            [8.0, 2.0, 4.0],  # Agent 0: strongly prefers apples
            [2.0, 9.0, 3.0],  # Agent 1: strongly prefers bananas
            [3.0, 4.0, 7.0]   # Agent 2: strongly prefers cherries
        ])
        
        endowments = np.array([
            [5, 5, 5],  # Agent 0
            [5, 5, 5],  # Agent 1
            [5, 5, 5]   # Agent 2
        ])
        
        utilities = calculate_utility(preferences, endowments)
        
        state = GraphState(
            node_attrs={
                "fruit_preferences": jnp.array(preferences),
                "fruit_endowments": jnp.array(endowments),
                "utility": jnp.array(utilities)
            },
            adj_matrices={},
            global_attrs={"fruit_names": ["Apple", "Banana", "Cherry"]}
        )
        
        # Apply trade opportunity search
        transform = TradeOpportunitySearch()
        new_state = transform.apply(state)
        
        # Check that trading_opportunity matrix exists
        self.assertIn("trading_opportunity", new_state.adj_matrices)
        
        # Check dimensions
        opportunity_matrix = np.array(new_state.adj_matrices["trading_opportunity"])
        self.assertEqual(opportunity_matrix.shape, (3, 3))
        
        # Check that diagonal is zero (no self-trading)
        self.assertEqual(opportunity_matrix[0, 0], 0)
        self.assertEqual(opportunity_matrix[1, 1], 0)
        self.assertEqual(opportunity_matrix[2, 2], 0)
        
        # Agents with very different preferences should have higher opportunity scores
        # The exact values depend on the algorithm, but we can check relative scores
        # For these preferences:
        # - Agent 0 & 1 have very different preferences (apple vs banana)
        # - Agent 0 & 2 also have different preferences (apple vs cherry)
        # - Agent 1 & 2 have somewhat different preferences (banana vs cherry)
        
        # So we should expect non-zero opportunity scores between all pairs
        self.assertGreater(opportunity_matrix[0, 1], 0)
        self.assertGreater(opportunity_matrix[0, 2], 0)
        self.assertGreater(opportunity_matrix[1, 2], 0)
    
    def test_simple_trade_negotiation(self):
        """Test that trade negotiation improves utility."""
        # Create a simple scenario with two agents who clearly benefit from trading
        preferences = np.array([
            [10.0, 1.0],  # Agent 0: strongly prefers fruit 0
            [1.0, 10.0]   # Agent 1: strongly prefers fruit 1
        ])
        
        endowments = np.array([
            [2, 8],  # Agent 0: has more of fruit 1 (which they don't prefer)
            [8, 2]   # Agent 1: has more of fruit 0 (which they don't prefer)
        ])
        
        initial_utilities = calculate_utility(preferences, endowments)
        
        state = GraphState(
            node_attrs={
                "fruit_preferences": jnp.array(preferences),
                "fruit_endowments": jnp.array(endowments),
                "utility": jnp.array(initial_utilities)
            },
            adj_matrices={
                # Create direct trading match between agents 0 and 1
                "trading_matches": jnp.array([[0, 1], [1, 0]])
            },
            global_attrs={
                "fruit_names": ["Apple", "Banana"],
                "trade_history": [],
                "round": 0
            }
        )
        
        # Apply trade negotiation without LLM
        transform = TradeNegotiation(llm_client=None)
        new_state = transform.apply(state)
        
        # Get new endowments and utilities
        new_endowments = np.array(new_state.node_attrs["fruit_endowments"])
        new_utilities = np.array(new_state.node_attrs["utility"])
        
        # Check that utility improved for both agents
        self.assertGreater(new_utilities[0], initial_utilities[0])
        self.assertGreater(new_utilities[1], initial_utilities[1])
        
        # Check that at least some trading happened
        self.assertFalse(np.array_equal(endowments, new_endowments))
        
        # Check that trade was recorded
        self.assertGreater(len(new_state.global_attrs["trade_history"]), 0)
    
    def test_full_simulation(self):
        """Test that a full simulation runs without errors."""
        # Create simulation with small config for faster testing
        small_config = FruitTradingConfig(
            num_agents=3, 
            num_fruits=3, 
            num_rounds=2, 
            fruits=["Apple", "Banana", "Cherry"]
        )
        simulation = FruitTradingSimulation(small_config)
        
        # Run simulation without visualizations
        final_state = simulation.run(verbose=False, visualize=False)
        
        # Check that simulation completed all rounds
        self.assertEqual(final_state.global_attrs.get("round"), small_config.num_rounds)
        
        # Check that states were recorded
        self.assertEqual(len(simulation.get_state_history()), small_config.num_rounds + 1)
        
        # Calculate efficiency metrics
        efficiency = simulation.calculate_trading_efficiency()
        
        # Efficiency metrics should have key values
        self.assertIn("utility_gain", efficiency)
        self.assertIn("total_trades", efficiency)
        
        # No agents should be worse off (all trades should be Pareto improvements)
        self.assertEqual(efficiency["agents_worse_off"], 0)


if __name__ == "__main__":
    unittest.main()