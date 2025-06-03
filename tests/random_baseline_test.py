# tests/random_baseline_test.py
"""
Test suite to compare mechanism performance against a random choice baseline
based on aggregated expected yields over multiple rounds.
"""

import unittest
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

import sys
from pathlib import Path

# Project imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState
from environments.noise_democracy.configuration import (
    PortfolioDemocracyConfig,
    CropConfig,
    PortfolioStrategyConfig,
    create_thesis_baseline_config as actual_create_thesis_baseline_config # Alias
)
from environments.noise_democracy.initialization import (
    initialize_portfolio_democracy_graph_state,
    get_true_expected_yields_for_round
)
from environments.noise_democracy.mechanism_factory import create_portfolio_mechanism_pipeline
from services.llm import create_llm_service, LLMService

class TestMechanismVsRandomBaseline(unittest.TestCase):
    """
    Tests comparing mechanism-driven portfolio selection against a random baseline.
    Focuses on the aggregation of *expected* yields based on *true* crop data.
    """

    @classmethod
    def setUpClass(cls):
        """Initialize LLM service (optional, for mechanism run)."""
        try:
            import os
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                cls.llm_service = create_llm_service(model="google/gemini-2.5-flash-preview-05-20", api_key=api_key)
                print("[TEST_RANDOM_BASELINE] Real LLM service initialized.")
            else:
                cls.llm_service = None
                print("[TEST_RANDOM_BASELINE] No API key for LLM service; mechanism run might be limited.")
        except Exception as e:
            cls.llm_service = None
            print(f"[TEST_RANDOM_BASELINE] LLM service initialization failed: {e}")

    def _run_simulation_get_expected_yield_trajectory(
        self,
        config: PortfolioDemocracyConfig,
        initial_state: GraphState,
        num_rounds: int,
        strategy: str = "mechanism" # "mechanism" or "random"
    ) -> List[float]:
        """
        Helper to run a simulation for N rounds and collect a trajectory of
        EXPECTED yields based on the chosen portfolio and TRUE crop yields.
        """
        if strategy == "mechanism" and not self.llm_service:
            print("Skipping mechanism-based run as LLM service is not available for test.")
            return [1.0] * num_rounds # Return a dummy trajectory (e.g., always 1.0 yield)

        current_state = initial_state
        expected_yield_trajectory = []
        
        round_pipeline = create_portfolio_mechanism_pipeline(
            mechanism=config.mechanism,
            llm_service=self.llm_service,
            sim_config=config
        )

        for i_round in range(num_rounds):
            # Ensure round_num in global_attrs is an integer for get_true_expected_yields_for_round
            current_round_num_for_yields = int(current_state.global_attrs.get("round_num", i_round))

            true_expected_crop_yields_this_round = get_true_expected_yields_for_round(
                current_round_num_for_yields, 
                config.crops
            )
            current_state = current_state.replace(
                global_attrs={
                    **current_state.global_attrs,
                    "current_true_expected_crop_yields": true_expected_crop_yields_this_round
                }
            )

            if strategy == "random":
                num_portfolios = len(config.portfolios)
                key_random_choice = jr.PRNGKey(config.seed + current_round_num_for_yields + 777)
                random_portfolio_idx = jr.choice(key_random_choice, jnp.arange(num_portfolios))
                
                current_state = current_state.replace(
                    global_attrs={**current_state.global_attrs, "current_decision": random_portfolio_idx}
                )
            elif strategy == "mechanism":
                current_state = round_pipeline(current_state) # This updates round_num internally
            
            chosen_portfolio_idx = current_state.global_attrs.get("current_decision")
            if chosen_portfolio_idx is None or not (0 <= chosen_portfolio_idx < len(config.portfolios)):
                print(f"Warning: Invalid chosen_portfolio_idx {chosen_portfolio_idx} in round {current_round_num_for_yields}. Defaulting to 0 yield.")
                expected_yield_trajectory.append(0.0)
                # Minimal state update for next round if we continue after warning
                current_state = current_state.replace(
                    global_attrs={**current_state.global_attrs, "round_num": current_state.global_attrs.get("round_num", -1) + 1}
                )
                continue

            chosen_portfolio_config = config.portfolios[int(chosen_portfolio_idx)]
            portfolio_weights = jnp.array(chosen_portfolio_config.weights)
            
            current_round_expected_yield = jnp.sum(portfolio_weights * true_expected_crop_yields_this_round)
            expected_yield_trajectory.append(float(current_round_expected_yield))

            # If strategy is 'random', we need to manually increment round_num for the next iteration's yield fetching
            # If strategy is 'mechanism', round_pipeline (via housekeeping_transform) handles it.
            if strategy == "random":
                 current_state = current_state.replace(
                    global_attrs={**current_state.global_attrs, "round_num": current_state.global_attrs.get("round_num", -1) + 1}
                )
            
        return expected_yield_trajectory

    def test_mechanism_vs_random_expected_performance(self):
        """
        Calculates overall expected crop yield, then compares aggregated
        expected yields of a mechanism vs. random choice over 30 rounds.
        """
        print("\n=== Testing Mechanism vs. Random Expected Performance ===")
        num_test_rounds = 30
        test_mechanism = "PDD" 
        adversarial_prop = 0.0 # Test with no adversaries for a clearer baseline

        config = actual_create_thesis_baseline_config(
            mechanism=test_mechanism, 
            adversarial_proportion_total=adversarial_prop, 
            seed=2024
        )
        key = jr.PRNGKey(config.seed)
        initial_state = initialize_portfolio_democracy_graph_state(key, config)

        # 1. Calculate Overall Expected Crop Yield (long-term average)
        mean_yields_per_crop = [np.mean(crop.true_expected_yields_per_round) for crop in config.crops]
        overall_expected_crop_yield = np.mean(mean_yields_per_crop) if mean_yields_per_crop else 0.0
        print(f"Overall Average Expected Crop Yield (across all crops & rounds): {overall_expected_crop_yield:.3f}")

        # 2. Run with mechanism
        mechanism_yield_trajectory = self._run_simulation_get_expected_yield_trajectory(
            config, initial_state, num_test_rounds, strategy="mechanism"
        )
        # 3. Run with random choice
        # Re-initialize state for a fair comparison if state is modified in-place by the helper
        initial_state_for_random = initialize_portfolio_democracy_graph_state(key, config)
        random_yield_trajectory = self._run_simulation_get_expected_yield_trajectory(
            config, initial_state_for_random, num_test_rounds, strategy="random"
        )

        cumulative_mechanism_expected_yield = sum(mechanism_yield_trajectory)
        cumulative_random_expected_yield = sum(random_yield_trajectory)

        print(f"Mechanism ({test_mechanism}) Cumulative Expected Yield over {num_test_rounds} rounds: {cumulative_mechanism_expected_yield:.2f}")
        print(f"Random Choice Cumulative Expected Yield over {num_test_rounds} rounds: {cumulative_random_expected_yield:.2f}")

        self.assertEqual(len(mechanism_yield_trajectory), num_test_rounds)
        self.assertEqual(len(random_yield_trajectory), num_test_rounds)
        
        # This is an exploratory print; a formal assertion would depend on specific expectations
        print(f"Note: Comparison depends on LLM quality, specific config, and noise if actual yields were used.")

if __name__ == '__main__':
    unittest.main()