# tests/simulations/test_full_simulation_runs.py
import unittest
import jax
import jax.numpy as jnp
import jax.random as jr
import os
import sys
from pathlib import Path
import time
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.graph import GraphState
from environments.democracy.configuration import create_thesis_baseline_config, PortfolioDemocracyConfig, CropConfig, PortfolioStrategyConfig
from services.llm import LLMService, create_llm_service
from environments.democracy.mechanism_factory import create_llm_agent_decision_transform # To add logging

run_single_simulation_imported_function = None
import_error_message = None
try:
    from main.run_portfolio_simulation import run_single_simulation as rss_func
    run_single_simulation_imported_function = rss_func
    print("[INFO] Successfully imported run_single_simulation.")
except ModuleNotFoundError:
    import_error_message = "ERROR: Could not find run_portfolio_simulation.py. Check path. Assumed root."
    print(import_error_message)
except ImportError as e:
    import_error_message = f"ERROR: Imported run_portfolio_simulation.py but failed to import run_single_simulation function: {e}"
    print(import_error_message)

module_llm_service = None

# --- Monkey patch create_llm_agent_decision_transform for logging within tests ---
# Store the original function
original_create_llm_agent_decision_transform = create_llm_agent_decision_transform

def patched_create_llm_agent_decision_transform(llm_service, mechanism, sim_config):
    original_transform_func = original_create_llm_agent_decision_transform(llm_service, mechanism, sim_config)
    
    def logged_transform_wrapper(state: GraphState) -> GraphState:
        # Logic to extract and print portfolio_options_str before calling original transform
        # This needs to be done *inside* the agent loop of the original transform logic
        # or we need to reconstruct it here just for printing, which is less ideal.
        
        # Simpler: Add print inside the original create_llm_agent_decision_transform
        # as suggested in the previous response. If that's done, no patch needed here.
        # Assuming the print is added to the original function, this patch isn't strictly needed.
        # This is more for if you can't modify the original easily.

        # For this example, we'll assume the print is in the original, so no patch needed here for now.
        # If you can't modify original, we'd have to re-implement part of its logic to get portfolio_options_str
        # before calling it.

        print(f"DEBUG_PATCH: Called patched_create_llm_agent_decision_transform for {mechanism}")
        return original_transform_func(state)

    # return logged_transform_wrapper # Uncomment if you implement a patching wrapper
    return original_transform_func # Return original if print is in original file
# --- End of Patching ---


def setUpModule():
    print("\nDEBUG: test_full_simulation_runs.py - Entering setUpModule()")
    global module_llm_service
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        print(f"DEBUG: test_full_simulation_runs.py - API Key from os.getenv: '{api_key}'")
        if api_key:
            module_llm_service = create_llm_service(model="google/gemini-flash-1.5", api_key=api_key)
            print("[SIM_TEST_INFO] Real LLM Service (Gemini 1.5 Flash) initialized for simulation tests in setUpModule.")
            print(f"DEBUG: test_full_simulation_runs.py - module_llm_service object: {module_llm_service}")
        else:
            print("[SIM_TEST_WARN] OPENROUTER_API_KEY not found. Real LLM simulation tests will use None for LLM service.")
            module_llm_service = None
    except Exception as e:
        print(f"[SIM_TEST_ERROR] Failed to initialize real LLM service for simulation tests: {e}")
        module_llm_service = None
    print("DEBUG: test_full_simulation_runs.py - Exiting setUpModule()")

class TestFullSimulationRuns(unittest.TestCase):

    def _run_mechanism_test(self, mechanism_name, num_rounds_test=3, num_agents_test=3):
        if run_single_simulation_imported_function is None:
            self.skipTest(f"run_single_simulation function not available. Import error: {import_error_message}")

        print(f"\n--- Testing Mechanism: {mechanism_name} ({num_agents_test} agents, {num_rounds_test} rounds) ---")
        
        config_seed = int(jr.key_data(jr.PRNGKey(hash(mechanism_name + "_config_seed")))[0])
        
        # Create baseline config with VERY LOW NOISE for testing
        # And ensure it uses 3 crops for compatibility with default portfolio weights
        sim_config_base = create_thesis_baseline_config(
            mechanism=mechanism_name,
            seed=config_seed,
            prediction_market_sigma=0.01, # <<< VERY LOW NOISE
            num_crops_config=3 # Ensure 3 crops for default portfolios
        )
        
        num_test_delegates = max(1, int(num_agents_test * 0.4))

        current_test_config = PortfolioDemocracyConfig(
            mechanism=mechanism_name,
            num_agents=num_agents_test,
            num_delegates=num_test_delegates,
            num_rounds=num_rounds_test,
            seed=config_seed, # Seed for initializing agent attributes etc.
            crops=sim_config_base.crops,
            portfolios=sim_config_base.portfolios, # These now match 3 crops
            resources=sim_config_base.resources,
            agent_settings=sim_config_base.agent_settings,
            token_budget_settings=sim_config_base.token_budget_settings,
            market_settings=sim_config_base.market_settings, # Will have sigma=0.01
            prompt_settings=sim_config_base.prompt_settings
        )

        llm_to_use = module_llm_service
        print(f"Config: {current_test_config.num_agents} agents, {current_test_config.num_delegates} delegates, {current_test_config.num_rounds} rounds.")
        print(f"Using {len(current_test_config.crops)} crops and {len(current_test_config.portfolios)} portfolios for this test.")
        print(f"Market Noise Sigma for test: {current_test_config.market_settings.prediction_noise_sigma}")
        # Optional: Print true expected yields for the first round for context
        if current_test_config.crops:
            print(f"True Expected Yields for Round 1 (from config): {[c.true_expected_yields_per_round[0] for c in current_test_config.crops if c.true_expected_yields_per_round]}")
        
        print(f"LLM service for this run: {'Real LLM' if llm_to_use else 'None (API Key not found or init failed)'}")

        sim_run_key = jr.PRNGKey(current_test_config.seed + 789) # Different seed for the simulation run key

        results_df = run_single_simulation_imported_function(
            key=sim_run_key,
            sim_config=current_test_config,
            llm_service=llm_to_use
        )

        self.assertIsNotNone(results_df, f"{mechanism_name} simulation failed to produce results.")
        self.assertIsInstance(results_df, pd.DataFrame, f"{mechanism_name} did not return a DataFrame.")
        self.assertFalse(results_df.empty, f"{mechanism_name} simulation produced an empty DataFrame.")
        
        self.assertEqual(len(results_df), num_rounds_test,
                         f"{mechanism_name} did not run for the expected {num_rounds_test} rounds. Actual rows: {len(results_df)} vs expected rounds {num_rounds_test}.")

        final_resources = results_df['resources_after'].iloc[-1]
        print(f"Mechanism {mechanism_name} final resources: {final_resources:.2f}")
        self.assertTrue(jnp.isfinite(final_resources), f"{mechanism_name} final resources is not finite: {final_resources}")
        return final_resources

    def test_run_pdd_simulation(self):
        self._run_mechanism_test("PDD")

    def test_run_prd_simulation(self):
        self._run_mechanism_test("PRD")

    def test_run_pld_simulation(self):
        self._run_mechanism_test("PLD")

    def test_compare_mechanism_outcomes(self):
        results = {}
        mechanisms_to_test = ["PDD", "PRD", "PLD"]
        for mech_idx, mech in enumerate(mechanisms_to_test):
            if mech_idx > 0 and module_llm_service is not None:
                print(f"DEBUG: Sleeping for 3s before testing {mech}...") # Slightly longer sleep
                time.sleep(3)
            results[mech] = self._run_mechanism_test(mech)

        print("\n--- Final Resource Comparison (End-to-End Test) ---")
        for mech, res in results.items():
            print(f"{mech}: {res:.2f}")

        if module_llm_service:
            if len(set(results.values())) == 1 and len(mechanisms_to_test) > 1 :
                 print("[WARN] All mechanisms produced identical final resources in end-to-end test. "
                       "Check LLM responses for differentiated choices based on new yield patterns.")
        else:
            print("Skipping outcome comparison details as no real LLM service was used for end-to-end test.")

if __name__ == '__main__':
    unittest.main()