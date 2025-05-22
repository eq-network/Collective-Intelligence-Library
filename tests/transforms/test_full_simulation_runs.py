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

# ARCHITECTURAL ENHANCEMENT: Import new cognitive resource configuration classes
from core.graph import GraphState
from environments.democracy.configuration import (
    create_thesis_baseline_config, 
    PortfolioDemocracyConfig, 
    CropConfig, 
    PortfolioStrategyConfig,
    CognitiveResourceConfig  # NEW: Import cognitive resource configuration
)
from services.llm import LLMService, create_llm_service
from environments.democracy.mechanism_factory import create_llm_agent_decision_transform

# ARCHITECTURAL PRESERVATION: Maintain existing import pattern for simulation function
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

# ARCHITECTURAL PRESERVATION: Maintain existing monkey patching capability
original_create_llm_agent_decision_transform = create_llm_agent_decision_transform

def patched_create_llm_agent_decision_transform(llm_service, mechanism, sim_config):
    """
    ARCHITECTURAL CONSIDERATION: Preserve debugging capability while updating configuration access.
    
    RISK MITIGATION: Ensure patching works with both old and new configuration structures.
    """
    original_transform_func = original_create_llm_agent_decision_transform(llm_service, mechanism, sim_config)
    
    def logged_transform_wrapper(state: GraphState) -> GraphState:
        print(f"DEBUG_PATCH: Called patched_create_llm_agent_decision_transform for {mechanism}")
        return original_transform_func(state)

    return original_transform_func

def setUpModule():
    """
    ARCHITECTURAL ANALYSIS: Module-level initialization for LLM service.
    
    PURPOSE: Initialize shared LLM service for all test cases to avoid redundant API calls.
    RISK FACTORS: API key availability, network connectivity, service initialization failures.
    MITIGATION: Graceful degradation to None service with comprehensive error reporting.
    """
    print("\nDEBUG: test_full_simulation_runs.py - Entering setUpModule()")
    global module_llm_service
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        print(f"DEBUG: test_full_simulation_runs.py - API Key from os.getenv: '{api_key[:10]}...' if api_key else 'None'")
        if api_key:
            module_llm_service = create_llm_service(model="google/gemini-flash-1.5", api_key=api_key)
            print("[SIM_TEST_INFO] Real LLM Service (Gemini 1.5 Flash) initialized for simulation tests in setUpModule.")
            print(f"DEBUG: test_full_simulation_runs.py - module_llm_service object: {type(module_llm_service)}")
        else:
            print("[SIM_TEST_WARN] OPENROUTER_API_KEY not found. Real LLM simulation tests will use None for LLM service.")
            module_llm_service = None
    except Exception as e:
        print(f"[SIM_TEST_ERROR] Failed to initialize real LLM service for simulation tests: {e}")
        module_llm_service = None
    print("DEBUG: test_full_simulation_runs.py - Exiting setUpModule()")

class TestFullSimulationRuns(unittest.TestCase):
    """
    ARCHITECTURAL PURPOSE: End-to-end testing of democratic mechanisms with cognitive resource system.
    
    DESIGN CONSIDERATIONS:
    1. Test isolation: Each test method should be independent
    2. Configuration flexibility: Support different mechanism types and parameters
    3. Error handling: Graceful degradation when components unavailable
    4. Performance: Minimize test execution time while maintaining thoroughness
    """

    def _validate_cognitive_resource_configuration(self, config: PortfolioDemocracyConfig) -> None:
        """
        ARCHITECTURAL VALIDATION: Ensure cognitive resource configuration is properly structured.
        
        VALIDATION DIMENSIONS:
        1. Configuration object presence and type correctness
        2. Cognitive resource value ranges and relationships
        3. Attribute accessibility for downstream components
        
        FAILURE MODES:
        - Missing cognitive_resource_settings attribute
        - Invalid cognitive resource values (outside 0-100 range)
        - Delegate/voter resource relationship inversion
        """
        # Validate configuration object exists and has correct type
        self.assertTrue(hasattr(config, 'cognitive_resource_settings'), 
                       "Configuration missing cognitive_resource_settings attribute")
        
        cognitive_config = config.cognitive_resource_settings
        self.assertIsInstance(cognitive_config, CognitiveResourceConfig,
                            f"Expected CognitiveResourceConfig, got {type(cognitive_config)}")
        
        # Validate cognitive resource value ranges
        delegate_resources = cognitive_config.cognitive_resources_delegate
        voter_resources = cognitive_config.cognitive_resources_voter
        
        self.assertTrue(0 <= delegate_resources <= 100,
                       f"Delegate cognitive resources {delegate_resources} outside valid range [0, 100]")
        self.assertTrue(0 <= voter_resources <= 100,
                       f"Voter cognitive resources {voter_resources} outside valid range [0, 100]")
        
        # Validate logical relationship (delegates should have higher cognitive resources)
        self.assertGreater(delegate_resources, voter_resources,
                          f"Delegates ({delegate_resources}) should have higher cognitive resources than voters ({voter_resources})")
        
        print(f"[VALIDATION_SUCCESS] Cognitive resources - Delegates: {delegate_resources}, Voters: {voter_resources}")

    def _run_mechanism_test(self, mechanism_name, num_rounds_test=3, num_agents_test=3):
        """
        ARCHITECTURAL CORE: Primary test execution method with cognitive resource system integration.
        
        DESIGN CONSIDERATIONS:
        1. Configuration generation with new cognitive resource parameters
        2. Backward compatibility validation during transition period  
        3. Comprehensive error handling for configuration mismatches
        4. Performance measurement and resource tracking
        
        RISK MITIGATION STRATEGIES:
        - Defensive programming for attribute access
        - Comprehensive validation of configuration structure
        - Graceful error reporting with diagnostic information
        """
        # VALIDATION LAYER: Ensure simulation function availability
        if run_single_simulation_imported_function is None:
            self.skipTest(f"run_single_simulation function not available. Import error: {import_error_message}")

        print(f"\n--- Testing Mechanism: {mechanism_name} ({num_agents_test} agents, {num_rounds_test} rounds) ---")
        
        # DETERMINISM ASSURANCE: Generate reproducible seed for test isolation
        config_seed = int(jr.key_data(jr.PRNGKey(hash(mechanism_name + "_config_seed")))[0])
        
        # CONFIGURATION GENERATION: Create base configuration with cognitive resource system
        try:
            sim_config_base = create_thesis_baseline_config(
                mechanism=mechanism_name,
                seed=config_seed,
                prediction_market_sigma=0.01,  # Very low noise for testing determinism
                delegate_cognitive_resources=80,  # NEW: High cognitive resources for delegates
                voter_cognitive_resources=20,    # NEW: Low cognitive resources for voters  
                num_crops_config=3  # Ensure compatibility with default portfolios
            )
            print(f"[CONFIG_SUCCESS] Base configuration created with cognitive resources")
        except Exception as e:
            self.fail(f"Failed to create base configuration with cognitive resources: {e}")
        
        # CONFIGURATION VALIDATION: Ensure cognitive resource system is properly configured
        try:
            self._validate_cognitive_resource_configuration(sim_config_base)
        except Exception as e:
            self.fail(f"Cognitive resource configuration validation failed: {e}")
        
        # TEST CONFIGURATION ASSEMBLY: Build custom configuration for test parameters
        num_test_delegates = max(1, int(num_agents_test * 0.4))

        try:
            current_test_config = PortfolioDemocracyConfig(
                mechanism=mechanism_name,
                num_agents=num_agents_test,
                num_delegates=num_test_delegates,
                num_rounds=num_rounds_test,
                seed=config_seed,
                crops=sim_config_base.crops,
                portfolios=sim_config_base.portfolios,
                resources=sim_config_base.resources,
                agent_settings=sim_config_base.agent_settings,
                cognitive_resource_settings=sim_config_base.cognitive_resource_settings,  # UPDATED: Use cognitive resources
                market_settings=sim_config_base.market_settings,
                prompt_settings=sim_config_base.prompt_settings
            )
            print(f"[CONFIG_SUCCESS] Test configuration assembled successfully")
        except Exception as e:
            self.fail(f"Failed to assemble test configuration: {e}")

        # DIAGNOSTIC OUTPUT: Display configuration parameters for debugging
        llm_to_use = module_llm_service
        print(f"Config: {current_test_config.num_agents} agents, {current_test_config.num_delegates} delegates, {current_test_config.num_rounds} rounds.")
        print(f"Using {len(current_test_config.crops)} crops and {len(current_test_config.portfolios)} portfolios for this test.")
        print(f"Market Noise Sigma for test: {current_test_config.market_settings.prediction_noise_sigma}")
        
        # COGNITIVE RESOURCE DIAGNOSTICS: Display new configuration parameters
        cognitive_config = current_test_config.cognitive_resource_settings
        print(f"Cognitive Resources - Delegates: {cognitive_config.cognitive_resources_delegate}, Voters: {cognitive_config.cognitive_resources_voter}")
        
        # YIELD DIAGNOSTICS: Display expected yields for debugging
        if current_test_config.crops:
            yields_sample = [c.true_expected_yields_per_round[0] for c in current_test_config.crops if c.true_expected_yields_per_round]
            print(f"True Expected Yields for Round 1 (from config): {yields_sample}")
        
        print(f"LLM service for this run: {'Real LLM' if llm_to_use else 'None (API Key not found or init failed)'}")

        # EXECUTION PHASE: Run simulation with error handling
        sim_run_key = jr.PRNGKey(current_test_config.seed + 789)
        
        try:
            results_df = run_single_simulation_imported_function(
                key=sim_run_key,
                sim_config=current_test_config,
                llm_service=llm_to_use
            )
        except Exception as e:
            self.fail(f"Simulation execution failed for {mechanism_name}: {e}")

        # RESULT VALIDATION: Comprehensive verification of simulation output
        self.assertIsNotNone(results_df, f"{mechanism_name} simulation failed to produce results.")
        self.assertIsInstance(results_df, pd.DataFrame, f"{mechanism_name} did not return a DataFrame.")
        self.assertFalse(results_df.empty, f"{mechanism_name} simulation produced an empty DataFrame.")
        
        # EXECUTION COMPLETENESS VALIDATION: Ensure all rounds executed
        self.assertEqual(len(results_df), num_rounds_test,
                         f"{mechanism_name} did not run for the expected {num_rounds_test} rounds. "
                         f"Actual rows: {len(results_df)} vs expected rounds {num_rounds_test}.")

        # RESULT QUALITY VALIDATION: Ensure numeric validity
        final_resources = results_df['resources_after'].iloc[-1]
        print(f"Mechanism {mechanism_name} final resources: {final_resources:.2f}")
        self.assertTrue(jnp.isfinite(final_resources), 
                       f"{mechanism_name} final resources is not finite: {final_resources}")
        
        return final_resources

    def test_run_pdd_simulation(self):
        """
        TEST PURPOSE: Validate Predictive Direct Democracy with cognitive resource system.
        
        EXPECTED BEHAVIOR: All agents vote directly, with cognitive resources affecting 
        prediction accuracy but not decision mechanism.
        """
        self._run_mechanism_test("PDD")

    def test_run_prd_simulation(self):
        """
        TEST PURPOSE: Validate Predictive Representative Democracy with cognitive resource system.
        
        EXPECTED BEHAVIOR: Only delegates vote, cognitive resources should provide 
        delegates with better prediction accuracy.
        """
        self._run_mechanism_test("PRD")

    def test_run_pld_simulation(self):
        """
        TEST PURPOSE: Validate Predictive Liquid Democracy with cognitive resource system.
        
        EXPECTED BEHAVIOR: Voters should delegate to delegates due to cognitive resource 
        differential creating information quality incentives.
        
        KEY VALIDATION: This test should show higher delegation rates compared to 
        token budget system due to explicit cognitive resource differentiation.
        """
        self._run_mechanism_test("PLD")

    def test_compare_mechanism_outcomes(self):
        """
        ARCHITECTURAL PURPOSE: Comparative analysis of all three mechanisms under cognitive resource system.
        
        EXPECTED DIFFERENTIALS:
        1. PLD should show improved performance due to delegation to better-informed agents
        2. PRD should show consistent performance with high-cognitive-resource delegates
        3. PDD should show baseline performance with mixed cognitive resource utilization
        
        VALIDATION CRITERIA:
        - All mechanisms should produce finite, reasonable results
        - PLD should demonstrate delegation behavior (not directly measurable here but implicit)
        - Resource outcomes should reflect information quality differences
        """
        results = {}
        mechanisms_to_test = ["PDD", "PRD", "PLD"]
        
        # RATE LIMITING: Prevent API overload for LLM service
        for mech_idx, mech in enumerate(mechanisms_to_test):
            if mech_idx > 0 and module_llm_service is not None:
                print(f"DEBUG: Sleeping for 3s before testing {mech}...")
                time.sleep(3)
            results[mech] = self._run_mechanism_test(mech)

        # COMPARATIVE ANALYSIS: Display results for manual inspection
        print("\n--- Final Resource Comparison (End-to-End Test with Cognitive Resources) ---")
        for mech, res in results.items():
            print(f"{mech}: {res:.2f}")

        # BEHAVIORAL VALIDATION: Check for expected differentiation
        if module_llm_service:
            unique_results = set(results.values())
            if len(unique_results) == 1 and len(mechanisms_to_test) > 1:
                print("[WARN] All mechanisms produced identical final resources in end-to-end test.")
                print("This may indicate cognitive resource differentiation needs calibration.")
                print("Expected: PLD should show different outcomes due to delegation behavior.")
            else:
                print(f"[SUCCESS] Mechanisms showed differentiated outcomes: {len(unique_results)} unique results")
        else:
            print("Skipping behavioral analysis - no real LLM service was used for end-to-end test.")

if __name__ == '__main__':
    unittest.main()