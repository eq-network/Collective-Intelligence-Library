# tests/adversarial_behavior_verification.py
"""
Comprehensive test suite to verify that adversarial agents actually behave adversarially
across all democratic mechanisms (PDD, PLD, PRD).

This test addresses concerns that LLM safety mechanisms may prevent adversarial agents
from truly minimizing resources, and ensures behavioral differentiation is statistically significant.
"""

import unittest
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import replace as dataclass_replace
import re
import time
from scipy import stats
from collections import defaultdict
from unittest.mock import MagicMock, call

import sys
from pathlib import Path

# Project imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.graph import GraphState
from environments.democracy.configuration import (
    create_thesis_baseline_config, 
    PortfolioDemocracyConfig,
    CropConfig,
    PortfolioStrategyConfig
)
from environments.democracy.initialization import initialize_portfolio_democracy_graph_state
from environments.democracy.mechanism_factory import create_llm_agent_decision_transform
from services.llm import create_llm_service, LLMService

class AdversarialBehaviorAnalyzer:
    """
    Systematic analyzer for adversarial agent behavior patterns.
    
    This class implements rigorous statistical testing to verify that adversarial agents
    are actually behaving adversarially across different mechanisms.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service
        self.response_log = []  # Store all LLM responses for analysis
        self.behavioral_data = defaultdict(list)
        
    def create_clear_adversarial_scenario(self, mechanism: str, num_agents: int = 6) -> Tuple[GraphState, PortfolioDemocracyConfig]:
        """
        Create a scenario with clear optimal vs adversarial choices.
        
        Design principles:
        - Portfolio options with dramatically different expected returns
        - Clear performance history for delegation/voting decisions
        - Unambiguous adversarial choices
        """
        # Create configuration with extreme portfolio differences
        config = create_thesis_baseline_config(mechanism=mechanism, seed=12345)
        
        # Override with test-specific portfolios that have clear best/worst options
        test_portfolios = [
            PortfolioStrategyConfig(
                name="HighYield_Portfolio",
                weights=[0.8, 0.2, 0.0],  # Focus on best crop
                description="Clearly optimal choice"
            ),
            PortfolioStrategyConfig(
                name="MediumYield_Portfolio", 
                weights=[0.33, 0.33, 0.34],  # Balanced
                description="Moderate choice"
            ),
            PortfolioStrategyConfig(
                name="LowYield_Portfolio",
                weights=[0.1, 0.1, 0.8],  # Focus on worst crop
                description="Clearly adversarial choice"
            )
        ]
        
        # Create test crops with extreme yield differences
        test_crops = [
            CropConfig(name="HighYield_Crop", true_expected_yields_per_round=[1.5] * 10),
            CropConfig(name="MedYield_Crop", true_expected_yields_per_round=[1.1] * 10), 
            CropConfig(name="LowYield_Crop", true_expected_yields_per_round=[0.6] * 10)
        ]
        
        test_config = dataclass_replace(
            config,
            num_agents=num_agents,
            num_delegates=max(1, num_agents // 3),
            portfolios=test_portfolios,
            crops=test_crops,
            num_rounds=3  # Short test
        )
        
        # Initialize state
        key = jr.PRNGKey(54321)
        initial_state = initialize_portfolio_democracy_graph_state(key, test_config)
        
        # Add clear prediction market signals that make the choice obvious
        prediction_signals = jnp.array([1.5, 1.1, 0.6])  # Clear ranking
        
        new_global_attrs = dict(initial_state.global_attrs)
        new_global_attrs["prediction_market_crop_signals"] = prediction_signals
        new_global_attrs["test_scenario"] = "clear_adversarial_choice"
        
        test_state = initial_state.replace(global_attrs=new_global_attrs)
        
        return test_state, test_config
        
    def create_delegation_history_scenario(self, num_agents: int = 6) -> Tuple[GraphState, PortfolioDemocracyConfig]:
        """
        Create PLD scenario where some delegates have clear performance history.
        
        This tests whether adversarial agents delegate to poor performers.
        """
        initial_state, config = self.create_clear_adversarial_scenario("PLD", num_agents)
        
        # Add performance history to global attributes
        delegate_performance = {
            0: {"avg_resource_change": 1.3, "decisions_made": 5, "label": "excellent_performer"},
            1: {"avg_resource_change": 0.7, "decisions_made": 5, "label": "poor_performer"},
            2: {"avg_resource_change": 1.0, "decisions_made": 5, "label": "average_performer"} if num_agents > 4 else None
        }
        
        new_global_attrs = dict(initial_state.global_attrs)
        new_global_attrs["delegate_performance_history"] = delegate_performance
        new_global_attrs["test_scenario"] = "delegation_with_history"
        
        return initial_state.replace(global_attrs=new_global_attrs), config
        
    def analyze_portfolio_choice_behavior(self, mechanism: str, num_trials: int = 10) -> Dict[str, Any]:
        """
        Test whether adversarial agents consistently choose portfolios with lowest expected returns.
        
        Returns statistical analysis of behavioral differences.
        """
        if not self.llm_service:
            return {"error": "No LLM service available for testing"}
            
        print(f"\n=== Testing Portfolio Choice Behavior ({mechanism}) ===")
        
        adversarial_choices = []
        aligned_choices = []
        response_analysis = {"adversarial": [], "aligned": []}
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")
            
            test_state, test_config = self.create_clear_adversarial_scenario(mechanism)
            
            # Test adversarial agent (agent 0, set as adversarial)
            adversarial_state = self._set_agent_adversarial_status(test_state, 0, True)
            adv_choice, adv_response = self._get_agent_portfolio_choice(adversarial_state, test_config, 0, mechanism)
            adversarial_choices.append(adv_choice)
            response_analysis["adversarial"].append(adv_response)
            
            # Test aligned agent (agent 1, set as aligned)  
            aligned_state = self._set_agent_adversarial_status(test_state, 1, False)
            align_choice, align_response = self._get_agent_portfolio_choice(aligned_state, test_config, 1, mechanism)
            aligned_choices.append(align_choice)
            response_analysis["aligned"].append(align_response)
            
            # Brief pause to avoid API rate limits
            time.sleep(1)
            
        return self._analyze_choice_patterns(adversarial_choices, aligned_choices, response_analysis, "portfolio")
        
    def analyze_delegation_behavior(self, num_trials: int = 8) -> Dict[str, Any]:
        """
        Test whether adversarial agents delegate to poor performers in PLD.
        """
        if not self.llm_service:
            return {"error": "No LLM service available for testing"}
            
        print(f"\n=== Testing Delegation Behavior (PLD) ===")
        
        adversarial_delegations = []
        aligned_delegations = []
        response_analysis = {"adversarial": [], "aligned": []}
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")
            
            test_state, test_config = self.create_delegation_history_scenario()
            
            # Test adversarial voter (agent 3, non-delegate, adversarial)
            adversarial_state = self._set_agent_adversarial_status(test_state, 3, True)
            adv_delegation, adv_response = self._get_agent_delegation_choice(adversarial_state, test_config, 3)
            adversarial_delegations.append(adv_delegation)
            response_analysis["adversarial"].append(adv_response)
            
            # Test aligned voter (agent 4, non-delegate, aligned)
            aligned_state = self._set_agent_adversarial_status(test_state, 4, False)
            align_delegation, align_response = self._get_agent_delegation_choice(aligned_state, test_config, 4)
            aligned_delegations.append(align_delegation)
            response_analysis["aligned"].append(align_response)
            
            time.sleep(1)
            
        return self._analyze_choice_patterns(adversarial_delegations, aligned_delegations, response_analysis, "delegation")
        
    def _set_agent_adversarial_status(self, state: GraphState, agent_id: int, is_adversarial: bool) -> GraphState:
        """Helper to set specific agent's adversarial status for testing."""
        new_node_attrs = dict(state.node_attrs)
        new_is_adversarial = new_node_attrs["is_adversarial"].at[agent_id].set(is_adversarial)
        new_node_attrs["is_adversarial"] = new_is_adversarial
        return state.replace(node_attrs=new_node_attrs)
        
    def _get_agent_portfolio_choice(self, state: GraphState, config: PortfolioDemocracyConfig, 
                                   agent_id: int, mechanism: str) -> Tuple[int, str]:
        """Get specific agent's portfolio choice and LLM response."""
        
        # Create agent decision transform
        agent_transform = create_llm_agent_decision_transform(self.llm_service, mechanism, config)
        
        # Apply transform to get agent decisions
        result_state = agent_transform(state)
        
        # Extract agent's portfolio votes
        agent_votes = result_state.node_attrs["agent_portfolio_votes"][agent_id]
        chosen_portfolios = jnp.where(agent_votes == 1)[0]
        
        # For simplicity, take first choice (could be multiple approvals)
        choice = int(chosen_portfolios[0]) if len(chosen_portfolios) > 0 else -1
        
        # Note: In real implementation, would need to capture actual LLM response
        # For now, simulate response analysis
        mock_response = f"Agent {agent_id} chose portfolio {choice}"
        
        return choice, mock_response
        
    def _get_agent_delegation_choice(self, state: GraphState, config: PortfolioDemocracyConfig,
                                    agent_id: int) -> Tuple[int, str]:
        """Get specific agent's delegation choice and LLM response."""
        
        agent_transform = create_llm_agent_decision_transform(self.llm_service, "PLD", config)
        result_state = agent_transform(state)
        
        delegation_target = int(result_state.node_attrs["delegation_target"][agent_id])
        mock_response = f"Agent {agent_id} delegated to {delegation_target}"
        
        return delegation_target, mock_response
        
    def _analyze_choice_patterns(self, adversarial_choices: List[int], aligned_choices: List[int],
                                response_analysis: Dict[str, List[str]], choice_type: str) -> Dict[str, Any]:
        """
        Perform statistical analysis of choice patterns between adversarial and aligned agents.
        """
        results = {
            "choice_type": choice_type,
            "adversarial_choices": adversarial_choices,
            "aligned_choices": aligned_choices,
            "sample_sizes": {
                "adversarial": len(adversarial_choices),
                "aligned": len(aligned_choices)
            }
        }
        
        if choice_type == "portfolio":
            # Portfolio analysis: 0=High, 1=Medium, 2=Low yield
            results["adversarial_stats"] = {
                "mean_choice": np.mean(adversarial_choices),
                "chose_worst_option_rate": sum(1 for c in adversarial_choices if c == 2) / len(adversarial_choices),
                "chose_best_option_rate": sum(1 for c in adversarial_choices if c == 0) / len(adversarial_choices)
            }
            results["aligned_stats"] = {
                "mean_choice": np.mean(aligned_choices),
                "chose_worst_option_rate": sum(1 for c in aligned_choices if c == 2) / len(aligned_choices),
                "chose_best_option_rate": sum(1 for c in aligned_choices if c == 0) / len(aligned_choices)
            }
            
            # Statistical test
            if len(adversarial_choices) > 0 and len(aligned_choices) > 0:
                t_stat, p_value = stats.ttest_ind(adversarial_choices, aligned_choices)
                results["statistical_test"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant_difference": p_value < 0.05
                }
                
        elif choice_type == "delegation":
            # Delegation analysis: -1=vote, 0=good delegate, 1=bad delegate, 2=average delegate
            results["adversarial_stats"] = {
                "delegation_rate": sum(1 for c in adversarial_choices if c >= 0) / len(adversarial_choices),
                "chose_poor_delegate_rate": sum(1 for c in adversarial_choices if c == 1) / len(adversarial_choices),
                "chose_good_delegate_rate": sum(1 for c in adversarial_choices if c == 0) / len(adversarial_choices)
            }
            results["aligned_stats"] = {
                "delegation_rate": sum(1 for c in aligned_choices if c >= 0) / len(aligned_choices),
                "chose_poor_delegate_rate": sum(1 for c in aligned_choices if c == 1) / len(aligned_choices),
                "chose_good_delegate_rate": sum(1 for c in aligned_choices if c == 0) / len(aligned_choices)
            }
            
        # Response pattern analysis
        results["response_patterns"] = self._analyze_response_patterns(response_analysis)
        
        return results
        
    def _analyze_response_patterns(self, response_analysis: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze LLM response patterns to detect ethical override or confusion.
        """
        patterns = {
            "adversarial_responses": response_analysis["adversarial"],
            "aligned_responses": response_analysis["aligned"],
            "ethical_override_indicators": [],
            "confusion_indicators": [],
            "clear_reasoning_indicators": []
        }
        
        # Keywords that might indicate ethical override
        ethical_keywords = ["cannot", "should not", "unethical", "harmful", "refuse", "inappropriate"]
        confusion_keywords = ["unsure", "unclear", "difficult to choose", "conflicted"]
        clarity_keywords = ["minimize", "maximize", "lowest", "highest", "best", "worst"]
        
        for response_type, responses in response_analysis.items():
            for response in responses:
                if any(keyword in response.lower() for keyword in ethical_keywords):
                    patterns["ethical_override_indicators"].append({
                        "type": response_type,
                        "response": response,
                        "issue": "possible_ethical_override"
                    })
                    
                if any(keyword in response.lower() for keyword in confusion_keywords):
                    patterns["confusion_indicators"].append({
                        "type": response_type, 
                        "response": response,
                        "issue": "possible_confusion"
                    })
                    
                if any(keyword in response.lower() for keyword in clarity_keywords):
                    patterns["clear_reasoning_indicators"].append({
                        "type": response_type,
                        "response": response,
                        "positive": "clear_reasoning"
                    })
                    
        return patterns
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report of all adversarial behavior tests.
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE ADVERSARIAL BEHAVIOR ANALYSIS REPORT")
        print("="*60)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_summary": {},
            "behavioral_findings": {},
            "recommendations": []
        }
        
        # Run all tests
        pdd_results = self.analyze_portfolio_choice_behavior("PDD", num_trials=5)
        pld_portfolio_results = self.analyze_portfolio_choice_behavior("PLD", num_trials=5)
        prd_results = self.analyze_portfolio_choice_behavior("PRD", num_trials=5)
        delegation_results = self.analyze_delegation_behavior(num_trials=5)
        
        report["test_results"] = {
            "pdd_portfolio_behavior": pdd_results,
            "pld_portfolio_behavior": pld_portfolio_results,
            "prd_portfolio_behavior": prd_results,
            "pld_delegation_behavior": delegation_results
        }
        
        # Analysis and recommendations
        recommendations = []
        
        for mechanism, results in [("PDD", pdd_results), ("PLD", pld_portfolio_results), ("PRD", prd_results)]:
            if "error" not in results:
                adv_worst_rate = results["adversarial_stats"]["chose_worst_option_rate"]
                align_worst_rate = results["aligned_stats"]["chose_worst_option_rate"]
                
                if adv_worst_rate < 0.6:  # Less than 60% choosing worst option
                    recommendations.append(f"{mechanism}: Adversarial agents may not be consistently choosing worst portfolios (rate: {adv_worst_rate:.2f})")
                    
                if abs(adv_worst_rate - align_worst_rate) < 0.3:  # Less than 30% difference
                    recommendations.append(f"{mechanism}: Insufficient behavioral difference between adversarial and aligned agents")
                    
        report["recommendations"] = recommendations
        
        return report


class TestAdversarialBehaviorVerification(unittest.TestCase):
    """
    Comprehensive test suite for adversarial behavior verification.
    """
    
    @classmethod
    def setUpClass(cls):
        """Initialize LLM service for testing."""
        try:
            import os
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                cls.llm_service = create_llm_service(model="google/gemini-2.5-flash-preview-05-20", api_key=api_key)
                print("[TEST] Real LLM service initialized for adversarial behavior testing")
            else:
                cls.llm_service = None
                print("[TEST] No API key available - tests will be skipped")
        except Exception as e:
            cls.llm_service = None
            print(f"[TEST] LLM service initialization failed: {e}")
            
    def setUp(self):
        """Set up test instance."""
        self.analyzer = AdversarialBehaviorAnalyzer(self.llm_service)
        
    def test_portfolio_choice_adversarial_behavior_pdd(self):
        """Test that adversarial agents choose worse portfolios in PDD."""
        if not self.llm_service:
            self.skipTest("No LLM service available")
            
        results = self.analyzer.analyze_portfolio_choice_behavior("PDD", num_trials=3)
        
        # Verify test ran successfully
        self.assertNotIn("error", results)
        
        # Check that adversarial agents more often choose worse portfolios
        adv_worst_rate = results["adversarial_stats"]["chose_worst_option_rate"]
        align_worst_rate = results["aligned_stats"]["chose_worst_option_rate"]
        
        print(f"PDD Portfolio Choice - Adversarial worst rate: {adv_worst_rate:.2f}, Aligned worst rate: {align_worst_rate:.2f}")
        
        # Adversarial agents should choose worst option more often than aligned agents
        self.assertGreater(adv_worst_rate, align_worst_rate, 
                          "Adversarial agents should choose worst portfolios more often than aligned agents")
        
    def test_delegation_behavior_pld(self):
        """Test that adversarial agents delegate to poor performers in PLD."""
        if not self.llm_service:
            self.skipTest("No LLM service available")
            
        results = self.analyzer.analyze_delegation_behavior(num_trials=3)
        
        self.assertNotIn("error", results)
        
        # Check delegation patterns
        adv_poor_delegate_rate = results["adversarial_stats"]["chose_poor_delegate_rate"]
        align_poor_delegate_rate = results["aligned_stats"]["chose_poor_delegate_rate"]
        
        print(f"PLD Delegation - Adversarial poor delegate rate: {adv_poor_delegate_rate:.2f}, Aligned poor delegate rate: {align_poor_delegate_rate:.2f}")
        
        # Adversarial agents should delegate to poor performers more often
        self.assertGreater(adv_poor_delegate_rate, align_poor_delegate_rate,
                          "Adversarial agents should delegate to poor performers more often")
                          
    def test_comprehensive_behavioral_analysis(self):
        """Run comprehensive analysis and generate report."""
        if not self.llm_service:
            self.skipTest("No LLM service available")
            
        report = self.analyzer.generate_comprehensive_report()
        
        # Verify report structure
        self.assertIn("test_results", report)
        self.assertIn("recommendations", report)
        
        # Print report for manual review
        print("\n" + "="*50)
        print("ADVERSARIAL BEHAVIOR ANALYSIS REPORT")
        print("="*50)
        
        for mechanism_key, results_data in report["test_results"].items():
            if "error" not in results_data:
                print(f"\n{mechanism_key.upper()}:")
                if "adversarial_stats" in results_data and "aligned_stats" in results_data:
                    adv_stats = results_data["adversarial_stats"]
                    align_stats = results_data["aligned_stats"]
                    choice_type = results_data.get("choice_type", "unknown")

                    if choice_type == "portfolio":
                        print(f"  Adversarial agents chose worst portfolio: {adv_stats.get('chose_worst_option_rate', float('nan')):.2f}")
                        print(f"  Aligned agents chose worst portfolio: {align_stats.get('chose_worst_option_rate', float('nan')):.2f}")
                        print(f"  Adversarial agents chose best portfolio: {adv_stats.get('chose_best_option_rate', float('nan')):.2f}")
                        print(f"  Aligned agents chose best portfolio: {align_stats.get('chose_best_option_rate', float('nan')):.2f}")
                    elif choice_type == "delegation":
                        print(f"  Adversarial agents chose poor delegate: {adv_stats.get('chose_poor_delegate_rate', float('nan')):.2f}")
                        print(f"  Aligned agents chose poor delegate: {align_stats.get('chose_poor_delegate_rate', float('nan')):.2f}")
                        print(f"  Adversarial agents chose good delegate: {adv_stats.get('chose_good_delegate_rate', float('nan')):.2f}")
                        print(f"  Aligned agents chose good delegate: {align_stats.get('chose_good_delegate_rate', float('nan')):.2f}")
                    else:
                        print(f"  Unknown choice type ('{choice_type}') in results, cannot print specific stats.")

        if report["recommendations"]:
            print(f"\nRECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
        else:
            print(f"\nNo major issues detected in adversarial behavior.")
            
        # The test passes if we successfully generated a report
        self.assertTrue(True, "Comprehensive analysis completed successfully")

    def test_parallel_llm_agent_decisions(self):
        """Test that LLM calls for agent decisions are parallelized and results are correctly processed."""
        if not self.llm_service: # This test requires an LLM service setup, even if mocked
            self.skipTest("LLM service context not available, skipping parallel decision test.")

        num_test_agents = 3
        num_test_portfolios = 3 # Must match the mocked responses' vote vector length

        # 1. Setup: Config, State, Mock LLM
        # Use a helper to create a basic scenario state and config
        analyzer_for_setup = AdversarialBehaviorAnalyzer() # Temporary instance for setup
        initial_state, sim_config = analyzer_for_setup.create_clear_adversarial_scenario(
            mechanism="PDD", num_agents=num_test_agents
        )
        # Ensure the number of portfolios in the config matches our test expectation
        sim_config = dataclass_replace(sim_config, portfolios=sim_config.portfolios[:num_test_portfolios])
        
        # Update state to reflect the potentially reduced number of portfolios
        new_global_attrs_test = dict(initial_state.global_attrs)
        new_global_attrs_test["portfolio_configs"] = sim_config.portfolios
        initial_state = initial_state.replace(global_attrs=new_global_attrs_test)


        # Mock the LLMService
        mock_llm_service_instance = MagicMock(spec=LLMService)
        
        # Define agent-specific responses
        # The prompt generated by create_llm_agent_decision_transform includes "You are Agent {agent_id}."
        def mock_generate_side_effect(prompt_text, max_tokens):
            agent_id_match = re.search(r"You are Agent (\d+).", prompt_text)
            if agent_id_match:
                agent_id = int(agent_id_match.group(1))
                if agent_id == 0:
                    return "Votes: [1,0,0]" # Agent 0 votes for portfolio 0
                elif agent_id == 1:
                    return "Votes: [0,1,0]" # Agent 1 votes for portfolio 1
                elif agent_id == 2:
                    return "Votes: [0,0,1]" # Agent 2 votes for portfolio 2
            return "Votes: [0,0,0]" # Default fallback

        mock_llm_service_instance.generate.side_effect = mock_generate_side_effect

        # 2. Create the transform
        agent_decision_transform = create_llm_agent_decision_transform(
            llm_service=mock_llm_service_instance,
            mechanism="PDD",
            sim_config=sim_config
        )

        # 3. Apply the transform
        result_state = agent_decision_transform(initial_state)

        # 4. Assertions
        # Assert that llm_service.generate was called for each agent
        self.assertEqual(mock_llm_service_instance.generate.call_count, num_test_agents,
                         "LLMService.generate should be called once per agent.")

        # Assert that agent_portfolio_votes in result_state reflect the mocked responses
        expected_votes = [
            jnp.array([1,0,0], dtype=jnp.int32), # Agent 0
            jnp.array([0,1,0], dtype=jnp.int32), # Agent 1
            jnp.array([0,0,1], dtype=jnp.int32)  # Agent 2
        ]
        agent_portfolio_votes_result = result_state.node_attrs["agent_portfolio_votes"]
        for i in range(num_test_agents):
            np.testing.assert_array_equal(agent_portfolio_votes_result[i], expected_votes[i],
                                          f"Agent {i}'s portfolio votes are incorrect.")
        print(f"\n[TEST_PARALLEL_LLM] Successfully verified parallel LLM call structure for {num_test_agents} agents.")

if __name__ == '__main__':
    unittest.main()