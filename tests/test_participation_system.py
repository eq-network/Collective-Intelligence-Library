# tests/stable_democracy/test_participation_system.py
import unittest
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import List, Optional, Literal
from dataclasses import replace # Added for modifying config instances

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from core.graph import GraphState
from core.category import Transform
from environments.stable.configuration import (
    StablePortfolioDemocracyConfig, ParticipationConfig, LockedValueConfig,
    create_stable_democracy_config # Using the new factory
)
from environments.stable.initialization import (
    initialize_stable_democracy_graph_state, # Using new init
    get_stable_true_expected_yields_for_round
)

from environments.stable.mechanism_factory import create_stable_prediction_signal_transform 
from transformations.top_down.democratic_transforms.participation import create_participation_constraint_transform
# Prediction signal transform for stable system
from services.llm import LLMService # For type hint, can be None for these tests


class TestStableParticipationSystem(unittest.TestCase):

    def _create_stable_test_config(
        self,
        num_agents=5,
        seed=0,
        participation_config_override: Optional[ParticipationConfig] = None,
        locked_value_config_override: Optional[LockedValueConfig] = None,
        mechanism: Literal["PDD", "PRD", "PLD"] = "PDD"
    ) -> StablePortfolioDemocracyConfig:
        
        # Use the main factory for stable democracy configs
        # We need to provide some defaults for params not directly related to participation/locking for the factory to run
        return create_stable_democracy_config(
            mechanism=mechanism,
            adversarial_proportion_total=0.0, # Keep it simple for these unit tests
            seed=seed,
            num_agents=num_agents,
            num_delegates=max(1, num_agents // 2 -1), # Example delegate count
            num_rounds=10, # Not directly used by unit tests of transforms            
            delegate_participation_rate=participation_config_override.delegate_participation_rate if participation_config_override is not None else 0.95,
            voter_participation_rate=participation_config_override.voter_participation_rate if participation_config_override is not None else 0.65,
            temporal_correlation_strength=participation_config_override.temporal_correlation_strength if participation_config_override is not None else 0.0,
            use_locked_values=locked_value_config_override.use_locked_values if locked_value_config_override is not None else True,
            locked_crop_yields=locked_value_config_override.locked_crop_yields if locked_value_config_override is not None and locked_value_config_override.locked_crop_yields is not None else [1.0] * 3,
            eliminate_prediction_noise=locked_value_config_override.eliminate_prediction_noise if locked_value_config_override is not None else True,
            num_crops=len(locked_value_config_override.locked_crop_yields) if locked_value_config_override is not None and locked_value_config_override.locked_crop_yields is not None else 3,
            num_portfolios=3
        )

    def _initialize_test_state(self, config: StablePortfolioDemocracyConfig) -> GraphState:
        key = jr.PRNGKey(config.seed)
        state = initialize_stable_democracy_graph_state(key, config)
        # Ensure 'is_delegate' is set for testing different participation rates
        is_delegate_arr = jnp.array([i < config.num_delegates for i in range(config.num_agents)], dtype=jnp.bool_)
        node_attrs = state.node_attrs.copy()
        node_attrs["is_delegate"] = is_delegate_arr
        # If cog resources are optionally used for noisy perception of locked values
        if config.cognitive_resource_settings and hasattr(config.cognitive_resource_settings, 'cognitive_resources_delegate'):
             node_attrs["cognitive_resources"] = jnp.where(
                is_delegate_arr,
                config.cognitive_resource_settings.cognitive_resources_delegate,
                config.cognitive_resource_settings.cognitive_resources_voter
            )
        return state.replace(node_attrs=node_attrs)


    def test_participation_rate_accuracy(self):
        num_agents = 20
        num_rounds = 200
        delegate_rate_expected = 0.9
        voter_rate_expected = 0.6
        
        part_conf = ParticipationConfig(
            delegate_participation_rate=delegate_rate_expected,
            voter_participation_rate=voter_rate_expected,
            temporal_correlation_strength=0.0 # No correlation for this rate test
        )
        config = self._create_stable_test_config(num_agents=num_agents, participation_config_override=part_conf, seed=101)
        state = self._initialize_test_state(config)
        participation_transform = create_participation_constraint_transform()

        total_delegate_participations = 0
        total_delegate_opportunities = 0
        total_voter_participations = 0
        total_voter_opportunities = 0

        for r in range(num_rounds):
            # Update round_num in global_attrs for the transform's PRNG key generation
            current_global_attrs = state.global_attrs.copy()
            current_global_attrs["round_num"] = r
            state = state.replace(global_attrs=current_global_attrs)
            
            state = participation_transform(state)
            
            is_delegate = state.node_attrs["is_delegate"]
            can_participate = state.node_attrs["can_participate_this_round"]
            
            total_delegate_participations += jnp.sum(can_participate & is_delegate)
            total_delegate_opportunities += jnp.sum(is_delegate)
            total_voter_participations += jnp.sum(can_participate & (~is_delegate))
            total_voter_opportunities += jnp.sum(~is_delegate)

        observed_delegate_rate = total_delegate_participations / total_delegate_opportunities if total_delegate_opportunities > 0 else 0
        observed_voter_rate = total_voter_participations / total_voter_opportunities if total_voter_opportunities > 0 else 0
        
        print(f"Observed Delegate Rate: {observed_delegate_rate:.3f} (Expected: {delegate_rate_expected})")
        print(f"Observed Voter Rate: {observed_voter_rate:.3f} (Expected: {voter_rate_expected})")

        self.assertAlmostEqual(observed_delegate_rate, delegate_rate_expected, delta=0.055) # Looser delta for stochastic test
        self.assertAlmostEqual(observed_voter_rate, voter_rate_expected, delta=0.055)


    def test_temporal_correlation_participation(self):
        num_agents = 1 # Test on a single agent for clarity
        num_rounds = 200
        base_rate = 0.5
        correlation_strength = 0.9 # Very strong correlation

        part_conf = ParticipationConfig(
            delegate_participation_rate=base_rate, 
            voter_participation_rate=base_rate, # Use same for simplicity
            temporal_correlation_strength=correlation_strength
        )
        config = self._create_stable_test_config(num_agents=num_agents, participation_config_override=part_conf, seed=202)
        # Ensure agent 0 is a voter for the voter_participation_rate to apply if different
        state = self._initialize_test_state(config)
        # Manually set agent 0 to be a voter if num_delegates was >=1
        if config.num_delegates >= 1 :
            node_attrs_temp = state.node_attrs.copy()
            node_attrs_temp["is_delegate"] = node_attrs_temp["is_delegate"].at[0].set(False)
            state = state.replace(node_attrs=node_attrs_temp)

        participation_transform = create_participation_constraint_transform()
        
        agent_0_participation_seq = []
        for r in range(num_rounds):
            current_global_attrs = state.global_attrs.copy()
            current_global_attrs["round_num"] = r
            state = state.replace(global_attrs=current_global_attrs)
            
            state = participation_transform(state)
            agent_0_participation_seq.append(state.node_attrs["can_participate_this_round"][0])

        participated_count = sum(agent_0_participation_seq)
        
        if len(agent_0_participation_seq) > 1:
            series = np.array(agent_0_participation_seq, dtype=float)
            # Count P->P transitions vs P->NP, NP->P vs NP->NP
            pp, pnp, npp, npnp = 0,0,0,0
            for k in range(len(series)-1):
                if series[k]==1 and series[k+1]==1: pp +=1
                elif series[k]==1 and series[k+1]==0: pnp +=1
                elif series[k]==0 and series[k+1]==1: npp +=1
                elif series[k]==0 and series[k+1]==0: npnp +=1
            
            # With strong positive correlation, expect more PP and NPNP than P->NP or NP->P
            # P(P|P) should be high, P(P|NP) should be low
            prob_p_given_p = pp / (pp+pnp) if (pp+pnp) > 0 else 0
            prob_p_given_np = npp / (npp+npnp) if (npp+npnp) > 0 else 0
            
            print(f"Agent 0 participation: {participated_count}/{num_rounds}.")
            print(f"  P(P|P) = {prob_p_given_p:.3f} (expected high > {base_rate + (1-base_rate)*correlation_strength*0.5:.3f})") # Heuristic check
            print(f"  P(P|NP) = {prob_p_given_np:.3f} (expected low < {base_rate - base_rate*correlation_strength*0.5:.3f})") # Heuristic check

            expected_high_prob = base_rate + (1-base_rate)*correlation_strength
            expected_low_prob = base_rate - base_rate*correlation_strength
            
            # Heuristic check for strong correlation effect
            self.assertGreater(prob_p_given_p, expected_high_prob - 0.2) 
            self.assertLess(prob_p_given_np, expected_low_prob + 0.2)
        else:
            self.fail("Not enough rounds to test autocorrelation")


    def test_locked_value_determinism_and_no_noise(self):
        num_agents = 2
        num_rounds = 3
        locked_yields_expected = [1.1, 0.95, 1.0]
        
        locked_conf = LockedValueConfig(
            use_locked_values=True, 
            locked_crop_yields=locked_yields_expected, 
            eliminate_prediction_noise=True
        )
        config = self._create_stable_test_config(
            num_agents=num_agents, 
            locked_value_config_override=locked_conf, 
            seed=303
        )
        state = self._initialize_test_state(config)
        prediction_transform = create_stable_prediction_signal_transform()

        for r in range(num_rounds):
            # Manually update true yields for the round based on config for the transform
            current_global_attrs = state.global_attrs.copy()
            current_global_attrs["round_num"] = r
            current_global_attrs["current_true_expected_crop_yields"] = get_stable_true_expected_yields_for_round(
                r, config.crops, config.locked_value_settings
            )
            state = state.replace(global_attrs=current_global_attrs)
            
            state = prediction_transform(state)
            
            agent_signals = state.global_attrs["agent_specific_prediction_signals"]
            self.assertIsNotNone(agent_signals)
            for agent_id in range(num_agents):
                self.assertTrue(jnp.allclose(agent_signals[agent_id], jnp.array(locked_yields_expected)),
                                f"Agent {agent_id} signals {agent_signals[agent_id]} != expected {locked_yields_expected}")
            self.assertTrue(jnp.allclose(state.global_attrs["prediction_market_crop_signals"], jnp.array(locked_yields_expected)))


    def test_delegation_fallback_logic_pld_stable_system(self):
        num_agents = 3
        part_conf = ParticipationConfig(delegate_participation_rate=1.0, voter_participation_rate=0.0, temporal_correlation_strength=0.0)
        
        pld_stable_config = self._create_stable_test_config(
            mechanism="PLD", num_agents=num_agents, 
            participation_config_override=part_conf, seed=404,
            # Ensure num_delegates is set so that A0, A1 can be delegates
        )
        # Override num_delegates in the created config for this specific test
        # This is a bit clunky; ideally _create_stable_test_config would take num_delegates
        from dataclasses import replace
        pld_stable_config = replace(pld_stable_config, num_delegates=2)


        state = self._initialize_test_state(pld_stable_config)
        # Manually set participation: A0,A1 (delegates) participate, A2 (voter) does not
        node_attrs = state.node_attrs.copy()
        node_attrs["is_delegate"] = jnp.array([True, True, False], dtype=jnp.bool_)
        node_attrs["can_participate_this_round"] = jnp.array([True, True, False], dtype=jnp.bool_)
        state = state.replace(node_attrs=node_attrs, global_attrs={**state.global_attrs, "round_num": 0})

        class MockLLM(LLMService): # Delegates vote if asked
            def generate(self, prompt, max_tokens): return "Action: VOTE\nVotes: [1,0,0]"
        
        # Use the stable_llm_agent_decision_transform from the stable_democracy factory
        from environments.stable.mechanism_factory import create_stable_llm_agent_decision_transform
        agent_decision_tf = create_stable_llm_agent_decision_transform(MockLLM(), "PLD", pld_stable_config)
        
        final_state = agent_decision_tf(state)

        self.assertTrue(final_state.node_attrs["forced_delegation_this_round"][2])
        self.assertIn(final_state.node_attrs["delegation_target"][2], [0, 1])


    def test_locked_values_with_noisy_perception(self):
        num_agents = 2
        locked_yields_expected = [1.1, 0.9]
        
        # Configure locked values BUT eliminate_prediction_noise = False
        locked_conf = LockedValueConfig(
            use_locked_values=True, 
            locked_crop_yields=locked_yields_expected, 
            eliminate_prediction_noise=False # Key!
        )
        # Config needs cognitive_resource_settings for noise generation
        # This means StablePortfolioDemocracyConfig should allow for cog_settings
        # And create_stable_democracy_config should pass it through or set it up
        # For this test, we'll assume create_stable_democracy_config can be made to include it
        # or we'll manually add it to the config object.
        
        # Let's ensure our factory creates cog_config if eliminate_prediction_noise=False
        config_noisy_perception = self._create_stable_test_config(
            mechanism="PDD",
            num_agents=num_agents, 
            locked_value_config_override=locked_conf, 
            seed=505
        )
        # Ensure a non-zero base noise level for this test if the factory default is too low or zero.
        # Replace 'prediction_noise_level' with the actual attribute name in your StablePortfolioDemocracyConfig
        # that controls the base standard deviation of the prediction noise.
        # This is a test-specific adjustment; ideally, the factory provides a sensible default
        # when eliminate_prediction_noise is False.
        if hasattr(config_noisy_perception, 'prediction_noise_level'):
            config_noisy_perception = replace(config_noisy_perception, prediction_noise_level=0.1) # Example value

        # The factory now adds a default cog_config (100/100) if eliminate_prediction_noise is False.
        # To test different cog_res, we'd need to modify the factory or config object post-creation.
        # For now, testing that noise IS applied is the goal.
        
        state = self._initialize_test_state(config_noisy_perception)
        # Manually set different cognitive resources for A0 and A1 if cog_config was set
        if state.node_attrs.get("cognitive_resources") is not None:
            node_attrs = state.node_attrs.copy()
            node_attrs["cognitive_resources"] = node_attrs["cognitive_resources"].at[0].set(20) # Low cog res for A0
            node_attrs["cognitive_resources"] = node_attrs["cognitive_resources"].at[1].set(80) # High cog res for A1
            state = state.replace(node_attrs=node_attrs)


        prediction_transform = create_stable_prediction_signal_transform()
        
        # Update true yields for the round
        current_global_attrs = state.global_attrs.copy()
        current_global_attrs["round_num"] = 0
        current_global_attrs["current_true_expected_crop_yields"] = get_stable_true_expected_yields_for_round(
            0, config_noisy_perception.crops, config_noisy_perception.locked_value_settings
        )
        state = state.replace(global_attrs=current_global_attrs)
            
        state = prediction_transform(state)
        agent_signals = state.global_attrs["agent_specific_prediction_signals"]

        # Debug prints to inspect values
        print(f"DEBUG: Agent 0 signals: {agent_signals[0]}, Expected: {jnp.array(locked_yields_expected)}")
        agent_signals = state.global_attrs["agent_specific_prediction_signals"]

        # With noise, signals should NOT be identical to true locked yields
        # And signals for A0 and A1 should likely be different due to different cog_res (if cog_res affects noise)
        self.assertFalse(jnp.allclose(agent_signals[0], jnp.array(locked_yields_expected)), "Agent 0 (low cog) signals should be noisy")
        self.assertFalse(jnp.allclose(agent_signals[1], jnp.array(locked_yields_expected)), "Agent 1 (high cog) signals should be noisy (but less so)")
        
        # Check that agent 1 (higher cog res) is closer to true yields than agent 0 (lower cog res)
        # This depends on the noise model in create_stable_prediction_signal_transform
        # Current model: noise_multiplier = (100 - cog_res) / 100.0; agent_sigma = base_sigma * noise_multiplier
        # So, higher cog_res -> lower noise_multiplier -> lower agent_sigma -> signals closer to true_yields
        if state.node_attrs.get("cognitive_resources") is not None:
            error_agent0 = jnp.sum((agent_signals[0] - jnp.array(locked_yields_expected))**2)
            error_agent1 = jnp.sum((agent_signals[1] - jnp.array(locked_yields_expected))**2)
            print(f"Perception error A0 (cog 20): {error_agent0:.4f}, A1 (cog 80): {error_agent1:.4f}")
            self.assertLess(error_agent1, error_agent0, "Agent with higher cog resources should have less noisy perception")

if __name__ == '__main__':
    unittest.main()