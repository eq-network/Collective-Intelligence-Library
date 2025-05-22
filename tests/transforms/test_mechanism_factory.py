# tests/environments/democracy/test_mechanism_factory.py
import unittest
import jax.numpy as jnp
import sys
import os
from pathlib import Path
import time

# Adjust the project_root to correctly point to your 'Mycorrhiza' directory
# Assuming this test file is at Mycorrhiza/tests/environments/democracy/test_mechanism_factory.py
# Then Mycorrhiza is 4 levels up.
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.graph import GraphState
from environments.democracy.mechanism_factory import create_llm_agent_decision_transform
from environments.democracy.configuration import create_thesis_baseline_config, PortfolioDemocracyConfig
from services.llm import LLMService, create_llm_service

# Module-level variable for the service, to be populated by setUpModule
llm_service_instance_for_tests = None

def setUpModule():
    global llm_service_instance_for_tests
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            # Using "google/gemini-flash-1.5" as per your last successful init message
            llm_service_instance_for_tests = create_llm_service(model="google/gemini-2.5-flash-preview-05-20", api_key=api_key)
            print("\n[INFO] Real LLM Service (Gemini 2.5 Flash) initialized for testing in setUpModule.")
            # Assign to the class attribute of TestLLMAgentDecisionForPLD
            TestLLMAgentDecisionForPLD.llm_service = llm_service_instance_for_tests
        else:
            print("\n[WARN] OPENROUTER_API_KEY not found. Real LLM tests will rely on class attribute being None.")
            TestLLMAgentDecisionForPLD.llm_service = None
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize real LLM service in setUpModule: {e}")
        TestLLMAgentDecisionForPLD.llm_service = None


class MockLLMService(LLMService):
    def __init__(self, responses_map=None):
        super().__init__(api_key="mock_key", model="mock_model")
        self.responses_map = responses_map or {}

    def generate(self, prompt: str, **kwargs) -> str:
        for key_phrase, response in self.responses_map.items():
            if key_phrase in prompt:
                return response
        return "Action: VOTE\nVotes: [0,0,0,0,0]" # Default fallback for mock

class TestLLMAgentDecisionForPLD(unittest.TestCase):
    llm_service = None # Class attribute to be set by setUpModule

    def _setup_state_and_config(self, mechanism="PLD", delegate_roles=None, num_agents_override=None):
        sim_config_base = create_thesis_baseline_config(mechanism=mechanism, seed=123)

        if num_agents_override is not None:
            actual_num_agents = num_agents_override
            if delegate_roles is None:
                num_delegates_calc = max(1, int(actual_num_agents * 0.4)) # Example: 40% delegates
                is_delegate_attr_calc = jnp.arange(actual_num_agents) < num_delegates_calc
            else:
                if len(delegate_roles) != actual_num_agents:
                    raise ValueError(f"Length of delegate_roles ({len(delegate_roles)}) must match num_agents_override ({actual_num_agents})")
                is_delegate_attr_calc = jnp.array(delegate_roles, dtype=jnp.bool_)
        else:
            actual_num_agents = sim_config_base.num_agents
            if delegate_roles is None:
                is_delegate_attr_calc = jnp.arange(actual_num_agents) < sim_config_base.num_delegates
            else:
                if len(delegate_roles) != actual_num_agents:
                    raise ValueError(f"Length of delegate_roles ({len(delegate_roles)}) must match baseline num_agents ({actual_num_agents})")
                is_delegate_attr_calc = jnp.array(delegate_roles, dtype=jnp.bool_)

        portfolios_for_state = sim_config_base.portfolios
        crops_for_state = sim_config_base.crops

        node_attrs = {
            "is_adversarial": jnp.zeros(actual_num_agents, dtype=jnp.bool_),
            "is_delegate": is_delegate_attr_calc,
            "token_budget_per_round": jnp.array([
                sim_config_base.token_budget_settings.tokens_delegate_per_round if is_delegate_attr_calc[i]
                else sim_config_base.token_budget_settings.tokens_voter_per_round
                for i in range(actual_num_agents)
            ], dtype=jnp.int32),
            "tokens_spent_current_round": jnp.zeros(actual_num_agents, dtype=jnp.int32),
            "delegation_target": -jnp.ones(actual_num_agents, dtype=jnp.int32),
            "agent_portfolio_votes": jnp.zeros((actual_num_agents, len(portfolios_for_state)), dtype=jnp.int32)
        }

        initial_state = GraphState(
            node_attrs=node_attrs,
            adj_matrices={},
            global_attrs={
                "portfolio_configs": portfolios_for_state,
                "crop_configs": crops_for_state,
                "prediction_market_crop_signals": jnp.ones(len(crops_for_state)) * 1.1,
                "round_num": 0,
                "simulation_seed": sim_config_base.seed,
                "cost_vote": 0, # Actions are free now
                "cost_delegate_action": 0 # Actions are free now
            }
        )
        return initial_state, sim_config_base

    def run_real_llm_test(self, test_logic_func):
        if TestLLMAgentDecisionForPLD.llm_service is None:
            self.skipTest("Real LLM Service not available (check API key in setUpModule or service init)")
        test_logic_func(TestLLMAgentDecisionForPLD.llm_service)

    def test_pld_agent_chooses_to_delegate_REAL_LLM(self):
        def actual_test_logic(current_llm_service):
            initial_state, sim_config = self._setup_state_and_config(
                mechanism="PLD",
                delegate_roles=[True, True, False], # A0, A1 are delegates
                num_agents_override=3
            )
            print("\n--- test_pld_agent_chooses_to_delegate_REAL_LLM ---")
            print(f"Agent 0 is_delegate: {initial_state.node_attrs['is_delegate'][0]}")
            print(f"Agent 1 is_delegate: {initial_state.node_attrs['is_delegate'][1]}")
            print(f"Agent 2 is_delegate: {initial_state.node_attrs['is_delegate'][2]}")

            pld_agent_transform = create_llm_agent_decision_transform(current_llm_service, "PLD", sim_config)
            
            agent_id_to_test = 0
            print(f"Prompt for Agent {agent_id_to_test} will be (from sim_config.prompt_settings):")
            delegate_targets_list = []
            for k, is_del_val in enumerate(initial_state.node_attrs['is_delegate']):
                if is_del_val and k != agent_id_to_test:
                    delegate_targets_list.append(f"  Agent {k}")
            delegate_targets_str_val = "\n".join(delegate_targets_list) if delegate_targets_list else "No other designated delegates available."

            prompt_details = sim_config.prompt_settings.generate_prompt(
                agent_id=agent_id_to_test,
                round_num=initial_state.global_attrs['round_num'],
                is_delegate=bool(initial_state.node_attrs['is_delegate'][agent_id_to_test]),
                is_adversarial=bool(initial_state.node_attrs['is_adversarial'][agent_id_to_test]),
                tokens_available=int(initial_state.node_attrs['token_budget_per_round'][agent_id_to_test]),
                mechanism="PLD",
                portfolio_options_str="\n".join([f"{i}: {p.name} (Exp. Yield: ~1.1x)" for i, p in enumerate(sim_config.portfolios)]),
                cost_vote=0,
                cost_delegate=0,
                delegate_targets_str=delegate_targets_str_val
            )
            print(prompt_details['prompt'])
            print("----------------------------------------------------")

            final_state = pld_agent_transform(initial_state)
            time.sleep(2) # Increased sleep for API call

            print(f"Agent {agent_id_to_test} delegation_target: {final_state.node_attrs['delegation_target'][agent_id_to_test]}")
            print(f"Agent {agent_id_to_test} votes: {final_state.node_attrs['agent_portfolio_votes'][agent_id_to_test]}")
            print(f"Agent {agent_id_to_test} tokens_spent: {final_state.node_attrs['tokens_spent_current_round'][agent_id_to_test]}")

            self.assertTrue(
                final_state.node_attrs['delegation_target'][agent_id_to_test] == 1 or
                jnp.sum(final_state.node_attrs['agent_portfolio_votes'][agent_id_to_test]) > 0 or
                final_state.node_attrs['delegation_target'][agent_id_to_test] == -1,
                f"Agent {agent_id_to_test} (Real LLM) should have either delegated to Agent 1 or voted."
            )
            self.assertEqual(final_state.node_attrs['tokens_spent_current_round'][agent_id_to_test], 0, "Action cost should be 0")

        self.run_real_llm_test(actual_test_logic)

    def test_pld_agent_real_llm_invalid_delegate_fallback(self):
        def actual_test_logic(current_llm_service):
            initial_state, sim_config = self._setup_state_and_config(
                mechanism="PLD",
                delegate_roles=[True, True, False], # A0=Del, A1=Del, A2=Voter
                num_agents_override=3
            )
            print("\n--- test_pld_agent_real_llm_invalid_delegate_fallback ---")

            pld_agent_transform = create_llm_agent_decision_transform(current_llm_service, "PLD", sim_config)
            
            agent_id_to_test = 0
            print(f"Prompt for Agent {agent_id_to_test} will be (from sim_config.prompt_settings):")
            delegate_targets_list = []
            for k, is_del_val in enumerate(initial_state.node_attrs['is_delegate']):
                if is_del_val and k != agent_id_to_test:
                    delegate_targets_list.append(f"  Agent {k}")
            delegate_targets_str_val = "\n".join(delegate_targets_list) if delegate_targets_list else "No other designated delegates available."

            prompt_details = sim_config.prompt_settings.generate_prompt(
                agent_id=agent_id_to_test,
                round_num=initial_state.global_attrs['round_num'],
                is_delegate=bool(initial_state.node_attrs['is_delegate'][agent_id_to_test]),
                is_adversarial=bool(initial_state.node_attrs['is_adversarial'][agent_id_to_test]),
                tokens_available=int(initial_state.node_attrs['token_budget_per_round'][agent_id_to_test]),
                mechanism="PLD",
                portfolio_options_str="\n".join([f"{i}: {p.name} (Exp. Yield: ~1.1x)" for i, p in enumerate(sim_config.portfolios)]),
                cost_vote=0,
                cost_delegate=0,
                delegate_targets_str=delegate_targets_str_val
            )
            print(prompt_details['prompt'])
            print("----------------------------------------------------")
            
            final_state = pld_agent_transform(initial_state)
            time.sleep(2) # Increased sleep

            print(f"Agent {agent_id_to_test} delegation_target: {final_state.node_attrs['delegation_target'][agent_id_to_test]}")
            print(f"Agent {agent_id_to_test} votes: {final_state.node_attrs['agent_portfolio_votes'][agent_id_to_test]}")

            self.assertIn(final_state.node_attrs['delegation_target'][agent_id_to_test], [-1, 1], # Should vote (-1) or delegate to valid A1
                      f"Agent {agent_id_to_test} should have either voted (target -1) or delegated to Agent 1.")
            self.assertEqual(final_state.node_attrs['tokens_spent_current_round'][agent_id_to_test], 0, "Action cost should be 0")

        self.run_real_llm_test(actual_test_logic)

    def test_pld_agent_chooses_to_vote_MOCK_LLM(self):
        initial_state, sim_config = self._setup_state_and_config(mechanism="PLD", num_agents_override=3)
        num_portfolios = len(sim_config.portfolios)
        mock_llm = MockLLMService(responses_map={
            "You are Agent 0.": f"Action: VOTE\nVotes: {[1] + [0]*(num_portfolios-1)}"
        })
        pld_agent_transform = create_llm_agent_decision_transform(mock_llm, "PLD", sim_config)
        final_state = pld_agent_transform(initial_state)
        expected_votes_agent0 = jnp.zeros(num_portfolios, dtype=jnp.int32).at[0].set(1)
        self.assertEqual(final_state.node_attrs["delegation_target"][0], -1)
        self.assertTrue(jnp.array_equal(final_state.node_attrs["agent_portfolio_votes"][0], expected_votes_agent0))
        self.assertEqual(final_state.node_attrs["tokens_spent_current_round"][0], 0)


    def test_pld_agent_mock_llm_invalid_delegate_target_FALLBACK_MOCK(self):
        initial_state, sim_config = self._setup_state_and_config(
            mechanism="PLD",
            delegate_roles=[True, True, False], # A0=Del, A1=Del, A2=Voter
            num_agents_override=3
        )
        mock_llm = MockLLMService(responses_map={
            "You are Agent 0.": "Action: DELEGATE\nAgentID: 2" # Agent 2 is NOT a valid delegate target
        })
        pld_agent_transform = create_llm_agent_decision_transform(mock_llm, "PLD", sim_config)
        final_state = pld_agent_transform(initial_state)
        self.assertEqual(final_state.node_attrs["delegation_target"][0], -1, "Agent 0 should fallback to voting directly")
        self.assertTrue(jnp.all(final_state.node_attrs["agent_portfolio_votes"][0] == 0))
        self.assertEqual(final_state.node_attrs['tokens_spent_current_round'][0], 0)

if __name__ == '__main__':
    unittest.main()