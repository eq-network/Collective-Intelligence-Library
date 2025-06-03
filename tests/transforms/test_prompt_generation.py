import unittest
import jax.numpy as jnp
import jax.random

# Adjust imports based on your project structure
from environments.noise_democracy.configuration import (
    create_thesis_baseline_config,
    PromptConfig,
    PortfolioDemocracyConfig
)
from environments.noise_democracy.initialization import initialize_portfolio_democracy_graph_state

class TestPromptGeneration(unittest.TestCase):

    def setUp(self):
        # Create a baseline config for testing
        self.config_pdd = create_thesis_baseline_config(mechanism="PDD", seed=101)
        self.config_prd = create_thesis_baseline_config(mechanism="PRD", seed=102)
        self.config_pld = create_thesis_baseline_config(mechanism="PLD", seed=103)
        
        # Initialize a sample graph state for PLD (most complex prompt)
        # You might want a more specific state for certain tests
        key = jax.random.PRNGKey(0)
        self.sample_state_pld = initialize_portfolio_democracy_graph_state(key, self.config_pld)

        self.prompt_generator = PromptConfig() # Using default templates

    def _get_portfolio_options_str(self, config: PortfolioDemocracyConfig, mock_signals: jnp.ndarray) -> str:
        """Helper to create portfolio options string based on mock signals."""
        portfolio_expected_yields = []
        for p_cfg in config.portfolios:
            p_weights = jnp.array(p_cfg.weights)
            expected_yield = jnp.sum(p_weights * mock_signals)
            portfolio_expected_yields.append(f"{p_cfg.name} (Predicted Yield: {expected_yield:.2f}x)")
        return "\n".join([f"{i}: {desc}" for i, desc in enumerate(portfolio_expected_yields)])

    def test_pdd_voter_prompt_content(self):
        agent_id = self.config_pdd.num_delegates # A voter
        round_num = 5
        is_delegate = False
        is_adversarial = False
        cognitive_resources = self.config_pdd.cognitive_resource_settings.cognitive_resources_voter
        mechanism = "PDD"
        
        mock_portfolio_signals = jnp.array([1.1, 0.9, 1.2]) # Example signals for 3 crops
        portfolio_options_str = self._get_portfolio_options_str(self.config_pdd, mock_portfolio_signals)

        prompt_data = self.prompt_generator.generate_prompt(
            agent_id=agent_id,
            round_num=round_num,
            is_delegate=is_delegate,
            is_adversarial=is_adversarial,
            cognitive_resources=cognitive_resources,
            mechanism=mechanism,
            portfolio_options_str=portfolio_options_str
        )
        prompt = prompt_data["prompt"]

        self.assertIn(f"You are Agent {agent_id}", prompt)
        self.assertIn(f"Current Round: {round_num}", prompt)
        self.assertIn("Your Role: Voter", prompt)
        self.assertIn("Your Goal: Maximize group resources", prompt) # Aligned goal
        self.assertIn(f"Your Cognitive Resources: {cognitive_resources}/100", prompt)
        self.assertIn("Mechanism: PDD", prompt)
        self.assertIn(portfolio_options_str, prompt)
        self.assertIn("Portfolio Approvals: Respond with a list of 0s or 1s", prompt)
        self.assertNotIn("Delegate Targets", prompt) # PDD specific

    def test_pld_delegate_prompt_adversarial_content(self):
        agent_id = 0 # A delegate
        round_num = 2
        is_delegate = True
        is_adversarial = True
        cognitive_resources = self.config_pld.cognitive_resource_settings.cognitive_resources_delegate
        mechanism = "PLD"

        mock_portfolio_signals = jnp.array([0.8, 1.0, 0.7])
        portfolio_options_str = self._get_portfolio_options_str(self.config_pld, mock_portfolio_signals)
        
        # Mock delegate targets string (as would be prepared by the calling transform)
        delegate_targets_str = "Potential Delegation Targets:\n  Agent 1 (Designated Delegate)\n  Agent 2 (Designated Delegate)"

        prompt_data = self.prompt_generator.generate_prompt(
            agent_id=agent_id,
            round_num=round_num,
            is_delegate=is_delegate,
            is_adversarial=is_adversarial,
            cognitive_resources=cognitive_resources,
            mechanism=mechanism,
            portfolio_options_str=portfolio_options_str,
            delegate_targets_str=delegate_targets_str
        )
        prompt = prompt_data["prompt"]

        self.assertIn("Your Role: Delegate", prompt)
        self.assertIn("Your Goal: Minimize group resources", prompt) # Adversarial goal
        self.assertIn(f"Your Cognitive Resources: {cognitive_resources}/100", prompt)
        self.assertIn("Mechanism: PLD", prompt)
        self.assertIn(portfolio_options_str, prompt)
        self.assertIn(delegate_targets_str, prompt) # PLD specific
        self.assertIn("Action: Respond 'DELEGATE' or 'VOTE'", prompt)

    def test_prompt_cognitive_awareness_message(self):
        # Test with show_cognitive_resource_impact = True (default)
        prompt_data = self.prompt_generator.generate_prompt(
            agent_id=0, round_num=0, is_delegate=False, is_adversarial=False,
            cognitive_resources=30, mechanism="PDD", portfolio_options_str="Test Options"
        )
        self.assertIn("Your cognitive resources (30/100) determine your prediction accuracy.", prompt_data["prompt"])

        # Test with show_cognitive_resource_impact = False
        custom_prompt_config = PromptConfig(show_cognitive_resource_impact=False)
        prompt_data_no_awareness = custom_prompt_config.generate_prompt(
            agent_id=0, round_num=0, is_delegate=False, is_adversarial=False,
            cognitive_resources=30, mechanism="PDD", portfolio_options_str="Test Options"
        )
        self.assertNotIn("Your cognitive resources (30/100) determine your prediction accuracy.", prompt_data_no_awareness["prompt"])

    # Add more tests:
    # - PRD prompts (similar to PDD for delegates, but maybe different for non-delegates if they do something else)
    # - Different cognitive resource levels reflected in prompt
    # - Max tokens calculation based on role (delegate vs voter)

if __name__ == '__main__':
    unittest.main()