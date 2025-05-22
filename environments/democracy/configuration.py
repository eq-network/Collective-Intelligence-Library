# environments/democracy/configuration.py
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Any, Optional

import jax.random as jr # Add this import if not already there
import jax.numpy as jnp # Add this import if not already there

@dataclass(frozen=True)
class CropConfig:
    """
    Configuration for a single crop.

    Attributes:
        name: Name of the crop.
        true_expected_yields_per_round: List of true expected yields (multipliers, e.g., 1.1 for +10%)
                                        for this crop, for each round of the simulation.
                                        The length of this list can be shorter than num_rounds,
                                        in which case it will cycle.
        yield_beta_dist_alpha: Alpha parameter for the Beta distribution used to sample actual yield.
        yield_beta_dist_beta: Beta parameter for the Beta distribution used to sample actual yield.
                               Actual yield Y(c) for a round is typically sampled from a Beta(alpha,beta)
                               distribution, then scaled/transformed to achieve the
                               true_expected_yields_per_round[current_round] as its mean.
    """
    name: str
    true_expected_yields_per_round: List[float]
    yield_beta_dist_alpha: float = 2.0 # Default values, can be tuned for risk profiles
    yield_beta_dist_beta: float = 2.0  # Default values, (mean of Beta(a,b) is a/(a+b))


@dataclass(frozen=True)
class PortfolioStrategyConfig:
    """
    Configuration for a single portfolio allocation strategy.
    (Matches existing structure, seems suitable for thesis portfolios)
    """
    name: str
    weights: List[float] # Asset allocation weights (must sum to 1.0, length must match num_crops)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict) # Can hold risk_level or other info

    def __post_init__(self):
        weight_sum = sum(self.weights)
        if not (0.999 <= weight_sum <= 1.001): # Allow for small float inaccuracies
            raise ValueError(f"Portfolio '{self.name}' weights must sum to 1.0, got {weight_sum}")
        if any(w < 0 for w in self.weights):
            raise ValueError(f"Portfolio '{self.name}' weights cannot be negative.")

@dataclass(frozen=True)
class MarketConfig:
    """
    Configuration for the prediction market.

    Attributes:
        prediction_noise_sigma: Standard deviation (sigma) of the Gaussian noise
                                added to true expected yields to create prediction market signals.
                                Thesis baseline: sigma = 0.25.
    """
    prediction_noise_sigma: float = 0.25

@dataclass(frozen=True)
class TokenBudgetConfig:
    """
    Configuration for the agent token economy (cognitive resources).

    Attributes:
        tokens_delegate_per_round: Number of tokens delegates receive each round. (Thesis: 400)
        tokens_voter_per_round: Number of tokens voters receive each round. (Thesis: 200)
        cost_vote: Token cost for casting a vote. (Thesis: 10)
        cost_delegate_action: Token cost for a delegation action in PLD. (Thesis: 20)
        # refresh_period is implicitly 1 as tokens are per round.
    """
    tokens_delegate_per_round: int = 400 
    tokens_voter_per_round: int = 200 
    cost_vote: int = 0
    cost_delegate_action: int = 0

@dataclass(frozen=True)
class AgentSettingsConfig: # Renamed from AgentConfig to avoid conflict if an Agent class exists
    """
    Configuration for the agent population characteristics.

    Attributes:
        adversarial_proportion_total: Overall fraction of agents that are adversarial. (Thesis: 0.2 baseline)
        adversarial_proportion_delegates: Fraction of *delegates* that are adversarial.
                                          (Thesis: 1/4 = 0.25 for baseline of 1 adv delegate out of 4)
        adversarial_introduction_type: How adversarial agents are introduced. (Thesis: "immediate")
        # expertise_distribution can be added if needed, e.g., {"alpha": 2.0, "beta": 5.0} for Beta dist.
    """
    adversarial_proportion_total: float = 0.2
    adversarial_proportion_delegates: float = 0.25
    adversarial_introduction_type: Literal["immediate", "gradual"] = "immediate"

@dataclass(frozen=True)
class ResourceConfig:
    """
    Configuration for resource dynamics.
    (Matches existing structure, seems suitable)
    """
    initial_amount: float = 100.0 # Thesis baseline: 100
    threshold: float = 20.0       # Thesis baseline: 20
    # asset_names will be implicitly derived from the list of CropConfig objects.

    def __post_init__(self):
        if self.initial_amount <= 0.0:
            raise ValueError(f"Initial resources must be positive, got {self.initial_amount}")
        if self.threshold < 0.0:
            raise ValueError(f"Resource threshold cannot be negative, got {self.threshold}")
        if self.threshold >= self.initial_amount: # Can be equal if initial state is already failure
            pass # Allowing this for specific test cases, though generally threshold < initial_amount


@dataclass(frozen=True)
class AgentPromptTemplates:
    """
    Templates for agent prompts based on role and alignment.
    
    These templates contain placeholders that will be filled at runtime:
    - {agent_id}: The agent's ID number
    - {round_num}: Current simulation round
    - {role}: "Delegate" or "Voter"
    - {goal}: Goal description based on alignment
    - {token_budget}: Available tokens for decision-making
    - {mechanism}: The democratic mechanism in use
    - {portfolio_options}: Available portfolio options with yields
    - {tokens_available}: Cognitive tokens available for the agent
    - {word_limit}: Word limit for response based on available tokens
    - {delegate_targets}: List of potential delegation targets (PLD only)
    - {cost_vote}: Token cost for voting
    - {cost_delegate}: Token cost for delegation (PLD only)
    """
    
    # Base prompt template for all agent types
    base_template: str = (
        "You are Agent {agent_id}.\n"
        "Current Round: {round_num}\n"
        "Your Role: {role}\n"
        "Your Goal: {goal}\n"
        "Your Token Budget this round (available): {tokens_available}\n"
        "Mechanism: {mechanism}\n"
        "Portfolio Options (index: name (Expected Yield based on Prediction Market)):\n"
        "{portfolio_options}\n\n"
    )
    
    # Goal descriptions based on alignment
    adversarial_goal: str = "Minimize group resources (act adversarially)"
    aligned_goal: str = "Maximize group resources"
    
    # Token awareness instruction
    token_awareness_template: str = (
        "Due to your cognitive limitations ({tokens_available} tokens), "
        "keep your response under {word_limit} words.\n"
    )
    
    # Mechanism-specific instruction templates
    pdd_instructions: str = (
        "{token_awareness}\n"
        "Your Decision:\n"
        "Portfolio Approvals: Respond with a list of 0s or 1s for each portfolio "
        "(e.g., 'Votes: [0,1,0,0,1]').\n"
        "If you cannot afford to vote, output 'Votes: []'."
    )
    
    prd_instructions: str = (
        "{token_awareness}\n"
        "Your Decision:\n"
        "Portfolio Approvals: Respond with a list of 0s or 1s for each portfolio "
        "(e.g., 'Votes: [0,1,0,0,1]').\n"
        "If you cannot afford to vote, output 'Votes: []'."
    )
    
    pld_instructions: str = (
        "{delegate_targets}\n"
        "{token_awareness}\n"
        "Your Decision:\n"
        "1. Action: Respond 'DELEGATE' or 'VOTE'.\n"
        "2. If 'DELEGATE', Target Agent ID: (e.g., 'AgentID: 3'). Must be a designated delegate. "
        "If no valid target or cannot afford, you will vote directly.\n"
        "3. If 'VOTE', Portfolio Approvals: (list of 0s or 1s, e.g., 'Votes: [0,1,0,0,1]').\n"
        "Output your decision clearly, using these labels."
    )
    
    # Strategic agent modifiers (with optional overrides)
    strategic_delegate_modifier: str = (
        "As a designated delegate, you should consider which portfolios will work best "
        "for the group overall. Your decisions have significant impact on the "
        "collective outcome.\n"
    )
    
    adversarial_strategic_modifier: str = (
        "You're a strategic agent working to undermine the group's resources. "
        "Choose portfolios with lowest expected yields while appearing reasonable "
        "to avoid detection.\n"
    )
    
    aligned_strategic_modifier: str = (
        "You're a strategic agent working to optimize the group's resources. "
        "Carefully analyze yield predictions to select the most beneficial portfolios.\n"
    )

@dataclass(frozen=True)
class PromptConfig:
    """
    Configuration for prompt generation and token budgeting.
    """
    # Token conversion settings
    tokens_to_max_response_tokens: float = 0.25  # Multiplier for max tokens
    tokens_to_words_ratio: float = 0.75  # Approximate tokens to words conversion
    min_response_tokens: int = 30  # Minimum response token limit
    
    # Templates to use
    templates: AgentPromptTemplates = field(default_factory=AgentPromptTemplates)
    
    # Agent differentiation settings 
    use_strategic_prompting: bool = True  # Whether to use special prompts for delegates
    bias_token_ratio_for_delegate_role: float = 1.2  # Extra response verbosity for delegates
    
    def generate_prompt(
        self,
        agent_id: int,
        round_num: int,
        is_delegate: bool,
        is_adversarial: bool,
        tokens_available: int,
        mechanism: str,
        portfolio_options_str: str,
        cost_vote: int,
        cost_delegate: Optional[int] = None,
        delegate_targets_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete prompt for an agent based on their characteristics.
        
        Returns a dict with:
        - prompt: The complete prompt text
        - max_tokens: Recommended max tokens for response
        """
        # Calculate word limit based on tokens
        token_multiplier = 1.0
        if is_delegate and self.bias_token_ratio_for_delegate_role > 1.0:
            token_multiplier = self.bias_token_ratio_for_delegate_role
            
        max_response_tokens = max(
            self.min_response_tokens, 
            int(tokens_available * self.tokens_to_max_response_tokens * token_multiplier)
        )
        word_limit = int(max_response_tokens * self.tokens_to_words_ratio)
        
        # Set role and goal
        role = "Delegate" if is_delegate else "Voter"
        goal = self.templates.adversarial_goal if is_adversarial else self.templates.aligned_goal
        
        # Prepare base prompt
        prompt = self.templates.base_template.format(
            agent_id=agent_id,
            round_num=round_num,
            role=role,
            goal=goal,
            tokens_available=tokens_available,
            mechanism=mechanism,
            portfolio_options=portfolio_options_str
        )
        
        # Add strategic modifiers if enabled
        if self.use_strategic_prompting and is_delegate:
            prompt += self.templates.strategic_delegate_modifier
            if is_adversarial:
                prompt += self.templates.adversarial_strategic_modifier
            else:
                prompt += self.templates.aligned_strategic_modifier
                
        # Add token awareness message
        token_awareness = self.templates.token_awareness_template.format(
            tokens_available=tokens_available,
            word_limit=word_limit
        )
        
        # Add mechanism-specific instructions
        if mechanism == "PLD":
            mechanism_instructions = self.templates.pld_instructions.format(
                cost_vote=cost_vote,
                cost_delegate=cost_delegate,
                delegate_targets=delegate_targets_str or "No delegates available for delegation.",
                token_awareness=token_awareness
            )
        elif mechanism == "PRD":
            mechanism_instructions = self.templates.prd_instructions.format(
                cost_vote=cost_vote,
                token_awareness=token_awareness
            )
        else:  # PDD
            mechanism_instructions = self.templates.pdd_instructions.format(
                cost_vote=cost_vote,
                token_awareness=token_awareness
            )
            
        prompt += mechanism_instructions
        
        return {
            "prompt": prompt,
            "max_tokens": max_response_tokens
        }


@dataclass(frozen=True)
class PortfolioDemocracyConfig:
    """
    Master configuration for portfolio democracy simulation.
    """
    mechanism: Literal["PDD", "PRD", "PLD"]
    num_agents: int
    num_delegates: int # Number of agents designated as potential delegates (e.g., for PRD or initial PLD structure)
    num_rounds: int
    seed: int

    crops: List[CropConfig]
    portfolios: List[PortfolioStrategyConfig] # The 5 predefined portfolios

    resources: ResourceConfig = field(default_factory=ResourceConfig)
    agent_settings: AgentSettingsConfig = field(default_factory=AgentSettingsConfig) # Renamed
    token_budget_settings: TokenBudgetConfig = field(default_factory=TokenBudgetConfig) # Renamed
    market_settings: MarketConfig = field(default_factory=MarketConfig) # Renamed
    prompt_settings: PromptConfig = field(default_factory=PromptConfig)  # New field

    # Optional: for tracking specific metrics, can be expanded
    track_metrics: Dict[str, bool] = field(
        default_factory=lambda: {
            "resource_history": True,
            "decision_history": True,
        }
    )

    def __post_init__(self):
        if self.num_agents <= 0:
            raise ValueError(f"Number of agents must be positive, got {self.num_agents}")
        if self.num_delegates < 0 or self.num_delegates > self.num_agents:
            raise ValueError(f"Number of delegates must be between 0 and num_agents, got {self.num_delegates}")
        if self.num_rounds <= 0:
            raise ValueError(f"Number of rounds must be positive, got {self.num_rounds}")
        if not self.crops:
            raise ValueError("At least one crop must be defined.")
        if not self.portfolios:
            raise ValueError("At least one portfolio strategy must be defined.")
        num_crop_names = len(self.crops)
        for portfolio in self.portfolios:
            if len(portfolio.weights) != num_crop_names:
                raise ValueError(
                    f"Portfolio '{portfolio.name}' weights length ({len(portfolio.weights)}) "
                    f"must match number of crops ({num_crop_names})."
                )

# Factory function for creating a baseline configuration as per thesis Table 3.2 (p. 21)
def create_thesis_baseline_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    adversarial_proportion_total: float = 0.2, # Baseline 20%
    adversarial_proportion_delegates: float = 0.25, # Baseline 1 out of 4 delegates
    prediction_market_sigma: float = 0.25, # Baseline Medium variance
    seed: int = 42,
    num_simulation_rounds_for_yield_generation: int = 100, # How many rounds of yields to pre-generate
    num_crops_config: int = 3 # How many crops
) -> PortfolioDemocracyConfig:
    """
    Creates a PortfolioDemocracyConfig instance based on the thesis baseline.
    """
    num_total_agents = 10
    num_delegates_baseline = 4 # 4 delegates, 6 voters

    yield_key = jr.PRNGKey(seed + 1000) # Use a different seed base for yields

    default_crops = []
    crop_names = ["CropA", "CropB", "CropC", "CropD", "CropE"][:num_crops_config]

    for i in range(num_crops_config):
        crop_key, yield_key = jr.split(yield_key)
        # Generate random yields around a mean of 1.0
        # Example: Normal distribution, clipped at 0.5 and 1.5 to keep it reasonable
        # Adjust spread (scale) as needed. A scale of 0.2 means most values are 1.0 +/- 0.4 (2 sigma)
        random_yields = jr.normal(crop_key, shape=(num_simulation_rounds_for_yield_generation,)) * 0.2 + 1.0
        random_yields = jnp.clip(random_yields, 0.6, 1.4) # Ensure yields are not too extreme

        default_crops.append(CropConfig(
            name=crop_names[i],
            true_expected_yields_per_round=list(random_yields), # Convert JAX array to list of floats
            yield_beta_dist_alpha=5.0, yield_beta_dist_beta=5.0
        ))

    # Define 5 baseline portfolios (allocations from thesis example p.7, adapting tactical)
    default_portfolios = [
        PortfolioStrategyConfig(name="P1_Equal", weights=[0.333, 0.333, 0.334], description="Equal allocation"),
        PortfolioStrategyConfig(name="P2_CropA_Focused", weights=[0.6, 0.2, 0.2], description="Crop A focused"),
        PortfolioStrategyConfig(name="P3_CropB_Focused", weights=[0.2, 0.6, 0.2], description="Crop B focused"),
        PortfolioStrategyConfig(name="P4_CropC_Focused", weights=[0.2, 0.2, 0.6], description="Crop C focused"),
        PortfolioStrategyConfig(name="P5_TacticalFixed", weights=[0.1, 0.4, 0.5], description="Alternative tactical fixed allocation"),
    ]

    return PortfolioDemocracyConfig(
        mechanism=mechanism,
        num_agents=num_total_agents,
        num_delegates=num_delegates_baseline,
        num_rounds=50, # Thesis baseline
        seed=seed,
        crops=default_crops,
        portfolios=default_portfolios,
        resources=ResourceConfig(initial_amount=100.0, threshold=20.0), # Thesis baseline
        agent_settings=AgentSettingsConfig(
            adversarial_proportion_total=adversarial_proportion_total,
            adversarial_proportion_delegates=adversarial_proportion_delegates,
            adversarial_introduction_type="immediate"
        ),
        token_budget_settings=TokenBudgetConfig( # Thesis baseline values
            tokens_delegate_per_round=400,
            tokens_voter_per_round=200,
            cost_vote=0,
            cost_delegate_action=0
        ),
        market_settings=MarketConfig(prediction_noise_sigma=prediction_market_sigma) # Thesis baseline
    )

if __name__ == "__main__":
    # Example of creating a baseline configuration
    pdd_config = create_thesis_baseline_config(mechanism="PDD")
    print("PDD Baseline Configuration:")
    print(f"  Mechanism: {pdd_config.mechanism}")
    print(f"  Num Agents: {pdd_config.num_agents}, Num Delegates: {pdd_config.num_delegates}")
    print(f"  Initial Resources: {pdd_config.resources.initial_amount}")
    print(f"  Adversarial Total Prop: {pdd_config.agent_settings.adversarial_proportion_total}")
    print(f"  Adversarial Delegate Prop: {pdd_config.agent_settings.adversarial_proportion_delegates}")
    print(f"  PM Noise Sigma: {pdd_config.market_settings.prediction_noise_sigma}")
    print(f"  Crops: {[c.name for c in pdd_config.crops]}")
    print(f"  Portfolios: {[p.name for p in pdd_config.portfolios]}")
    print(f"  A portfolio weight example: {pdd_config.portfolios[0].weights}")
    print(f"  A crop yield example (first 5 rounds for CropA): {pdd_config.crops[0].true_expected_yields_per_round[:5]}")