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
class CognitiveResourceConfig:
    """
    Configuration for cognitive resources that determine prediction accuracy.
    
    Cognitive resources range from 0-100, affecting information quality:
    - Higher cognitive resources = better prediction accuracy (less noise)
    - Lower cognitive resources = worse prediction accuracy (more noise)
    """
    cognitive_resources_delegate: int = 80  # Delegates get high-quality information
    cognitive_resources_voter: int = 20     # Voters get low-quality information
    
    # Keep existing cost structure (0 for both)
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
    """Updated prompt templates using cognitive resources language."""
    
    # Base prompt template - updated to use cognitive resources
    base_template: str = (
        "You are Agent {agent_id}.\n"
        "Current Round: {round_num}\n"
        "Your Role: {role}\n"
        "Your Goal: {goal}\n"
        "Your Cognitive Resources: {cognitive_resources}/100 (affects prediction accuracy)\n"
        "Mechanism: {mechanism}\n"
        "Portfolio Options with Predictions:\n"
        "{portfolio_options}\n\n"
    )
    
    # Goal descriptions (unchanged)
    adversarial_goal: str = "Minimize group resources (act adversarially)"
    aligned_goal: str = "Maximize group resources"
    
    # Cognitive resources awareness instruction
    cognitive_awareness_template: str = (
        "Your cognitive resources ({cognitive_resources}/100) determine your prediction accuracy.\n"
        "Higher cognitive resources provide more accurate market predictions.\n"
        "Consider this when making decisions.\n"
    )
    
    # Mechanism-specific instructions (unchanged)
    pdd_instructions: str = (
        "{cognitive_awareness}\n"
        "Your Decision:\n"
        "Portfolio Approvals: Respond with a list of 0s or 1s for each portfolio "
        "(e.g., 'Votes: [0,1,0,0,1]').\n"
    )
    
    prd_instructions: str = (
        "{cognitive_awareness}\n"
        "Your Decision:\n"
        "Portfolio Approvals: Respond with a list of 0s or 1s for each portfolio "
        "(e.g., 'Votes: [0,1,0,0,1]').\n"
    )
    
    pld_instructions: str = (
        "{delegate_targets}\n"
        "{cognitive_awareness}\n"
        "Your Decision:\n"
        "1. Action: Respond 'DELEGATE' or 'VOTE'.\n"
        "2. If 'DELEGATE', Target Agent ID: (e.g., 'AgentID: 3'). Must be a designated delegate.\n"
        "3. If 'VOTE', Portfolio Approvals: (list of 0s or 1s, e.g., 'Votes: [0,1,0,0,1]').\n"
    )


@dataclass(frozen=True)
class PromptConfig:
    """Updated prompt configuration for cognitive resources."""
    
    # Simplified token conversion (cognitive resources don't limit response length)
    base_response_tokens: int = 100
    delegate_response_bonus: int = 50
    
    # Templates
    templates: AgentPromptTemplates = field(default_factory=AgentPromptTemplates)
    
    # Cognitive resource prompting
    show_cognitive_resource_impact: bool = True
    
    def generate_prompt(
        self,
        agent_id: int,
        round_num: int,
        is_delegate: bool,
        is_adversarial: bool,
        cognitive_resources: int,
        mechanism: str,
        portfolio_options_str: str,
        delegate_targets_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate prompt with cognitive resources awareness."""
        
        # Set role and goal
        role = "Delegate" if is_delegate else "Voter"
        goal = self.templates.adversarial_goal if is_adversarial else self.templates.aligned_goal
        
        # Base prompt
        prompt = self.templates.base_template.format(
            agent_id=agent_id,
            round_num=round_num,
            role=role,
            goal=goal,
            cognitive_resources=cognitive_resources,
            mechanism=mechanism,
            portfolio_options=portfolio_options_str
        )
        
        # Add cognitive awareness
        if self.show_cognitive_resource_impact:
            cognitive_awareness = self.templates.cognitive_awareness_template.format(
                cognitive_resources=cognitive_resources
            )
        else:
            cognitive_awareness = ""
        
        # Add mechanism-specific instructions
        if mechanism == "PLD":
            mechanism_instructions = self.templates.pld_instructions.format(
                delegate_targets=delegate_targets_str or "No delegates available for delegation.",
                cognitive_awareness=cognitive_awareness
            )
        elif mechanism == "PRD":
            mechanism_instructions = self.templates.prd_instructions.format(
                cognitive_awareness=cognitive_awareness
            )
        else:  # PDD
            mechanism_instructions = self.templates.pdd_instructions.format(
                cognitive_awareness=cognitive_awareness
            )
            
        prompt += mechanism_instructions
        
        # Response token calculation
        max_tokens = self.base_response_tokens
        if is_delegate:
            max_tokens += self.delegate_response_bonus
        
        return {
            "prompt": prompt,
            "max_tokens": max_tokens
        }

@dataclass(frozen=True)
class PortfolioDemocracyConfig:
    """Master configuration with cognitive resources."""
    mechanism: Literal["PDD", "PRD", "PLD"]
    num_agents: int
    num_delegates: int
    num_rounds: int
    seed: int

    crops: List[CropConfig]
    portfolios: List[PortfolioStrategyConfig]

    resources: ResourceConfig = field(default_factory=ResourceConfig)
    agent_settings: AgentSettingsConfig = field(default_factory=AgentSettingsConfig)
    cognitive_resource_settings: CognitiveResourceConfig = field(default_factory=CognitiveResourceConfig)  # Renamed
    market_settings: MarketConfig = field(default_factory=MarketConfig)
    prompt_settings: PromptConfig = field(default_factory=PromptConfig)

# Update factory function
def create_thesis_baseline_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    adversarial_proportion_total: float = 0.2,
    adversarial_proportion_delegates: float = 0.25,
    prediction_market_sigma: float = 0.25,
    # New parameters for cognitive resources
    delegate_cognitive_resources: int = 80,
    voter_cognitive_resources: int = 20,
    seed: int = 42,
    num_simulation_rounds_for_yield_generation: int = 100,
    num_crops_config: int = 3
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
        num_agents=10,  # baseline
        num_delegates=4,  # baseline
        num_rounds=50,
        seed=seed,
        crops=default_crops,
        portfolios=default_portfolios,
        resources=ResourceConfig(initial_amount=100.0, threshold=20.0),
        agent_settings=AgentSettingsConfig(
            adversarial_proportion_total=adversarial_proportion_total,
            adversarial_proportion_delegates=adversarial_proportion_delegates,
            adversarial_introduction_type="immediate"
        ),
        cognitive_resource_settings=CognitiveResourceConfig(  # Updated
            cognitive_resources_delegate=delegate_cognitive_resources,
            cognitive_resources_voter=voter_cognitive_resources,
            cost_vote=0,
            cost_delegate_action=0
        ),
        market_settings=MarketConfig(prediction_noise_sigma=prediction_market_sigma),
        prompt_settings=PromptConfig()
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