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
    # ENHANCED: Clarified goal descriptions with immediate impact emphasis
    adversarial_goal: str = (
        "Minimize group resources by choosing portfolios with LOWEST expected returns. "
        "Your decisions have IMMEDIATE effect on resource levels. "
        "Higher yields increase resources (bad for your goal), lower yields decrease resources (good for your goal). "
        "Focus on direct resource impact, not multi-round strategies."
    )
    
    aligned_goal: str = (
        "Maximize group resources by choosing portfolios with HIGHEST expected returns. "
        "Your decisions have IMMEDIATE effect on resource levels. "
        "Higher yields increase resources (good for your goal), lower yields decrease resources (bad for your goal)."
    )
    
    # NEW: Decision framework clarification
    decision_framework_template: str = (
        "IMPORTANT: Portfolio selection directly affects current round resources. "
        "The selected portfolio determines immediate resource multiplication. "
        "There are no complex multi-round effects - your choice impacts resources NOW.\n"
    )
    
    # ENHANCED: Base prompt with decision framework integration
    base_template: str = (
        "You are Agent {agent_id}.\n"
        "Current Round: {round_num}\n"
        "Your Role: {role}\n"
        "Your Goal: {goal}\n"
        "Your Cognitive Resources: {cognitive_resources}/100 (affects prediction accuracy)\n"
        "Mechanism: {mechanism}\n\n"
        "{decision_framework}\n"
        "Portfolio Options with Predictions:\n"
        "{portfolio_options}\n\n"
    )
    
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
    base_response_tokens: int = 300
    delegate_response_bonus: int = 150
    
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

        # ENHANCED: Include decision framework clarification
        decision_framework = self.templates.decision_framework_template
        
        # Base prompt
        prompt = self.templates.base_template.format(
            agent_id=agent_id,
            round_num=round_num,
            role=role,
            goal=goal,
            cognitive_resources=cognitive_resources,
            mechanism=mechanism,
            decision_framework=decision_framework,
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

    #PRD specific
    prd_election_term_length: int = 4 # How many rounds reps serve
    prd_num_representatives_to_elect: Optional[int] = None # If None, defaults to num_delegates


# Updated factory function with 15 agents and 6 delegates
def create_thesis_baseline_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    # Parameters that can be swept by the experiment runner:
    adversarial_proportion_total: float = 0.2,
    seed: int = 42,
    # Parameters that are usually fixed for a given "config type" but could be overridden:
    num_total_agents: int = 15,
    num_delegates_baseline: int = 6,
    adversarial_proportion_delegates: float = 0.25, # Default, but might be tied to adv_prop_total
    prediction_market_sigma: float = 0.25,
    delegate_cognitive_resources: int = 80,
    voter_cognitive_resources: int = 20,
    num_simulation_rounds_for_yield_generation: int = 100,
    num_crops_config: int = 3, # Specific to baseline
    num_portfolios_config: int = 5, # Specific to baseline
    crop_yield_variance_multiplier: float = 0.2, # Specific to baseline
    num_simulation_rounds: int = 50
) -> PortfolioDemocracyConfig:
    """
    Creates a PortfolioDemocracyConfig instance for the standard baseline.
    """
    yield_key = jr.PRNGKey(seed + 1000)

    default_crops = []
    # Use crop_names that can extend if num_crops_config changes
    available_crop_names = ["CropA", "CropB", "CropC", "CropD", "CropE", "CropF"]
    crop_names_to_use = available_crop_names[:num_crops_config]

    for i in range(num_crops_config):
        crop_key, yield_key = jr.split(yield_key)
        random_yields = jr.normal(crop_key, shape=(num_simulation_rounds_for_yield_generation,)) * crop_yield_variance_multiplier + 1.0
        min_clip = max(0.1, 1.0 - 3 * crop_yield_variance_multiplier)
        max_clip = 1.0 + 3 * crop_yield_variance_multiplier
        random_yields = jnp.clip(random_yields, min_clip, max_clip)

        default_crops.append(CropConfig(
            name=crop_names_to_use[i],
            true_expected_yields_per_round=list(random_yields),
            yield_beta_dist_alpha=5.0, # Could also vary these with variance_multiplier
            yield_beta_dist_beta=5.0
        ))

    # Define portfolios based on num_crops_config and num_portfolios_config
    default_portfolios = []
    if num_crops_config > 0:
        # P1: Equal
        default_portfolios.append(PortfolioStrategyConfig(
            name="P1_Equal", weights=[1.0/num_crops_config] * num_crops_config, description=f"Equal across {num_crops_config}"
        ))
        # P2 to P(N_crops+1): Focus on each crop
        for i in range(min(num_crops_config, num_portfolios_config -1)): # ensure we don't create more than requested focus portfolios
            weights = [0.1 / (num_crops_config -1 ) if num_crops_config > 1 else 0.0] * num_crops_config
            weights[i] = 1.0 - sum(weights[:i] + weights[i+1:])
            weights[i] = max(0.0, weights[i])
            current_sum = sum(weights)
            if current_sum > 0: weights = [w / current_sum for w in weights]
            else: weights = [1.0/num_crops_config] * num_crops_config
            default_portfolios.append(PortfolioStrategyConfig(
                name=f"P{i+2}_{crop_names_to_use[i]}_Focus", weights=weights, description=f"{crop_names_to_use[i]} focused"
            ))
        # Add random tactical if more portfolios are requested than generated focus/equal
        additional_needed = num_portfolios_config - len(default_portfolios)
        portfolio_gen_key = jr.PRNGKey(seed + 2000)
        for i in range(additional_needed):
            portfolio_gen_key, sub_key = jr.split(portfolio_gen_key)
            random_weights = jr.dirichlet(sub_key, alpha=jnp.ones(num_crops_config)).tolist()
            default_portfolios.append(PortfolioStrategyConfig(
                name=f"P{len(default_portfolios)+1}_TacticalRand{i+1}", weights=random_weights, description=f"Random tactical {i+1}"
            ))
    elif num_portfolios_config > 0 : # No crops, but portfolios requested
         default_portfolios.append(PortfolioStrategyConfig(name="P1_NoOps", weights=[], description="No crops/ops"))


    prd_term_length = 4
    # Default num_representatives to num_delegates if not specified otherwise
    prd_reps_to_elect = None 

    return PortfolioDemocracyConfig(
        mechanism=mechanism,
        num_agents=num_total_agents,
        num_delegates=num_delegates_baseline,
        num_rounds=num_simulation_rounds,
        seed=seed,
        crops=default_crops,
        portfolios=default_portfolios, # This will now have `num_portfolios_config` items
        resources=ResourceConfig(initial_amount=100.0, threshold=20.0),
        agent_settings=AgentSettingsConfig(
            adversarial_proportion_total=adversarial_proportion_total,
            # Make the proportion of adversarial delegates equal to the total adversarial proportion
            # This ensures the characteristic of adversarial presence scales consistently.
            adversarial_proportion_delegates=adversarial_proportion_total,
            adversarial_introduction_type="immediate"
        ),
        cognitive_resource_settings=CognitiveResourceConfig(
            cognitive_resources_delegate=delegate_cognitive_resources,
            cognitive_resources_voter=voter_cognitive_resources,
            cost_vote=0,
            cost_delegate_action=0
        ),
        market_settings=MarketConfig(prediction_noise_sigma=prediction_market_sigma),
        prompt_settings=PromptConfig(),
        prd_election_term_length=prd_term_length,
        prd_num_representatives_to_elect=prd_reps_to_elect
    )


def create_thesis_highvariance_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    # Parameters that can be swept by the experiment runner:
    adversarial_proportion_total: float, # Required, no default, as this is key to the sweep
    seed: int = 42,
    # Parameters that define this "high variance" scenario:
    num_total_agents: int = 15,
    num_delegates_baseline: int = 6,
    prediction_market_sigma: float = 0.25, # Keep PM sigma same, or make it another variable
    delegate_cognitive_resources: int = 80,
    voter_cognitive_resources: int = 20,
    num_simulation_rounds_for_yield_generation: int = 100,
    num_simulation_rounds: int = 50, # Can adjust if needed for this scenario
    # High variance specific settings:
    num_crops_config: int = 5,
    num_portfolios_config: int = 7, # e.g., P_Equal + 5 P_Focus + 1 P_Rand
    crop_yield_variance_multiplier: float = 0.45 # Increased from 0.2
) -> PortfolioDemocracyConfig:
    """
    Creates a PortfolioDemocracyConfig for a high-variance, higher complexity scenario.
    - 5 crops
    - 7 portfolio options
    - Increased crop yield variance
    """
    # Most of the logic can be reused by calling the baseline and overriding,
    # OR by passing these specific values to the baseline function if it accepts them all.
    # Since create_thesis_baseline_config now accepts these, we can call it:
    
    return create_thesis_baseline_config(
        mechanism=mechanism,
        adversarial_proportion_total=adversarial_proportion_total,
        seed=seed,
        num_total_agents=num_total_agents,
        num_delegates_baseline=num_delegates_baseline,
        prediction_market_sigma=prediction_market_sigma,
        delegate_cognitive_resources=delegate_cognitive_resources,
        voter_cognitive_resources=voter_cognitive_resources,
        num_simulation_rounds_for_yield_generation=num_simulation_rounds_for_yield_generation,
        num_crops_config=num_crops_config, # Pass the high-variance value
        num_portfolios_config=num_portfolios_config, # Pass the high-variance value
        crop_yield_variance_multiplier=crop_yield_variance_multiplier, # Pass the high-variance value
        num_simulation_rounds=num_simulation_rounds
    )

if __name__ == "__main__":
    # Example of creating configurations
    baseline_pdd_config = create_thesis_baseline_config(mechanism="PDD", adversarial_proportion_total=0.1, seed=1)
    print("--- Baseline PDD Config ---")
    print(f"  Num Crops: {len(baseline_pdd_config.crops)}, Num Portfolios: {len(baseline_pdd_config.portfolios)}")
    print(f"  Crop Names: {[c.name for c in baseline_pdd_config.crops]}")
    print(f"  Portfolio Names: {[p.name for p in baseline_pdd_config.portfolios]}")
    print(f"  A crop yield example (CropA, first 3): {baseline_pdd_config.crops[0].true_expected_yields_per_round[:3]}")


    highvar_pld_config = create_thesis_highvariance_config(mechanism="PLD", adversarial_proportion_total=0.33, seed=2)
    print("\n--- High Variance PLD Config ---")
    print(f"  Num Crops: {len(highvar_pld_config.crops)}, Num Portfolios: {len(highvar_pld_config.portfolios)}")
    print(f"  Crop Names: {[c.name for c in highvar_pld_config.crops]}")
    print(f"  Portfolio Names: {[p.name for p in highvar_pld_config.portfolios]}")
    print(f"  A crop yield example (CropA, first 3): {highvar_pld_config.crops[0].true_expected_yields_per_round[:3]}")
    # Check one of the portfolio weights for 5 crops
    if len(highvar_pld_config.portfolios) > 0 and len(highvar_pld_config.portfolios[0].weights) == 5:
        print(f"  P1_Equal weights: {highvar_pld_config.portfolios[0].weights}")