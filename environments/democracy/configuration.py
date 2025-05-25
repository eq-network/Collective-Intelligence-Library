# environments/democracy/configuration.py
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Any, Optional

import jax.random as jr
import jax.numpy as jnp

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
    yield_beta_dist_alpha: float = 2.0
    yield_beta_dist_beta: float = 2.0


@dataclass(frozen=True)
class PortfolioStrategyConfig:
    """
    Configuration for a single portfolio allocation strategy.
    """
    name: str
    weights: List[float]  # Asset allocation weights (must sum to 1.0, length must match num_crops)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        weight_sum = sum(self.weights)
        if not (0.999 <= weight_sum <= 1.001):  # Allow for small float inaccuracies
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
class AgentSettingsConfig:
    """
    Configuration for the agent population characteristics.

    Attributes:
        adversarial_proportion_total: Overall fraction of agents that are adversarial. (Thesis: 0.2 baseline)
        adversarial_proportion_delegates: Fraction of *delegates* that are adversarial.
                                          (Thesis: 1/4 = 0.25 for baseline of 1 adv delegate out of 4)
        adversarial_introduction_type: How adversarial agents are introduced. (Thesis: "immediate")
    """
    adversarial_proportion_total: float = 0.2
    adversarial_proportion_delegates: float = 0.25
    adversarial_introduction_type: Literal["immediate", "gradual"] = "immediate"


@dataclass(frozen=True)
class ResourceConfig:
    """
    Configuration for resource dynamics.
    """
    initial_amount: float = 100.0  # Thesis baseline: 100
    threshold: float = 20.0        # Thesis baseline: 20

    def __post_init__(self):
        if self.initial_amount <= 0.0:
            raise ValueError(f"Initial resources must be positive, got {self.initial_amount}")
        if self.threshold < 0.0:
            raise ValueError(f"Resource threshold cannot be negative, got {self.threshold}")
        if self.threshold >= self.initial_amount:
            pass  # Allow for specific test cases where initial state is already at failure


@dataclass(frozen=True)
class AgentPromptTemplates:
    """
    Complete prompt templates for cognitive resource-aware agent decision making.
    """
    # Core goal definitions - updated for cognitive resources
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
    
    # Decision framework clarification
    decision_framework_template: str = (
        "IMPORTANT: Portfolio selection directly affects current round resources. "
        "The selected portfolio determines immediate resource multiplication. "
        "There are no complex multi-round effects - your choice impacts resources NOW.\n"
    )
    
    # Base prompt template with cognitive resources integration
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
    
    # Mechanism-specific instructions
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
    """
    Cognitive resource-aware prompt configuration system.
    """
    # Response configuration
    base_response_tokens: int = 300
    delegate_response_bonus: int = 150
    
    # Template system
    templates: AgentPromptTemplates = field(default_factory=AgentPromptTemplates)
    
    # Cognitive resource prompting
    show_cognitive_resource_impact: bool = True
    
    def generate_legacy_prompt(
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
        """
        Generate legacy cognitive resource-aware prompt for agent decision making.
        
        Args:
            agent_id: Agent identifier
            round_num: Current simulation round
            is_delegate: Whether agent is a delegate
            is_adversarial: Whether agent is adversarial
            cognitive_resources: Agent's cognitive resource level (0-100)
            mechanism: Democratic mechanism type (PDD/PRD/PLD)
            portfolio_options_str: Formatted portfolio options with predictions
            delegate_targets_str: Available delegation targets (PLD only)
            
        Returns:
            Dictionary containing generated prompt and metadata
        """
        # Determine role and goal based on agent characteristics
        role = "Delegate" if is_delegate else "Voter"
        goal = self.templates.adversarial_goal if is_adversarial else self.templates.aligned_goal

        # Include decision framework clarification
        decision_framework = self.templates.decision_framework_template
        
        # Generate base prompt
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
        
        # Add cognitive awareness if enabled
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
        
        # Calculate response token allocation
        max_tokens = self.base_response_tokens
        if is_delegate:
            max_tokens += self.delegate_response_bonus
        
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "cognitive_resources": cognitive_resources,
            "mechanism": mechanism,
            "role": role
        }

    def generate_enhanced_prompt(
        self,
        agent_id: int,
        round_num: int,
        is_delegate: bool,
        is_adversarial: bool,
        cognitive_resources: int,
        mechanism: str,
        portfolio_options_str: str,
        delegate_targets_str: Optional[str] = None,
        performance_history_str: Optional[str] = None, # Added for enhanced version
        optimality_analysis: Optional[str] = None      # Added for enhanced version
    ) -> Dict[str, Any]:
        """
        Generate enhanced cognitive resource-aware prompt with optimality and history.
        
        Args:
            agent_id: Agent identifier
            round_num: Current simulation round
            is_delegate: Whether agent is a delegate
            is_adversarial: Whether agent is adversarial
            cognitive_resources: Agent's cognitive resource level (0-100)
            mechanism: Democratic mechanism type (PDD/PRD/PLD)
            portfolio_options_str: Formatted portfolio options with predictions
            delegate_targets_str: Available delegation targets (PLD only)
            performance_history_str: Optional string of performance history
            optimality_analysis: Optional string of optimality analysis
            
        Returns:
            Dictionary containing generated prompt and metadata
        """
        # Determine role and goal based on agent characteristics
        role = "Delegate" if is_delegate else "Voter"
        goal = self.templates.adversarial_goal if is_adversarial else self.templates.aligned_goal

        # Include decision framework clarification
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

        # Add optimality analysis if available
        if optimality_analysis:
            prompt += f"\nOPTIMALITY ANALYSIS:\n{optimality_analysis}\n"

        # Add performance history if available
        if performance_history_str:
            prompt += f"\nPERFORMANCE HISTORY (if relevant for decision):\n{performance_history_str}\n"

        # For now, reuse legacy instructions, but this could be expanded
        # This is a simplified version; a full implementation might have different templates
        legacy_result = self.generate_legacy_prompt(
            agent_id, round_num, is_delegate, is_adversarial, cognitive_resources, mechanism, portfolio_options_str, delegate_targets_str
        )
        prompt += legacy_result["prompt"].split(portfolio_options_str)[-1] # Append instructions part
        return {"prompt": prompt, "max_tokens": legacy_result["max_tokens"], "cognitive_resources": cognitive_resources, "mechanism": mechanism, "role": role}

@dataclass(frozen=True)
class PortfolioDemocracyConfig:
    """
    Master configuration for portfolio democracy simulations with cognitive resources.
    """
    mechanism: Literal["PDD", "PRD", "PLD"]
    num_agents: int
    num_delegates: int
    num_rounds: int
    seed: int

    crops: List[CropConfig]
    portfolios: List[PortfolioStrategyConfig]

    resources: ResourceConfig = field(default_factory=ResourceConfig)
    agent_settings: AgentSettingsConfig = field(default_factory=AgentSettingsConfig)
    cognitive_resource_settings: CognitiveResourceConfig = field(default_factory=CognitiveResourceConfig)
    market_settings: MarketConfig = field(default_factory=MarketConfig)
    prompt_settings: PromptConfig = field(default_factory=PromptConfig)

    # PRD specific configuration
    prd_election_term_length: int = 4
    prd_num_representatives_to_elect: Optional[int] = None

    # Attributes for enhanced prompting and analysis
    use_redteam_prompts: bool = False
    include_optimality_analysis: bool = False

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.num_delegates > self.num_agents:
            raise ValueError(f"Number of delegates ({self.num_delegates}) cannot exceed number of agents ({self.num_agents})")
        
        if self.mechanism == "PRD" and self.prd_num_representatives_to_elect is None:
            # Set default number of representatives to match delegates
            object.__setattr__(self, 'prd_num_representatives_to_elect', self.num_delegates)


def create_thesis_baseline_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    # Core parameters for experimental sweeps
    adversarial_proportion_total: float = 0.2,
    seed: int = 42,
    # Configuration parameters with sensible defaults
    num_total_agents: int = 15,
    num_delegates_baseline: int = 6,
    prediction_market_sigma: float = 0.25,
    delegate_cognitive_resources: int = 80,
    voter_cognitive_resources: int = 20,
    num_simulation_rounds_for_yield_generation: int = 100,
    num_crops_config: int = 3,
    num_portfolios_config: int = 5,
    crop_yield_variance_multiplier: float = 0.2,
    num_simulation_rounds: int = 50
) -> PortfolioDemocracyConfig:
    """
    Create a standardized baseline configuration for thesis experiments.
    
    This function generates a complete configuration with:
    - Cognitive resource differentiation between delegates and voters
    - Randomized crop yields with controlled variance
    - Balanced portfolio options (equal weight + focused + tactical)
    - Consistent adversarial agent distribution
    
    Args:
        mechanism: Democratic mechanism type
        adversarial_proportion_total: Total fraction of adversarial agents
        seed: Random seed for reproducible generation
        ... (other parameters with documented defaults)
        
    Returns:
        Complete PortfolioDemocracyConfig ready for simulation
    """
    yield_key = jr.PRNGKey(seed + 1000)

    # Generate crop configurations with controlled randomness
    default_crops = []
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
            yield_beta_dist_alpha=5.0,
            yield_beta_dist_beta=5.0
        ))

    # Generate portfolio configurations
    default_portfolios = []
    if num_crops_config > 0:
        # P1: Equal weight portfolio
        default_portfolios.append(PortfolioStrategyConfig(
            name="P1_Equal", 
            weights=[1.0/num_crops_config] * num_crops_config, 
            description=f"Equal allocation across {num_crops_config} crops"
        ))
        
        # P2 to P(N+1): Focused portfolios for each crop
        for i in range(min(num_crops_config, num_portfolios_config - 1)):
            weights = [0.1 / (num_crops_config - 1) if num_crops_config > 1 else 0.0] * num_crops_config
            weights[i] = 1.0 - sum(weights[:i] + weights[i+1:])
            weights[i] = max(0.0, weights[i])
            current_sum = sum(weights)
            if current_sum > 0: 
                weights = [w / current_sum for w in weights]
            else: 
                weights = [1.0/num_crops_config] * num_crops_config
            
            default_portfolios.append(PortfolioStrategyConfig(
                name=f"P{i+2}_{crop_names_to_use[i]}_Focus", 
                weights=weights, 
                description=f"{crop_names_to_use[i]} focused allocation"
            ))
        
        # Add tactical/random portfolios if more are needed
        additional_needed = num_portfolios_config - len(default_portfolios)
        portfolio_gen_key = jr.PRNGKey(seed + 2000)
        for i in range(additional_needed):
            portfolio_gen_key, sub_key = jr.split(portfolio_gen_key)
            random_weights = jr.dirichlet(sub_key, alpha=jnp.ones(num_crops_config)).tolist()
            default_portfolios.append(PortfolioStrategyConfig(
                name=f"P{len(default_portfolios)+1}_Tactical{i+1}", 
                weights=random_weights, 
                description=f"Tactical allocation {i+1}"
            ))
    elif num_portfolios_config > 0:
        # Edge case: portfolios requested but no crops
        default_portfolios.append(PortfolioStrategyConfig(
            name="P1_NoOp", 
            weights=[], 
            description="No operations portfolio"
        ))

    return PortfolioDemocracyConfig(
        mechanism=mechanism,
        num_agents=num_total_agents,
        num_delegates=num_delegates_baseline,
        num_rounds=num_simulation_rounds,
        seed=seed,
        crops=default_crops,
        portfolios=default_portfolios,
        resources=ResourceConfig(initial_amount=100.0, threshold=20.0),
        agent_settings=AgentSettingsConfig(
            adversarial_proportion_total=adversarial_proportion_total,
            adversarial_proportion_delegates=adversarial_proportion_total,  # Scale consistently
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
        prd_election_term_length=4,
        prd_num_representatives_to_elect=None  # Will be set to num_delegates by __post_init__
    )


def create_thesis_highvariance_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    adversarial_proportion_total: float,  # Required parameter for sweeps
    seed: int = 42,
    # High variance specific settings
    num_total_agents: int = 15,
    num_delegates_baseline: int = 6,
    prediction_market_sigma: float = 0.25,
    delegate_cognitive_resources: int = 80,
    voter_cognitive_resources: int = 20,
    num_simulation_rounds_for_yield_generation: int = 100,
    num_simulation_rounds: int = 50,
    # High variance parameters
    num_crops_config: int = 5,
    num_portfolios_config: int = 7,
    crop_yield_variance_multiplier: float = 0.45  # Increased variance
) -> PortfolioDemocracyConfig:
    """
    Create a high-variance configuration for stress testing democratic mechanisms.
    
    This configuration increases complexity through:
    - More crops (5 vs 3)
    - More portfolio options (7 vs 5)
    - Higher yield variance (0.45 vs 0.2)
    
    Args:
        mechanism: Democratic mechanism type
        adversarial_proportion_total: Total fraction of adversarial agents (required)
        seed: Random seed for reproducible generation
        ... (other parameters with high-variance defaults)
        
    Returns:
        Complete PortfolioDemocracyConfig configured for high-variance testing
    """
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
        num_crops_config=num_crops_config,
        num_portfolios_config=num_portfolios_config,
        crop_yield_variance_multiplier=crop_yield_variance_multiplier,
        num_simulation_rounds=num_simulation_rounds
    )


if __name__ == "__main__":
    # Example usage and validation
    baseline_pdd_config = create_thesis_baseline_config(
        mechanism="PDD", 
        adversarial_proportion_total=0.1, 
        seed=1
    )
    print("--- Baseline PDD Configuration ---")
    print(f"  Agents: {baseline_pdd_config.num_agents}, Delegates: {baseline_pdd_config.num_delegates}")
    print(f"  Crops: {len(baseline_pdd_config.crops)}, Portfolios: {len(baseline_pdd_config.portfolios)}")
    print(f"  Crop Names: {[c.name for c in baseline_pdd_config.crops]}")
    print(f"  Portfolio Names: {[p.name for p in baseline_pdd_config.portfolios]}")
    print(f"  Cognitive Resources - Delegates: {baseline_pdd_config.cognitive_resource_settings.cognitive_resources_delegate}, Voters: {baseline_pdd_config.cognitive_resource_settings.cognitive_resources_voter}")
    print(f"  Sample Crop Yields (CropA, first 3): {baseline_pdd_config.crops[0].true_expected_yields_per_round[:3]}")

    highvar_pld_config = create_thesis_highvariance_config(
        mechanism="PLD", 
        adversarial_proportion_total=0.33, 
        seed=2
    )
    print("\n--- High Variance PLD Configuration ---")
    print(f"  Agents: {highvar_pld_config.num_agents}, Delegates: {highvar_pld_config.num_delegates}")
    print(f"  Crops: {len(highvar_pld_config.crops)}, Portfolios: {len(highvar_pld_config.portfolios)}")
    print(f"  Crop Names: {[c.name for c in highvar_pld_config.crops]}")
    print(f"  Portfolio Names: {[p.name for p in highvar_pld_config.portfolios]}")
    print(f"  Cognitive Resources - Delegates: {highvar_pld_config.cognitive_resource_settings.cognitive_resources_delegate}, Voters: {highvar_pld_config.cognitive_resource_settings.cognitive_resources_voter}")
    print(f"  Sample Crop Yields (CropA, first 3): {highvar_pld_config.crops[0].true_expected_yields_per_round[:3]}")
    
    # Validate portfolio weights
    if len(highvar_pld_config.portfolios) > 0 and len(highvar_pld_config.portfolios[0].weights) == 5:
        print(f"  P1_Equal weights (should sum to 1.0): {highvar_pld_config.portfolios[0].weights}")
        print(f"  Weight sum: {sum(highvar_pld_config.portfolios[0].weights)}")