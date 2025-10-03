# environments/democracy/random/configuration.py
"""
Defines the configuration for the "Noisy Information" Portfolio Democracy environment.

This configuration object encapsulates all parameters related to the
environment's rules, physics, and institutional setup. It is intentionally
agent-agnostic; it does NOT contain agent-specific logic like prompting
strategies. Instead, it defines the world that agents will inhabit.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Any, Optional, Type

import jax.random as jr
import jax.numpy as jnp

from core.agents import Agent # Assumes a core Agent interface exists

# --- Data Structures for Environment Components ---

@dataclass(frozen=True)
class CropConfig:
    """
    Configuration for a single crop asset within the environment.
    Defines its name and underlying yield dynamics over time.
    """
    name: str
    true_expected_yields_per_round: List[float]
    yield_beta_dist_alpha: float = 2.0
    yield_beta_dist_beta: float = 2.0

@dataclass(frozen=True)
class PortfolioStrategyConfig:
    """
    Configuration for a single, fixed portfolio allocation strategy.
    This represents a discrete policy choice available to the agents.
    """
    name: str
    weights: List[float]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        weight_sum = sum(self.weights)
        if not jnp.isclose(weight_sum, 1.0):
            raise ValueError(f"Portfolio '{self.name}' weights must sum to 1.0, got {weight_sum}")
        if any(w < 0 for w in self.weights):
            raise ValueError(f"Portfolio '{self.name}' weights cannot be negative.")

@dataclass(frozen=True)
class MarketConfig:
    """
    Configuration for the environment's prediction market.
    This defines the quality of information available to agents.
    """
    prediction_noise_sigma: float = 0.25

@dataclass(frozen=True)
class CognitiveResourceConfig:
    """
    Configuration for the environment's handling of agent cognitive resources.
    This is an environmental parameter that determines how information is
    filtered or distorted for different agent roles.
    """
    cognitive_resources_delegate: int = 80
    cognitive_resources_voter: int = 20

@dataclass(frozen=True)
class AgentSettingsConfig:
    """
    Configuration for the composition of the agent population.
    The environment uses this to know how many of each agent type to create.
    """
    adversarial_proportion_total: float = 0.2
    # The framing is passed to agents during their creation.
    adversarial_framing: str = "antifragility"

@dataclass(frozen=True)
class ResourceConfig:
    """Configuration for the environment's collective resource dynamics."""
    initial_amount: float = 100.0
    threshold: float = 20.0

# --- Master Environment Configuration ---

@dataclass(frozen=True)
class PortfolioDemocracyConfig:
    """
    A self-contained, agent-agnostic configuration for the simulation environment.
    """
    # === Core Simulation & Institutional Parameters ===
    mechanism: Literal["PDD", "PRD", "PLD"]
    num_agents: int
    num_delegates: int
    num_rounds: int
    seed: int

    # === Agent Population Specification (by class) ===
    # This is the key change: we specify the agent *types*, not their logic.
    aligned_agent_class: Type[Agent]
    adversarial_agent_class: Type[Agent]
    
    # === Environment Component Configurations ===
    crops: List[CropConfig]
    portfolios: List[PortfolioStrategyConfig]
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    agent_settings: AgentSettingsConfig = field(default_factory=AgentSettingsConfig)
    cognitive_resource_settings: CognitiveResourceConfig = field(default_factory=CognitiveResourceConfig)
    market_settings: MarketConfig = field(default_factory=MarketConfig)

    # === PRD-Specific Institutional Rules ===
    prd_election_term_length: int = 4
    prd_num_representatives_to_elect: Optional[int] = None

    # === LLM Settings (for the environment to pass to agent factories) ===
    llm_model: Optional[str] = 'openai/gpt-4o-mini'

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.num_delegates > self.num_agents:
            raise ValueError(f"Number of delegates ({self.num_delegates}) cannot exceed num of agents ({self.num_agents})")
        
        if self.mechanism == "PRD" and self.prd_num_representatives_to_elect is None:
            object.__setattr__(self, 'prd_num_representatives_to_elect', self.num_delegates)


# --- Configuration Factory Functions ---

def create_thesis_baseline_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    adversarial_proportion_total: float,
    aligned_agent_class: Type[Agent],
    adversarial_agent_class: Type[Agent],
    seed: int = 42,
    # Other parameters can be exposed for sweeping
    num_total_agents: int = 10,
    num_delegates_baseline: int = 4,
    num_simulation_rounds: int = 30,
    prediction_market_sigma: float = 0.25,
    num_crops_config: int = 3,
    num_portfolios_config: int = 5,
    crop_yield_variance_multiplier: float = 0.2
) -> PortfolioDemocracyConfig:
    """
    Creates a standardized baseline configuration for the "Noisy Information" environment.
    This function procedurally generates a consistent and reproducible world.
    """
    yield_key = jr.PRNGKey(seed + 1000)

    # Generate crop configurations with controlled randomness
    default_crops = []
    crop_names = [f"Crop{chr(65+i)}" for i in range(num_crops_config)]
    for i in range(num_crops_config):
        crop_key, yield_key = jr.split(yield_key)
        # Generate stable but varied yield patterns
        base_yield = 1.0 + (jr.uniform(crop_key) * 0.1 - 0.05) # Centered around 1.0
        random_yields = jr.normal(crop_key, shape=(100,)) * crop_yield_variance_multiplier + base_yield
        default_crops.append(CropConfig(
            name=crop_names[i],
            true_expected_yields_per_round=list(random_yields)
        ))

    # Generate a standard set of portfolio configurations
    default_portfolios = []
    if num_crops_config > 0:
        # P1: Equal weight portfolio
        default_portfolios.append(PortfolioStrategyConfig(
            name="P1_EqualWeight",
            weights=[1.0/num_crops_config] * num_crops_config,
            description=f"A balanced portfolio with equal allocation across all {num_crops_config} crops."
        ))
        
        # Add focused portfolios for each crop
        for i in range(min(num_crops_config, num_portfolios_config - 1)):
            weights = [0.0] * num_crops_config
            weights[i] = 1.0
            default_portfolios.append(PortfolioStrategyConfig(
                name=f"P{i+2}_{crop_names[i]}_Focus",
                weights=weights,
                description=f"A specialized portfolio focusing entirely on {crop_names[i]}."
            ))

    return PortfolioDemocracyConfig(
        mechanism=mechanism,
        num_agents=num_total_agents,
        num_delegates=num_delegates_baseline,
        num_adversarial=int(round(num_total_agents * adversarial_proportion_total)),
        num_rounds=num_simulation_rounds,
        seed=seed,
        
        # Pass agent classes directly
        aligned_agent_class=aligned_agent_class,
        adversarial_agent_class=adversarial_agent_class,

        # Pass generated environment components
        crops=default_crops,
        portfolios=default_portfolios,
        market_settings=MarketConfig(prediction_noise_sigma=prediction_market_sigma),
        
        # Use default resource and cognitive settings
        resources=ResourceConfig(),
        cognitive_resource_settings=CognitiveResourceConfig(),
        agent_settings=AgentSettingsConfig(adversarial_proportion_total=adversarial_proportion_total),
    )

if __name__ == "__main__":
    # --- Example of how to use this new, cleaner configuration system ---
    
    # We would need to import the agent classes first
    # This demonstrates the new dependency: the experiment runner needs to know about agents.
    # from environments.democracy.agents.llm_agents import AlignedHeuristicAgent, RedTeamAgent
    
    # Mock classes for demonstration if run standalone
    class MockAlignedAgent(Agent): pass
    class MockAdversarialAgent(Agent): pass
    
    print("--- Creating a Baseline PLD Configuration with Specified Agent Classes ---")
    
    baseline_config = create_thesis_baseline_config(
        mechanism="PLD",
        adversarial_proportion_total=0.2,
        aligned_agent_class=MockAlignedAgent,
        adversarial_agent_class=MockAdversarialAgent,
        seed=123
    )

    print(f"Mechanism: {baseline_config.mechanism}")
    print(f"Number of Agents: {baseline_config.num_agents}")
    print(f"Number of Adversaries: {baseline_config.num_adversarial}")
    print(f"Aligned Agent Type: {baseline_config.aligned_agent_class.__name__}")
    print(f"Adversarial Agent Type: {baseline_config.adversarial_agent_class.__name__}")
    print("\nPortfolio Options:")
    for portfolio in baseline_config.portfolios:
        print(f"  - {portfolio.name}: {portfolio.description}")