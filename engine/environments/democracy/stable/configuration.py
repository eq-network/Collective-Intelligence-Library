# environments/democracy/stable/configuration.py
"""
Defines the configuration for the "Stable" Portfolio Democracy environment.

This module is responsible for describing the "world" of the simulation,
including its institutional rules and economic components. It uses factory
functions to procedurally generate complex but reproducible environments.

Crucially, this file is AGENT-AGNOSTIC. It does not contain any logic
related to agent decision-making or prompting.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Any, Optional, Type

import jax.random as jr
import jax.numpy as jnp

from core.agents import Agent # Abstract dependency

# --- Data Structures for Environment Components ---

@dataclass(frozen=True)
class CropConfig:
    """Defines a single crop asset, its name, and its underlying yield dynamics."""
    name: str
    true_expected_yields_per_round: List[float]

@dataclass(frozen=True)
class PortfolioStrategyConfig:
    """Defines a single, fixed investment portfolio available for selection."""
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
class ParticipationConfig:
    """Configures the stochastic participation model for agents."""
    voter_participation_rate: float = 1.0
    delegate_participation_rate: float = 1.0
    temporal_correlation_strength: float = 0.7

@dataclass(frozen=True)
class AgentSettingsConfig:
    """Configures high-level properties of the agent population makeup."""
    adversarial_proportion_total: float
    adversarial_framing: str

# --- Master Environment Configuration ---

@dataclass(frozen=True)
class StablePortfolioDemocracyConfig:
    """
    A self-contained, agent-agnostic configuration for the Stable environment.
    This object is the "blueprint" of the world, produced by the factory function.
    """
    # Core simulation and institutional parameters
    mechanism: Literal["PDD", "PRD", "PLD"]
    num_agents: int
    num_delegates: int
    num_rounds: int
    seed: int

    # Agent Population Specification (by class path)
    aligned_agent_class_path: str
    adversarial_agent_class_path: str

    # Environment components
    crops: List[CropConfig]
    portfolios: List[PortfolioStrategyConfig]
    participation_settings: ParticipationConfig
    agent_settings: AgentSettingsConfig
    
    initial_resources: float = 100.0
    survival_threshold: float = 20.0
    llm_model: Optional[str] = 'openai/gpt-4o-mini'

    # PRD-specific rules
    prd_election_term_length: int = 4
    prd_num_representatives_to_elect: Optional[int] = None

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.num_delegates > self.num_agents:
            raise ValueError("Number of delegates cannot exceed number of agents.")
        if self.mechanism == "PRD" and self.prd_num_representatives_to_elect is None:
            object.__setattr__(self, 'prd_num_representatives_to_elect', self.num_delegates)

# --- Factory Function for Procedural Environment Generation ---

def create_stable_democracy_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    adversarial_proportion_total: float,
    seed: int,
    # Allow overriding agent classes, with sensible defaults
    aligned_agent_path: str = 'environments.democracy.agents.llm_agents.AlignedHeuristicAgent',
    adversarial_agent_path: str = 'environments.democracy.agents.hardcoded.HardcodedAdversarialAgent',
    # Allow overriding key environment parameters for sweeps
    num_agents: int = 10,
    num_delegates: int = 4,
    num_rounds: int = 30,
    num_crops: int = 3,
    num_portfolios: int = 4,
    adversarial_framing: str = "antifragility",
    delegate_participation_rate: float = 1.0,
    voter_participation_rate: float = 1.0,
    llm_model: Optional[str] = 'openai/gpt-4o-mini'
) -> StablePortfolioDemocracyConfig:
    """
    Factory to create a standardized Stable environment configuration.
    
    This function contains the detailed, procedural logic for generating
    the crops and portfolios, preserving the original experimental design.
    """
    
    # 1. Procedurally generate Crop configurations
    # In a stable environment, yields are fixed and known.
    # We create a simple but non-trivial spread of yields.
    yield_key = jr.PRNGKey(seed)
    base_yields = jnp.linspace(0.9, 1.1, num_crops) # e.g., for 3 crops: [0.9, 1.0, 1.1]
    shuffled_yields = jr.permutation(yield_key, base_yields)

    crops = [
        CropConfig(
            name=f"Crop{chr(65+i)}",
            # In stable env, the yield is the same every round
            true_expected_yields_per_round=[float(shuffled_yields[i])] * num_rounds
        ) for i in range(num_crops)
    ]

    # 2. Procedurally generate Portfolio configurations
    # This preserves your original, detailed portfolio generation logic.
    portfolios = []
    if num_crops > 0:
        # P1: Equal weight portfolio
        portfolios.append(PortfolioStrategyConfig(
            name="P_Equal",
            weights=[1.0/num_crops] * num_crops,
            description=f"A balanced portfolio with equal allocation across all {num_crops} crops."
        ))

        # P2 to P(num_crops+1): Focused portfolios
        for i in range(min(num_crops, num_portfolios - 1)):
            weights = [0.0] * num_crops
            weights[i] = 1.0
            portfolios.append(PortfolioStrategyConfig(
                name=f"P_Focus_{crops[i].name}",
                weights=weights,
                description=f"A specialized portfolio focusing entirely on {crops[i].name}."
            ))

        # P(num_crops+2) onwards: Tactical/mixed portfolios
        additional_needed = num_portfolios - len(portfolios)
        tactical_key = jr.PRNGKey(seed + 1)
        for i in range(additional_needed):
            tactical_key, sub_key = jr.split(tactical_key)
            # Generate random weights that sum to 1
            random_weights = jr.dirichlet(sub_key, alpha=jnp.ones(num_crops)).tolist()
            portfolios.append(PortfolioStrategyConfig(
                name=f"P_Tactical_{i+1}",
                weights=random_weights,
                description=f"A mixed tactical allocation strategy."
            ))

    elif num_portfolios > 0:
        portfolios.append(PortfolioStrategyConfig(
            name="P_NoOp", weights=[], description="No-operation portfolio."
        ))

    # 3. Assemble the final configuration object
    return StablePortfolioDemocracyConfig(
        mechanism=mechanism,
        num_agents=num_agents,
        num_delegates=num_delegates,
        num_rounds=num_rounds,
        seed=seed,
        
        # Agent class paths are passed in directly
        aligned_agent_class_path=aligned_agent_path,
        adversarial_agent_class_path=adversarial_agent_path,

        # Pass the procedurally generated components
        crops=crops,
        portfolios=portfolios[:num_portfolios], # Ensure correct number of portfolios

        # Assemble component configs
        participation_settings=ParticipationConfig(
            voter_participation_rate=voter_participation_rate,
            delegate_participation_rate=delegate_participation_rate
        ),
        agent_settings=AgentSettingsConfig(
            adversarial_proportion_total=adversarial_proportion_total,
            adversarial_framing=adversarial_framing
        ),
        llm_model=llm_model
    )

if __name__ == "__main__":
    # --- Example of how to use this refactored configuration system ---
    print("--- Creating a Stable PLD Configuration with Defaults ---")
    
    # The experiment runner only needs to know the factory function and its core args.
    # The factory handles the complex defaults and procedural generation.
    stable_pld_config = create_stable_democracy_config(
        mechanism="PLD",
        adversarial_proportion_total=0.3,
        seed=42
    )
    
    print(f"Mechanism: {stable_pld_config.mechanism}")
    print(f"LLM Model: {stable_pld_config.llm_model}")
    print(f"Aligned Agent: {stable_pld_config.aligned_agent_class_path}")
    print(f"Adversarial Agent: {stable_pld_config.adversarial_agent_class_path} (Default is Hardcoded)")
    
    print("\nGenerated Portfolio Options:")
    for portfolio in stable_pld_config.portfolios:
        # Calculate the actual yield for this stable portfolio
        yield_val = sum(w * c.true_expected_yields_per_round[0] for w, c in zip(portfolio.weights, stable_pld_config.crops))
        print(f"  - {portfolio.name}: Weights={portfolio.weights}, True Yield={yield_val:.3f}x")