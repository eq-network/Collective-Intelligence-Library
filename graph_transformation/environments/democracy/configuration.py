# environments/democracy/portfolio_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union, Any
import jax.numpy as jnp

@dataclass(frozen=True)
class PortfolioStrategyConfig:
    """
    Configuration for a single portfolio allocation strategy.
    
    Attributes:
        name: Unique identifier for the strategy
        weights: Asset allocation weights (must sum to 1.0)
        description: Human-readable description of the strategy
        risk_level: Qualitative risk assessment
        metadata: Additional strategy-specific parameters
    """
    name: str
    weights: List[float]
    description: str = ""
    risk_level: Literal["low", "medium", "high"] = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate portfolio weights."""
        # Ensure weights sum to approximately 1.0 (allowing for floating point precision)
        weight_sum = sum(self.weights)
        if not 0.99 <= weight_sum <= 1.01:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {weight_sum}")

@dataclass(frozen=True)
class MarketConfig:
    """
    Configuration for prediction market and returns.
    
    Attributes:
        initial_predictions: Initial prediction market values
        true_returns: Actual return values (may differ from predictions)
        accuracy: Base accuracy of prediction market (0.0-1.0)
        noise_level: Noise in prediction market signals
        update_frequency: How often predictions update (in rounds)
    """
    initial_predictions: List[float] = field(default_factory=list)
    true_returns: List[float] = field(default_factory=list)
    accuracy: float = 0.7
    noise_level: float = 0.15
    update_frequency: int = 1
    
    def __post_init__(self):
        """Validate market parameters."""
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"Accuracy must be between 0.0 and 1.0, got {self.accuracy}")
        if self.noise_level < 0.0:
            raise ValueError(f"Noise level cannot be negative, got {self.noise_level}")

@dataclass(frozen=True)
class TokenBudgetConfig:
    """
    Configuration for token economy.
    
    Attributes:
        enabled: Whether token economy is active
        capacity_levels: Token allocations by capacity level
        capacity_distribution: Agent count by capacity level
        operation_costs: Token costs for different operations
        refresh_period: Rounds between token refreshes
    """
    enabled: bool = True
    capacity_levels: Dict[str, int] = field(
        default_factory=lambda: {
            "low": 150,
            "medium": 300, 
            "high": 500
        }
    )
    capacity_distribution: Dict[str, int] = field(
        default_factory=lambda: {
            "low_count": 3,
            "medium_count": 4,
            "high_count": 3
        }
    )
    operation_costs: Dict[str, int] = field(
        default_factory=lambda: {
            "basic_analysis": 10,
            "detailed_analysis": 30,
            "portfolio_comparison": 50
        }
    )
    refresh_period: int = 5
    
    def __post_init__(self):
        """Validate token budget parameters."""
        if self.refresh_period < 1:
            raise ValueError(f"Refresh period must be at least 1, got {self.refresh_period}")
        for level, amount in self.capacity_levels.items():
            if amount < 0:
                raise ValueError(f"Token budget cannot be negative, got {amount} for {level}")

@dataclass(frozen=True)
class AgentConfig:
    """
    Configuration for agent population.
    
    Attributes:
        adversarial_proportion: Fraction of agents that are adversarial
        adversarial_introduction: How adversarial agents are introduced
        expertise_distribution: Parameters of expertise beta distribution
    """
    adversarial_proportion: float = 0.2
    adversarial_introduction: Literal["immediate", "gradual"] = "immediate"
    expertise_distribution: Dict[str, float] = field(
        default_factory=lambda: {"alpha": 2.0, "beta": 3.0}
    )
    
    def __post_init__(self):
        """Validate agent parameters."""
        if not 0.0 <= self.adversarial_proportion <= 1.0:
            raise ValueError(
                f"Adversarial proportion must be between 0.0 and 1.0, got {self.adversarial_proportion}"
            )

@dataclass(frozen=True)
class ResourceConfig:
    """
    Configuration for resource dynamics.
    
    Attributes:
        initial_amount: Starting resource amount
        threshold: Minimum survival threshold
        asset_names: Names of underlying assets
    """
    initial_amount: float = 100.0
    threshold: float = 20.0
    asset_names: List[str] = field(
        default_factory=lambda: ["Wheat", "Corn", "Fungus"]
    )
    
    def __post_init__(self):
        """Validate resource parameters."""
        if self.initial_amount <= 0.0:
            raise ValueError(f"Initial resources must be positive, got {self.initial_amount}")
        if self.threshold < 0.0:
            raise ValueError(f"Resource threshold cannot be negative, got {self.threshold}")
        if self.threshold >= self.initial_amount:
            raise ValueError(
                f"Threshold ({self.threshold}) should be less than initial amount ({self.initial_amount})"
            )

@dataclass(frozen=True)
class PortfolioDemocracyConfig:
    """
    Master configuration for portfolio democracy simulation.
    
    Attributes:
        mechanism: Democratic mechanism type
        num_agents: Total number of agents
        num_rounds: Maximum simulation rounds
        seed: Random seed for reproducibility
        resources: Resource dynamics configuration
        token_system: Token economy configuration
        market: Prediction market configuration
        agents: Agent population configuration
        strategies: Portfolio strategies configuration
        track_metrics: Which metrics to track
    """
    mechanism: Literal["PDD", "PRD", "PLD"] = "PLD"
    num_agents: int = 10
    num_rounds: int = 15
    seed: int = 42
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    token_system: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    strategies: Dict[str, PortfolioStrategyConfig] = field(default_factory=dict)
    track_metrics: Dict[str, bool] = field(
        default_factory=lambda: {
            "resource_history": True,
            "delegation_patterns": True,
            "preference_evolution": True
        }
    )
    
    def __post_init__(self):
        """Validate complete configuration."""
        # Validate number of agents
        if self.num_agents <= 0:
            raise ValueError(f"Number of agents must be positive, got {self.num_agents}")
            
        # Validate number of rounds
        if self.num_rounds <= 0:
            raise ValueError(f"Number of rounds must be positive, got {self.num_rounds}")
            
        # If strategies empty, apply defaults during initialization rather than here
        # since dataclass is frozen and we can't modify strategies directly
    
    def get_default_strategies(self) -> Dict[str, PortfolioStrategyConfig]:
        """
        Create default portfolio strategies if none provided.
        
        Returns:
            Dictionary of strategy configs with default allocations
        """
        if self.strategies:
            return self.strategies
            
        # Ensure we have asset names to work with
        asset_count = len(self.resources.asset_names)
        if asset_count == 0:
            asset_count = 3  # Default fallback
            
        # Create default strategies
        return {
            "Conservative": PortfolioStrategyConfig(
                name="Conservative",
                weights=[0.6, 0.3, 0.1] if asset_count == 3 else [1.0/asset_count] * asset_count,
                description="Low risk, low return strategy focusing on stability",
                risk_level="low",
                metadata={"expected_volatility": 0.1}
            ),
            "Balanced": PortfolioStrategyConfig(
                name="Balanced",
                weights=[0.4, 0.3, 0.3] if asset_count == 3 else [1.0/asset_count] * asset_count,
                description="Moderate risk and return with diversified allocation",
                risk_level="medium",
                metadata={"expected_volatility": 0.2}
            ),
            "Aggressive": PortfolioStrategyConfig(
                name="Aggressive",
                weights=[0.2, 0.1, 0.7] if asset_count == 3 else [1.0/asset_count] * asset_count,
                description="High risk, high potential return focusing on growth",
                risk_level="high",
                metadata={"expected_volatility": 0.4}
            ),
            "Contrarian": PortfolioStrategyConfig(
                name="Contrarian",
                weights=[0.3, 0.6, 0.1] if asset_count == 3 else [1.0/asset_count] * asset_count,
                description="Betting against market consensus",
                risk_level="medium",
                metadata={"expected_volatility": 0.3}
            ),
            "Market-Weighted": PortfolioStrategyConfig(
                name="Market-Weighted",
                weights=[0.35, 0.15, 0.5] if asset_count == 3 else [1.0/asset_count] * asset_count,
                description="Allocation proportional to market capitalization",
                risk_level="medium",
                metadata={"expected_volatility": 0.25}
            )
        }


# Factory function for creating complete configuration with defaults
def create_default_portfolio_config(
    mechanism: Literal["PDD", "PRD", "PLD"] = "PLD",
    num_agents: int = 10,
    adversarial_proportion: float = 0.2,
    initial_resources: float = 115.0
) -> PortfolioDemocracyConfig:
    """
    Create a default portfolio democracy configuration.
    
    Args:
        mechanism: Democratic mechanism type
        num_agents: Number of agents in simulation
        adversarial_proportion: Fraction of adversarial agents
        initial_resources: Starting resource amount
        
    Returns:
        Complete configuration with reasonable defaults
    """
    # Create baseline configuration
    config = PortfolioDemocracyConfig(
        mechanism=mechanism,
        num_agents=num_agents,
        resources=ResourceConfig(
            initial_amount=initial_resources,
            threshold=20.0,
            asset_names=["Wheat", "Corn", "Fungus"]
        ),
        agents=AgentConfig(
            adversarial_proportion=adversarial_proportion
        ),
        market=MarketConfig(
            initial_predictions=[1.15, 0.6, 1.5],
            true_returns=[1.2, 0.5, 1.7]
        )
    )
    
    # Add default strategies (can't be done directly in __post_init__ due to frozen dataclass)
    strategies = config.get_default_strategies()
    
    # Create new config with strategies included
    return PortfolioDemocracyConfig(
        mechanism=config.mechanism,
        num_agents=config.num_agents,
        num_rounds=config.num_rounds,
        seed=config.seed,
        resources=config.resources,
        token_system=config.token_system,
        market=config.market,
        agents=config.agents,
        strategies=strategies,
        track_metrics=config.track_metrics
    )


# Example Usage
if __name__ == "__main__":
    # Create default configuration
    config = create_default_portfolio_config()
    
    # Access configuration components
    print(f"Mechanism: {config.mechanism}")
    print(f"Initial resources: {config.resources.initial_amount}")
    print(f"Adversarial proportion: {config.agents.adversarial_proportion}")
    
    # List available strategies
    for name, strategy in config.strategies.items():
        print(f"Strategy: {name}, Weights: {strategy.weights}, Risk: {strategy.risk_level}")