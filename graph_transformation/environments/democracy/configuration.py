# domains/democracy/configuration.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional, Union

@dataclass(frozen=True)
class InformationEnvironmentConfig:
    """Configuration for information distribution among agents."""
    distribution_type: Literal["uniform", "concentrated", "polarized"]
    expertise_concentration: float = 0.2  # Proportion of agents with high expertise
    expertise_asymmetry: float = 3.0  # Ratio between expert/non-expert information quality
    noise_level: float = 0.1  # Standard deviation of noise in signals

@dataclass(frozen=True)
class AdversarialConfig:
    """Configuration for adversarial agents."""
    proportion: float = 0.0  # 0.0-0.5 range
    introduction: Literal["immediate", "gradual"] = "immediate"
    strategy: Literal["direct", "deceptive", "strategic"] = "direct"
    targeting: Literal["random", "influence_based"] = "random"
    introduction_schedule: Optional[Dict[int, float]] = None  # Round -> proportion mapping

@dataclass(frozen=True)
class ResourceConfig:
    """Configuration for resource dynamics."""
    initial_amount: float = 1000.0
    threshold: float = 500.0  # Survival threshold
    options: List[str] = field(default_factory=lambda: ["Option A", "Option B", "Option C"])
    yield_volatility: Literal["stable", "variable"] = "stable"
    yield_distributions: Optional[Dict[str, Dict[str, float]]] = None

@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for agent network topology."""
    topology: Literal["random", "small_world", "scale_free"] = "small_world"
    avg_connections: int = 4
    clustering_coefficient: float = 0.3
    community_structure: bool = False
    num_communities: int = 2

@dataclass(frozen=True)
class PredictionMarketConfig:
    """Configuration for prediction markets."""
    enabled: bool = True
    accuracy: float = 0.7  # Base accuracy of predictions
    visibility: Literal["public", "representatives", "delegated"] = "public"
    update_frequency: int = 1  # Every N rounds

@dataclass(frozen=True)
class DemocraticEnvironmentConfig:
    """Full configuration for democratic environment simulation."""
    mechanism: Literal["PDD", "PRD", "PLD"] = "PLD"
    num_agents: int = 100
    num_rounds: int = 30
    seed: int = 42
    information: InformationEnvironmentConfig = field(default_factory=InformationEnvironmentConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    prediction_market: PredictionMarketConfig = field(default_factory=PredictionMarketConfig)
    
    # Advanced execution options
    jit_compile: bool = True
    num_trials: int = 1
    track_delegation_metrics: bool = True
    track_resource_metrics: bool = True
    track_information_metrics: bool = True

@dataclass(frozen=True)
class PortfolioConfig:
    """Configuration for portfolio options"""
    portfolio_strategies: List[str] = field(
        default_factory=lambda: ["Conservative", "Balanced", "Aggressive", "Contrarian", "Market-Weighted"]
    )
    use_tokens: bool = True
    token_distribution: Dict[str, int] = field(
        default_factory=lambda: {
            "low_capacity": 150,
            "medium_capacity": 300,
            "high_capacity": 500
        }
    )