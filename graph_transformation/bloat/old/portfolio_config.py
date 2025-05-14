# environments/democracy/portfolio_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal

@dataclass(frozen=True)
class PortfolioSpec:
    """Domain-specific portfolio configuration."""
    crop_names: List[str]
    alpha_values: List[float]
    beta_values: List[float]
    volatilities: List[float]
    expected_returns: List[List[float]] = field(default_factory=list)  # Optional historical data
    
    # No environment-specific execution details here