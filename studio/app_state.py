"""
AppState: Shared application state across screens.

This is the "save game" - everything needed to persist state
between screen transitions.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING

from core.graph import GraphState

# Avoid circular import - only import for type checking
if TYPE_CHECKING:
    from studio.edit_session import EditSession

# Type alias for transform factory function
TransformFactory = Callable[[int, float], Callable[[GraphState], GraphState]]


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""
    name: str
    description: str
    default_num_agents: int = 10
    default_num_rounds: int = 50
    # These will be callables that create the actual objects
    # None means not yet implemented
    create_initial_graph: Optional[Callable[..., GraphState]] = None
    create_transforms: Optional[TransformFactory] = None


@dataclass
class AppState:
    """
    Shared application state that persists across screen transitions.

    Mutable by design - this is the UI coordinator pattern.
    """
    # Current scenario selection (from Step 2)
    selected_scenario: Optional[ScenarioConfig] = None

    # Configuration parameters (from Step 3)
    num_agents: int = 10
    num_rounds: int = 50
    adversarial_ratio: float = 0.0
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    # Graph editing state (from Step 4)
    edit_session: Optional["EditSession"] = None

    # Simulation results (from Step 5)
    simulation_history: List[GraphState] = field(default_factory=list)
    current_round: int = 0
    is_running: bool = False

    # Export settings (for Step 6)
    visualization_mode: str = "messages"  # "messages", "trust", "market", "metrics"
