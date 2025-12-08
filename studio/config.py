"""
Centralized configuration for the Studio application.

Consolidates hardcoded values like colors, dimensions, rates, etc.
into a single configuration module.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SimulationDefaults:
    """Default values for simulation parameters."""
    consumption_rate_normal: float = 0.05
    consumption_rate_adversarial: float = 0.15
    growth_rate: float = 1.08
    network_density: float = 0.3
    default_num_agents: int = 10
    default_num_rounds: int = 50
    initial_resources: float = 100.0


@dataclass
class CanvasConfig:
    """Canvas dimensions and styling."""
    editor_width: int = 800
    editor_height: int = 500
    simulation_graph_width: int = 500
    simulation_graph_height: int = 350
    metrics_panel_width: int = 280
    metrics_panel_height: int = 350
    margin: int = 50
    bg_color: str = "white"
    metrics_bg_color: str = "#f8f8f8"


@dataclass
class NodeColors:
    """Color scheme for node types."""
    normal: str = "#87CEEB"       # Sky blue - normal agents
    adversarial: str = "#FF6B6B"  # Red - adversarial agents
    market: str = "#FFA500"       # Orange - market mechanism
    resource: str = "#90EE90"     # Light green - resource depot
    delegate: str = "#FFD700"     # Gold - delegate/leader
    default: str = "#D3D3D3"      # Gray - unknown type

    def by_type(self) -> Dict[int, str]:
        """Get color mapping by type ID."""
        return {
            0: self.normal,
            1: self.adversarial,
            2: self.market,
            3: self.resource,
        }


@dataclass
class ChartColors:
    """Colors for charts and metrics visualization."""
    total_resources: str = "#4CAF50"
    normal_line: str = "#87CEEB"
    adversarial_line: str = "#FF6B6B"
    inequality: str = "#FF9800"
    edge: str = "gray"


@dataclass
class NodeShapes:
    """Shape names for node types."""
    agent: str = "circle"
    market: str = "square"
    resource: str = "diamond"

    def by_type(self) -> Dict[int, str]:
        """Get shape mapping by type ID."""
        return {
            0: self.agent,      # Normal agents
            1: self.agent,      # Adversarial agents
            2: self.market,     # Market mechanism
            3: self.resource,   # Resource depot
        }


@dataclass
class StudioConfig:
    """
    Master configuration for the Studio application.

    Usage:
        from studio.config import studio_config
        color = studio_config.node_colors.normal
        rate = studio_config.simulation.consumption_rate_normal
    """
    simulation: SimulationDefaults = field(default_factory=SimulationDefaults)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    node_colors: NodeColors = field(default_factory=NodeColors)
    node_shapes: NodeShapes = field(default_factory=NodeShapes)
    chart_colors: ChartColors = field(default_factory=ChartColors)

    # Node radius settings
    node_radius: int = 20
    node_radius_min: int = 15
    node_radius_max: int = 30

    # Edge settings
    edge_color: str = "gray"
    edge_width: int = 2

    # Playback settings
    playback_step_interval: float = 0.5  # seconds between steps
    min_playback_speed: float = 0.25
    max_playback_speed: float = 4.0

    # History settings
    max_undo_history: int = 100


# Global singleton instance
studio_config = StudioConfig()
