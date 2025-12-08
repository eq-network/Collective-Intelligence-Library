"""
Shared graph visualization utilities for editor and simulation screens.

Centralizes node colors, shapes, positions, and rendering logic.
"""
import math
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

from core.graph import GraphState

# Import centralized config (lazy import to avoid circular deps)
def _get_studio_config():
    from studio.config import studio_config
    return studio_config


@dataclass
class VizConfig:
    """
    Configuration for graph visualization.

    Uses values from studio_config by default.

    Attributes:
        node_colors: Mapping from node type to color hex code
        node_shapes: Mapping from node type to shape name
        default_color: Fallback color for unknown node types
        default_shape: Fallback shape for unknown node types
        edge_color: Color for edges
        edge_width: Width of edge lines
        node_radius: Base radius for nodes
        margin: Margin around the graph area
    """
    node_colors: Dict[int, str] = field(
        default_factory=lambda: _get_studio_config().node_colors.by_type()
    )

    node_shapes: Dict[int, str] = field(
        default_factory=lambda: _get_studio_config().node_shapes.by_type()
    )

    default_color: str = field(
        default_factory=lambda: _get_studio_config().node_colors.default
    )
    default_shape: str = "circle"
    edge_color: str = field(default_factory=lambda: _get_studio_config().edge_color)
    edge_width: int = field(default_factory=lambda: _get_studio_config().edge_width)
    node_radius: int = field(default_factory=lambda: _get_studio_config().node_radius)
    margin: int = field(default_factory=lambda: _get_studio_config().canvas.margin)


def get_circle_layout(
    num_nodes: int,
    center: Tuple[float, float] = (0.5, 0.5),
    radius: float = 0.35
) -> Dict[int, Tuple[float, float]]:
    """
    Arrange nodes in a circle.

    Args:
        num_nodes: Number of nodes to arrange
        center: Center of the circle in normalized [0,1] coords
        radius: Radius of the circle in normalized coords

    Returns:
        Dict mapping node ID to (x, y) position
    """
    positions = {}
    for i in range(num_nodes):
        angle = 2 * math.pi * i / max(num_nodes, 1)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        positions[i] = (x, y)
    return positions


def get_node_positions(
    state: GraphState,
    custom_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    radius: float = 0.35
) -> Dict[int, Tuple[float, float]]:
    """
    Get positions for all active nodes.

    Custom positions take priority; fallback to circle layout.

    Args:
        state: Current graph state
        custom_positions: User-defined positions (from editor)
        radius: Circle layout radius for defaults

    Returns:
        Dict mapping node ID to (x, y) position
    """
    custom_positions = custom_positions or {}
    active_indices = state.get_active_indices()
    positions = {}

    for idx, node_id in enumerate(active_indices):
        node_id_int = int(node_id)

        if node_id_int in custom_positions:
            positions[node_id_int] = custom_positions[node_id_int]
        else:
            # Default: arrange in circle
            n = len(active_indices)
            angle = 2 * math.pi * idx / max(n, 1)
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            positions[node_id_int] = (x, y)

    return positions


def get_node_colors(
    state: GraphState,
    config: Optional[VizConfig] = None
) -> Dict[int, str]:
    """
    Get colors for all active nodes based on their type.

    Args:
        state: Current graph state
        config: Visualization configuration

    Returns:
        Dict mapping node ID to color hex code
    """
    config = config or VizConfig()
    colors = {}
    active_indices = state.get_active_indices()

    for node_id in active_indices:
        node_id_int = int(node_id)
        node_type = int(state.node_types[node_id_int])
        colors[node_id_int] = config.node_colors.get(node_type, config.default_color)

    return colors


def get_node_shapes(
    state: GraphState,
    config: Optional[VizConfig] = None
) -> Dict[int, str]:
    """
    Get shapes for all active nodes based on their type.

    Args:
        state: Current graph state
        config: Visualization configuration

    Returns:
        Dict mapping node ID to shape name ("circle", "square", "diamond")
    """
    config = config or VizConfig()
    shapes = {}
    active_indices = state.get_active_indices()

    for node_id in active_indices:
        node_id_int = int(node_id)
        node_type = int(state.node_types[node_id_int])
        shapes[node_id_int] = config.node_shapes.get(node_type, config.default_shape)

    return shapes


def get_node_labels(state: GraphState) -> Dict[int, str]:
    """
    Get labels for all active nodes (their IDs as strings).

    Args:
        state: Current graph state

    Returns:
        Dict mapping node ID to label string
    """
    return {int(i): str(int(i)) for i in state.get_active_indices()}


def get_edges(
    state: GraphState,
    connection_matrix: str = "connections"
) -> List[Tuple[int, int]]:
    """
    Get edges from an adjacency matrix.

    Args:
        state: Current graph state
        connection_matrix: Name of the adjacency matrix to use

    Returns:
        List of (from_id, to_id) tuples
    """
    edges = []

    if connection_matrix in state.adj_matrices:
        network = state.adj_matrices[connection_matrix]
        active_indices = state.get_active_indices()
        active_list = [int(i) for i in active_indices]

        for i in active_list:
            for j in active_list:
                if i < j and network[i, j] > 0:
                    edges.append((i, j))

    return edges


def normalize_to_screen(
    nx: float,
    ny: float,
    width: int,
    height: int,
    margin: int = 50
) -> Tuple[int, int]:
    """
    Convert normalized [0,1] coordinates to screen pixel coordinates.

    Args:
        nx: Normalized x coordinate
        ny: Normalized y coordinate
        width: Canvas width in pixels
        height: Canvas height in pixels
        margin: Margin in pixels

    Returns:
        (x, y) screen coordinates
    """
    return (
        int(nx * (width - 2 * margin) + margin),
        int(ny * (height - 2 * margin) + margin)
    )


def screen_to_normalized(
    x: int,
    y: int,
    width: int,
    height: int,
    margin: int = 50
) -> Tuple[float, float]:
    """
    Convert screen pixel coordinates to normalized [0,1] coordinates.

    Args:
        x: Screen x coordinate
        y: Screen y coordinate
        width: Canvas width in pixels
        height: Canvas height in pixels
        margin: Margin in pixels

    Returns:
        (nx, ny) normalized coordinates
    """
    return (
        (x - margin) / max(width - 2 * margin, 1),
        (y - margin) / max(height - 2 * margin, 1)
    )


class GraphRenderer:
    """
    Renders a GraphState to a tkinter canvas.

    Usage:
        renderer = GraphRenderer(canvas, config)
        renderer.render(state, node_positions)
    """

    def __init__(self, canvas, config: Optional[VizConfig] = None):
        """
        Initialize the renderer.

        Args:
            canvas: tkinter Canvas widget
            config: Visualization configuration
        """
        self.canvas = canvas
        self.config = config or VizConfig()

    def render(
        self,
        state: GraphState,
        custom_positions: Optional[Dict[int, Tuple[float, float]]] = None,
        selected_nodes: Optional[set] = None,
        show_labels: bool = True
    ) -> None:
        """
        Render the graph state to the canvas.

        Args:
            state: GraphState to render
            custom_positions: User-defined node positions
            selected_nodes: Set of selected node IDs (for highlighting)
            show_labels: Whether to show node labels
        """
        self.canvas.delete("all")

        width = self.canvas.winfo_width() or 800
        height = self.canvas.winfo_height() or 500
        margin = self.config.margin

        # Get visualization data
        positions = get_node_positions(state, custom_positions)
        colors = get_node_colors(state, self.config)
        shapes = get_node_shapes(state, self.config)
        labels = get_node_labels(state) if show_labels else {}
        edges = get_edges(state)

        # Scale positions to screen coordinates
        scaled = {
            nid: normalize_to_screen(pos[0], pos[1], width, height, margin)
            for nid, pos in positions.items()
        }

        # Draw edges
        self._draw_edges(edges, scaled)

        # Draw nodes
        selected_nodes = selected_nodes or set()
        for nid, (x, y) in scaled.items():
            color = colors.get(nid, self.config.default_color)
            shape = shapes.get(nid, self.config.default_shape)
            label = labels.get(nid, "")
            is_selected = nid in selected_nodes

            self._draw_node(x, y, color, shape, label, is_selected)

    def _draw_edges(
        self,
        edges: List[Tuple[int, int]],
        scaled_positions: Dict[int, Tuple[int, int]]
    ) -> None:
        """Draw edges between nodes."""
        for from_id, to_id in edges:
            if from_id in scaled_positions and to_id in scaled_positions:
                x1, y1 = scaled_positions[from_id]
                x2, y2 = scaled_positions[to_id]
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill=self.config.edge_color,
                    width=self.config.edge_width
                )

    def _draw_node(
        self,
        x: int,
        y: int,
        color: str,
        shape: str,
        label: str,
        is_selected: bool = False
    ) -> None:
        """Draw a single node at the given position."""
        r = self.config.node_radius

        # Selection highlight
        if is_selected:
            self.canvas.create_oval(
                x - r - 5, y - r - 5, x + r + 5, y + r + 5,
                outline="blue", width=3
            )

        # Draw shape
        if shape == "square":
            self.canvas.create_rectangle(
                x - r, y - r, x + r, y + r,
                fill=color, outline="black", width=2
            )
        elif shape == "diamond":
            self.canvas.create_polygon(
                x, y - r,      # top
                x + r, y,      # right
                x, y + r,      # bottom
                x - r, y,      # left
                fill=color, outline="black", width=2
            )
        else:  # circle (default)
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="black", width=2
            )

        # Label
        if label:
            self.canvas.create_text(x, y, text=label)
