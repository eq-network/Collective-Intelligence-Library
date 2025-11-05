"""
General visualization canvas.

Environment-agnostic - works with any GraphState and VizConfig.
Supports both view mode and interactive edit mode.
"""
from typing import Protocol, Optional, Callable
from core.graph import GraphState


class VizConfig(Protocol):
    """Protocol for environment-specific visualization configuration."""

    def get_node_positions(self, state: GraphState) -> dict:
        """Return {node_id: (x, y)} positions."""
        ...

    def get_node_colors(self, state: GraphState) -> dict:
        """Return {node_id: color_string} for each node."""
        ...

    def get_node_labels(self, state: GraphState) -> dict:
        """Return {node_id: label_string} for each node."""
        ...

    def get_edges(self, state: GraphState) -> list:
        """Return [(from_id, to_id), ...] edges to display."""
        ...


class Canvas:
    """
    General canvas for visualizing GraphState.

    Delegates environment-specific display logic to VizConfig.
    Supports interactive edit mode for building/modifying graphs.
    """

    def __init__(self, viz_config: VizConfig, edit_mode: bool = False):
        """
        Initialize canvas with environment-specific config.

        Args:
            viz_config: Configuration defining how to visualize this environment
            edit_mode: If True, enable interactive editing
        """
        self.viz_config = viz_config
        self.renderer = None
        self.edit_mode = edit_mode
        self.current_state: Optional[GraphState] = None
        self.on_state_change: Optional[Callable[[GraphState], None]] = None

    def render(self, state: GraphState):
        """
        Render a GraphState.

        Args:
            state: GraphState to visualize
        """
        from studio.renderer import TkinterRenderer

        self.current_state = state

        if self.renderer is None:
            self.renderer = TkinterRenderer()

            # Set up interaction handlers if in edit mode
            if self.edit_mode:
                self.renderer.on_click = self._handle_click
                self.renderer.on_drag = self._handle_drag

        # Get visualization data from config
        positions = self.viz_config.get_node_positions(state)
        colors = self.viz_config.get_node_colors(state)
        labels = self.viz_config.get_node_labels(state)
        edges = self.viz_config.get_edges(state)

        # Render
        self.renderer.render(
            positions=positions,
            colors=colors,
            labels=labels,
            edges=edges
        )

    def _handle_click(self, x: float, y: float):
        """
        Handle click event in edit mode.

        Click on empty space: add node
        Click on node: select/deselect

        Args:
            x, y: Click position in [0, 1] normalized coordinates
        """
        if not self.edit_mode or self.current_state is None:
            return

        from studio.graph_editor import add_node

        # Check if clicked near existing node
        positions = self.viz_config.get_node_positions(self.current_state)
        clicked_node = None
        for node_id, (nx, ny) in positions.items():
            dist = ((x - nx) ** 2 + (y - ny) ** 2) ** 0.5
            if dist < 0.05:  # Within 5% of canvas
                clicked_node = node_id
                break

        if clicked_node is None:
            # Add new node at click position
            # (Position will be recalculated by VizConfig)
            new_state = add_node(self.current_state)
            self._update_state(new_state)

    def _handle_drag(self, from_x: float, from_y: float, to_x: float, to_y: float):
        """
        Handle drag event in edit mode.

        Drag from node to node: add edge
        Drag from node to empty: do nothing

        Args:
            from_x, from_y: Start position in [0, 1] normalized coordinates
            to_x, to_y: End position in [0, 1] normalized coordinates
        """
        if not self.edit_mode or self.current_state is None:
            return

        from studio.graph_editor import add_edge

        # Find nodes at start and end positions
        positions = self.viz_config.get_node_positions(self.current_state)

        from_node = None
        to_node = None

        for node_id, (nx, ny) in positions.items():
            dist_from = ((from_x - nx) ** 2 + (from_y - ny) ** 2) ** 0.5
            if dist_from < 0.05:
                from_node = node_id

            dist_to = ((to_x - nx) ** 2 + (to_y - ny) ** 2) ** 0.5
            if dist_to < 0.05:
                to_node = node_id

        if from_node is not None and to_node is not None and from_node != to_node:
            # Get first adjacency matrix name (for simplicity)
            if self.current_state.adj_matrices:
                rel_name = next(iter(self.current_state.adj_matrices.keys()))
                new_state = add_edge(self.current_state, from_node, to_node, rel_name)
                self._update_state(new_state)

    def _update_state(self, new_state: GraphState):
        """
        Update current state and notify callback.

        Args:
            new_state: Updated GraphState
        """
        self.current_state = new_state

        if self.on_state_change:
            self.on_state_change(new_state)

        # Re-render
        self.render(new_state)
