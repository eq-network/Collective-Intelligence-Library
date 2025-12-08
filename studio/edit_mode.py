"""
EditMode: Interaction handler and tool coordinator for graph editing.

Manages user interactions (mouse, keyboard) and routes them to appropriate
tool handlers based on the currently active tool.

Tools:
- Select: Click to select, drag to move nodes
- Add Node: Click to create nodes
- Add Edge: Drag between nodes to create edges
- Delete: Click to delete nodes/edges
"""
from typing import Optional, Callable, Tuple
from enum import Enum
import math

from core.graph import GraphState
from studio.edit_session import EditSession
from studio.graph_editor import add_node, remove_node, add_edge, remove_edge


class Tool(Enum):
    """Available editing tools."""
    SELECT = "select"
    ADD_NODE = "add_node"
    ADD_MARKET = "add_market"
    ADD_RESOURCE = "add_resource"
    ADD_EDGE = "add_edge"
    DELETE = "delete"


class NodeType:
    """Node type constants for visualization and behavior."""
    AGENT = 0           # Regular agent (circle, blue)
    ADVERSARIAL = 1     # Adversarial agent (circle, red)
    MARKET = 2          # Market mechanism (square, orange)
    RESOURCE = 3        # Resource depot (diamond, green)


class EditMode:
    """
    Interaction handler for graph editing.

    Coordinates between user actions and EditSession based on active tool.
    Maintains tool state and provides handlers for mouse/keyboard events.
    """

    def __init__(
        self,
        session: EditSession,
        node_radius: float = 0.05  # Radius for click detection (normalized coords)
    ):
        """
        Initialize edit mode.

        Args:
            session: EditSession to coordinate with
            node_radius: Click detection radius in normalized [0,1] coords
        """
        self.session = session
        self.node_radius = node_radius

        # Current tool state
        self.active_tool: Tool = Tool.SELECT

        # Drag state for edge creation
        self.drag_start_node: Optional[int] = None
        self.drag_current_pos: Optional[Tuple[float, float]] = None

        # Callbacks for UI updates
        self.on_tool_changed: Optional[Callable[[Tool], None]] = None
        self.on_drag_preview: Optional[Callable[[Optional[Tuple[int, float, float]]], None]] = None

        # Default node/edge attributes
        self.default_node_type: int = 0
        self.default_node_attrs: dict = {"resources": 100.0}
        self.default_edge_rel: str = "connections"
        self.default_edge_weight: float = 1.0
        self.default_edge_directed: bool = False

    def set_tool(self, tool: Tool) -> None:
        """
        Set the active editing tool.

        Args:
            tool: Tool to activate
        """
        self.active_tool = tool

        # Clear drag state when switching tools
        self.drag_start_node = None
        self.drag_current_pos = None

        if self.on_tool_changed:
            self.on_tool_changed(tool)

    def handle_click(self, x: float, y: float) -> None:
        """
        Handle mouse click at position (x, y) in normalized [0,1] coords.

        Routes to appropriate tool handler.

        Args:
            x: X coordinate [0, 1]
            y: Y coordinate [0, 1]
        """
        # Find clicked node (if any)
        clicked_node = self._find_node_at(x, y)

        # Route to tool handler
        if self.active_tool == Tool.SELECT:
            self._handle_select_click(clicked_node, x, y)
        elif self.active_tool == Tool.ADD_NODE:
            self._handle_add_node_click(clicked_node, x, y, NodeType.AGENT)
        elif self.active_tool == Tool.ADD_MARKET:
            self._handle_add_node_click(clicked_node, x, y, NodeType.MARKET)
        elif self.active_tool == Tool.ADD_RESOURCE:
            self._handle_add_node_click(clicked_node, x, y, NodeType.RESOURCE)
        elif self.active_tool == Tool.ADD_EDGE:
            self._handle_add_edge_click(clicked_node)
        elif self.active_tool == Tool.DELETE:
            self._handle_delete_click(clicked_node)

    def handle_drag(self, from_x: float, from_y: float, to_x: float, to_y: float) -> None:
        """
        Handle mouse drag from (from_x, from_y) to (to_x, to_y).

        Args:
            from_x: Start X coordinate [0, 1]
            from_y: Start Y coordinate [0, 1]
            to_x: End X coordinate [0, 1]
            to_y: End Y coordinate [0, 1]
        """
        from_node = self._find_node_at(from_x, from_y)
        to_node = self._find_node_at(to_x, to_y)

        # Route to tool handler
        if self.active_tool == Tool.SELECT and from_node is not None:
            self._handle_move_node(from_node, to_x, to_y)
        elif self.active_tool == Tool.ADD_EDGE and from_node is not None and to_node is not None:
            self._handle_create_edge(from_node, to_node)

    def handle_drag_start(self, x: float, y: float) -> None:
        """
        Handle start of drag operation.

        Used for edge creation preview.

        Args:
            x: X coordinate [0, 1]
            y: Y coordinate [0, 1]
        """
        if self.active_tool == Tool.ADD_EDGE:
            node = self._find_node_at(x, y)
            if node is not None:
                self.drag_start_node = node
                self.drag_current_pos = (x, y)
                if self.on_drag_preview:
                    self.on_drag_preview((node, x, y))

    def handle_drag_move(self, x: float, y: float) -> None:
        """
        Handle drag movement (for preview).

        Args:
            x: Current X coordinate [0, 1]
            y: Current Y coordinate [0, 1]
        """
        if self.active_tool == Tool.ADD_EDGE and self.drag_start_node is not None:
            self.drag_current_pos = (x, y)
            if self.on_drag_preview:
                self.on_drag_preview((self.drag_start_node, x, y))

    def handle_drag_end(self, x: float, y: float) -> None:
        """
        Handle end of drag operation.

        Args:
            x: Final X coordinate [0, 1]
            y: Final Y coordinate [0, 1]
        """
        if self.active_tool == Tool.ADD_EDGE and self.drag_start_node is not None:
            end_node = self._find_node_at(x, y)
            if end_node is not None and end_node != self.drag_start_node:
                self._handle_create_edge(self.drag_start_node, end_node)

        # Clear drag state
        self.drag_start_node = None
        self.drag_current_pos = None
        if self.on_drag_preview:
            self.on_drag_preview(None)

    def handle_key(self, key: str, ctrl: bool = False) -> None:
        """
        Handle keyboard input.

        Args:
            key: Key pressed (lowercase)
            ctrl: Whether Ctrl key is held
        """
        if ctrl:
            if key == 'z':
                self.session.undo()
            elif key == 'y':
                self.session.redo()
        elif key == 'delete':
            self._handle_delete_selected()
        elif key == 'escape':
            self.session.deselect_all()
        elif key == 's':
            self.set_tool(Tool.SELECT)
        elif key == 'n':
            self.set_tool(Tool.ADD_NODE)
        elif key == 'm':
            self.set_tool(Tool.ADD_MARKET)
        elif key == 'r':
            self.set_tool(Tool.ADD_RESOURCE)
        elif key == 'e':
            self.set_tool(Tool.ADD_EDGE)
        elif key == 'd':
            self.set_tool(Tool.DELETE)

    # Tool-specific handlers

    def _handle_select_click(self, node_id: Optional[int], x: float, y: float) -> None:
        """Handle click in select mode."""
        if node_id is not None:
            # TODO: Support multi-select with Ctrl
            self.session.select_node(node_id, multi_select=False)
        else:
            self.session.deselect_all()

    def _handle_add_node_click(
        self,
        node_id: Optional[int],
        x: float,
        y: float,
        node_type: int = 0
    ) -> None:
        """Handle click in add node/market/resource mode."""
        if node_id is None:  # Only create if clicking empty space
            # Set default attributes based on node type
            if node_type == NodeType.MARKET:
                attrs = {"capacity": 1000.0, "price": 1.0}
                label = "market"
            elif node_type == NodeType.RESOURCE:
                attrs = {"resources": 500.0, "growth_rate": 1.1}
                label = "resource"
            else:
                attrs = self.default_node_attrs.copy()
                label = "node"

            def create_node(state: GraphState) -> GraphState:
                new_state = add_node(
                    state,
                    node_type=node_type,
                    initial_attrs=attrs
                )
                # Store position in metadata
                # Find the newly added node ID (last active node)
                active_indices = new_state.get_active_indices()
                if len(active_indices) > 0:
                    new_node_id = int(active_indices[-1])
                    self.session.set_node_position(new_node_id, x, y)
                return new_state

            self.session.apply_edit(create_node, f"Add {label} at ({x:.2f}, {y:.2f})")

    def _handle_add_edge_click(self, node_id: Optional[int]) -> None:
        """Handle click in add edge mode."""
        # Edge creation is handled via drag, not click
        pass

    def _handle_delete_click(self, node_id: Optional[int]) -> None:
        """Handle click in delete mode."""
        if node_id is not None:
            def delete_node_fn(state: GraphState) -> GraphState:
                return remove_node(state, node_id)

            self.session.apply_edit(delete_node_fn, f"Delete node {node_id}")

    def _handle_move_node(self, node_id: int, x: float, y: float) -> None:
        """Handle node movement in select mode."""
        self.session.set_node_position(node_id, x, y)

    def _handle_create_edge(self, from_node: int, to_node: int) -> None:
        """Handle edge creation between two nodes."""
        def create_edge_fn(state: GraphState) -> GraphState:
            return add_edge(
                state,
                from_id=from_node,
                to_id=to_node,
                rel_name=self.default_edge_rel,
                weight=self.default_edge_weight,
                directed=self.default_edge_directed
            )

        self.session.apply_edit(create_edge_fn, f"Create edge {from_node} → {to_node}")

    def _handle_delete_selected(self) -> None:
        """Delete currently selected nodes."""
        selected = self.session.get_selected_nodes()
        if not selected:
            return

        # Sort in descending order to avoid index shifting issues
        nodes_to_delete = sorted(selected, reverse=True)

        def delete_selected_fn(state: GraphState) -> GraphState:
            current = state
            for node_id in nodes_to_delete:
                try:
                    current = remove_node(current, node_id)
                except ValueError:
                    # Node might have been already removed
                    pass
            return current

        self.session.apply_edit(delete_selected_fn, f"Delete {len(selected)} nodes")
        self.session.deselect_all()

    # Helper methods

    def _find_node_at(self, x: float, y: float) -> Optional[int]:
        """
        Find node at position (x, y) within click radius.

        Returns:
            Node ID if found, None otherwise
        """
        state = self.session.current_state
        active_indices = state.get_active_indices()

        # Check custom positions first
        for node_id in active_indices:
            node_id_int = int(node_id)
            pos = self.session.get_node_position(node_id_int)
            if pos is not None:
                nx, ny = pos
                distance = math.sqrt((x - nx) ** 2 + (y - ny) ** 2)
                if distance <= self.node_radius:
                    return node_id_int

        # TODO: Fall back to VizConfig positions if no custom position

        return None
