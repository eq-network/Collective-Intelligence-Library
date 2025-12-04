"""
Tkinter-based graph renderer with interactive editing support.

Provides visual feedback for editing operations:
- Selection highlighting (blue rings)
- Drag preview for edge creation (dashed lines)
- Hover effects
- Keyboard event handling
"""
import tkinter as tk
from typing import Dict, List, Tuple, Optional, Callable, Set


class TkinterRenderer:
    """
    Interactive Tkinter renderer for displaying graphs.

    Supports editing interactions with visual feedback.
    """

    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize renderer.

        Args:
            width: Canvas width
            height: Canvas height
        """
        self.width = width
        self.height = height
        self.root = None
        self.canvas = None
        self._in_mainloop = False

        # Interaction handlers
        self.on_click: Optional[Callable[[float, float], None]] = None
        self.on_drag: Optional[Callable[[float, float, float, float], None]] = None
        self.on_key: Optional[Callable[[str, bool], None]] = None  # (key, ctrl_pressed)
        self.on_drag_start: Optional[Callable[[float, float], None]] = None
        self.on_drag_move: Optional[Callable[[float, float], None]] = None
        self.on_drag_end: Optional[Callable[[float, float], None]] = None

        # Drag state
        self._drag_start = None
        self._is_dragging = False

        # Store current visualization data for re-rendering
        self._current_positions = None
        self._current_colors = None
        self._current_labels = None
        self._current_edges = None

        # Selection state (for visual feedback)
        self._selected_nodes: Set[int] = set()

        # Drag preview state (for edge creation)
        self._drag_preview: Optional[Tuple[int, float, float]] = None

        # Hover state
        self._hovered_node: Optional[int] = None

    def render(
        self,
        positions: Dict[int, Tuple[float, float]],
        colors: Dict[int, str],
        labels: Dict[int, str],
        edges: List[Tuple[int, int]]
    ):
        """
        Render a graph.

        Args:
            positions: {node_id: (x, y)} in range [0, 1]
            colors: {node_id: color_string}
            labels: {node_id: label_string}
            edges: [(from_id, to_id), ...]
        """
        # Store current data
        self._current_positions = positions
        self._current_colors = colors
        self._current_labels = labels
        self._current_edges = edges

        if self.root is None:
            self.root = tk.Tk()
            self.root.title("Mycorrhiza Studio")
            self.canvas = tk.Canvas(
                self.root,
                width=self.width,
                height=self.height,
                bg="white"
            )
            self.canvas.pack()

            # Bind mouse events for interaction
            self.canvas.bind("<Button-1>", self._on_mouse_down)
            self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
            self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
            self.canvas.bind("<Motion>", self._on_mouse_move)

            # Bind keyboard events to root window
            self.root.bind("<KeyPress>", self._on_key_press)
            self.root.focus_set()

        # Draw the graph
        self._draw_graph()

        # Start event loop if not already running
        if not self._in_mainloop:
            self._in_mainloop = True
            self.root.mainloop()

    def _draw_graph(self):
        """Draw the current graph on the canvas with visual feedback."""
        if self.canvas is None or self._current_positions is None:
            return

        # Clear canvas
        self.canvas.delete("all")

        # Scale positions to canvas size
        def scale_pos(pos):
            x, y = pos
            return (
                int(x * (self.width - 100) + 50),
                int(y * (self.height - 100) + 50)
            )

        scaled_positions = {
            nid: scale_pos(pos) for nid, pos in self._current_positions.items()
        }

        # Draw edges first (so they're behind nodes)
        for from_id, to_id in self._current_edges:
            if from_id in scaled_positions and to_id in scaled_positions:
                x1, y1 = scaled_positions[from_id]
                x2, y2 = scaled_positions[to_id]
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="gray",
                    width=1
                )

        # Draw drag preview (dashed line during edge creation)
        if self._drag_preview is not None:
            start_node_id, end_x, end_y = self._drag_preview
            if start_node_id in scaled_positions:
                x1, y1 = scaled_positions[start_node_id]
                x2, y2 = scale_pos((end_x, end_y))
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="blue",
                    width=2,
                    dash=(4, 4)  # Dashed line
                )

        # Draw nodes
        node_radius = 20
        for node_id, (x, y) in scaled_positions.items():
            color = self._current_colors.get(node_id, "lightblue")
            label = self._current_labels.get(node_id, str(node_id))

            # Lighten color if hovered
            if node_id == self._hovered_node:
                # Simple hover effect: add a lighter outer ring
                self.canvas.create_oval(
                    x - node_radius - 3, y - node_radius - 3,
                    x + node_radius + 3, y + node_radius + 3,
                    fill="",
                    outline="lightgray",
                    width=3
                )

            # Draw selection highlight (blue ring)
            if node_id in self._selected_nodes:
                self.canvas.create_oval(
                    x - node_radius - 5, y - node_radius - 5,
                    x + node_radius + 5, y + node_radius + 5,
                    fill="",
                    outline="blue",
                    width=3
                )

            # Draw circle
            self.canvas.create_oval(
                x - node_radius, y - node_radius,
                x + node_radius, y + node_radius,
                fill=color,
                outline="black",
                width=2
            )

            # Draw label
            self.canvas.create_text(
                x, y,
                text=label,
                font=("Arial", 10, "bold")
            )

    def _normalize_coords(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert canvas coordinates to normalized [0, 1] coordinates.

        Args:
            x, y: Canvas pixel coordinates

        Returns:
            (x, y) in [0, 1] range
        """
        norm_x = (x - 50) / (self.width - 100)
        norm_y = (y - 50) / (self.height - 100)
        return (norm_x, norm_y)

    def _on_mouse_down(self, event):
        """Handle mouse button down event."""
        self._drag_start = (event.x, event.y)
        self._is_dragging = False

        # Notify drag start handler
        if self.on_drag_start:
            norm_x, norm_y = self._normalize_coords(event.x, event.y)
            self.on_drag_start(norm_x, norm_y)

    def _on_mouse_up(self, event):
        """Handle mouse button up event."""
        if self._drag_start is None:
            return

        start_x, start_y = self._drag_start
        end_x, end_y = event.x, event.y

        # Check if this was a click (not a drag)
        dist = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

        if dist < 5:  # Less than 5 pixels = click
            if self.on_click:
                norm_x, norm_y = self._normalize_coords(end_x, end_y)
                self.on_click(norm_x, norm_y)
        else:  # Drag
            if self.on_drag:
                start_norm = self._normalize_coords(start_x, start_y)
                end_norm = self._normalize_coords(end_x, end_y)
                self.on_drag(start_norm[0], start_norm[1], end_norm[0], end_norm[1])

        # Notify drag end handler
        if self.on_drag_end:
            norm_x, norm_y = self._normalize_coords(end_x, end_y)
            self.on_drag_end(norm_x, norm_y)

        self._drag_start = None
        self._is_dragging = False

    def _on_mouse_drag(self, event):
        """Handle mouse drag event (for visual feedback)."""
        if self._drag_start is not None:
            self._is_dragging = True

            # Notify drag move handler (for edge preview)
            if self.on_drag_move:
                norm_x, norm_y = self._normalize_coords(event.x, event.y)
                self.on_drag_move(norm_x, norm_y)

    def _on_mouse_move(self, event):
        """Handle mouse movement (for hover effects)."""
        if self._current_positions is None:
            return

        # Find node under cursor
        norm_x, norm_y = self._normalize_coords(event.x, event.y)
        hovered = None

        for node_id, (x, y) in self._current_positions.items():
            dist = ((norm_x - x) ** 2 + (norm_y - y) ** 2) ** 0.5
            if dist <= 0.05:  # Hover radius in normalized coords
                hovered = node_id
                break

        # Update hover state and redraw if changed
        if hovered != self._hovered_node:
            self._hovered_node = hovered
            self._draw_graph()

    def _on_key_press(self, event):
        """Handle keyboard events."""
        if self.on_key is None:
            return

        # Get key name
        key = event.keysym.lower()

        # Check for ctrl modifier
        ctrl_pressed = (event.state & 0x4) != 0

        # Handle special keys
        key_map = {
            "control_l": None,  # Ignore standalone ctrl
            "control_r": None,
            "shift_l": None,
            "shift_r": None,
        }

        if key in key_map:
            return  # Ignore modifier keys by themselves

        # Notify handler
        self.on_key(key, ctrl_pressed)

    # Public methods for updating visual state

    def set_selection(self, selected_nodes: Set[int]) -> None:
        """
        Update selected nodes for visual feedback.

        Args:
            selected_nodes: Set of node IDs to highlight
        """
        self._selected_nodes = selected_nodes
        self.refresh()

    def set_drag_preview(self, preview: Optional[Tuple[int, float, float]]) -> None:
        """
        Update drag preview for edge creation.

        Args:
            preview: (start_node_id, end_x, end_y) or None to clear
        """
        self._drag_preview = preview
        self.refresh()

    def refresh(self) -> None:
        """Refresh the display (redraw current state)."""
        if self.canvas is not None:
            self._draw_graph()
