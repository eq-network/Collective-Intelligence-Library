"""
Tkinter-based graph renderer.
"""
import tkinter as tk
from typing import Dict, List, Tuple, Optional, Callable


class TkinterRenderer:
    """Simple Tkinter renderer for displaying graphs."""

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

        # Drag state
        self._drag_start = None

        # Store current visualization data for re-rendering
        self._current_positions = None
        self._current_colors = None
        self._current_labels = None
        self._current_edges = None

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

        # Draw the graph
        self._draw_graph()

        # Start event loop if not already running
        if not self._in_mainloop:
            self._in_mainloop = True
            self.root.mainloop()

    def _draw_graph(self):
        """Draw the current graph on the canvas."""
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

        # Draw nodes
        node_radius = 20
        for node_id, (x, y) in scaled_positions.items():
            color = self._current_colors.get(node_id, "lightblue")
            label = self._current_labels.get(node_id, str(node_id))

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

        self._drag_start = None

    def _on_mouse_drag(self, event):
        """Handle mouse drag event (for visual feedback)."""
        # Could draw temporary line for visual feedback
        pass
