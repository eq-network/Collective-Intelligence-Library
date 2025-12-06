"""
Visual Graph Editor Demo: Interactive Graph Editing with Tkinter

This example demonstrates the complete EditSession pattern for interactive
graph editing in Mycorrhiza's Studio.

Features:
- Visual drag-and-drop graph creation
- Tool-based editing (Select, Add Node, Add Edge, Delete)
- Undo/redo with Ctrl+Z/Ctrl+Y
- Keyboard shortcuts (N, E, S, D for tools)
- Selection highlighting (blue rings)
- Drag preview for edge creation (dashed lines)
- Hover effects
- Live toolbar with status updates

Run with: python -m examples.visual_graph_editor
"""
import sys
from pathlib import Path

# Add parent directory to path for imports (when running directly)
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import tkinter as tk
import jax.numpy as jnp
from typing import Dict, List, Tuple

from core.graph import GraphState, create_padded_state
from studio.edit_session import EditSession
from studio.edit_mode import EditMode
from studio.toolbar import EditToolbar
from studio.renderer import TkinterRenderer


class SimpleVizConfig:
    """
    Simple visualization config for editing mode.

    Uses custom positions from EditMetadata when available.
    """

    def get_node_positions(
        self,
        state: GraphState,
        custom_positions: Dict[int, Tuple[float, float]]
    ) -> Dict[int, Tuple[float, float]]:
        """
        Get node positions, preferring custom positions from editing.

        Args:
            state: Current graph state
            custom_positions: Custom positions from EditMetadata

        Returns:
            {node_id: (x, y)} in [0, 1] normalized coords
        """
        import math

        active_indices = state.get_active_indices()
        positions = {}

        for idx, node_id in enumerate(active_indices):
            node_id_int = int(node_id)

            # Use custom position if available
            if node_id_int in custom_positions:
                positions[node_id_int] = custom_positions[node_id_int]
            else:
                # Default: arrange in circle
                angle = 2 * math.pi * idx / max(len(active_indices), 1)
                x = 0.5 + 0.3 * math.cos(angle)
                y = 0.5 + 0.3 * math.sin(angle)
                positions[node_id_int] = (x, y)

        return positions

    def get_node_colors(self, state: GraphState) -> Dict[int, str]:
        """Color nodes by type."""
        colors = {}
        active_indices = state.get_active_indices()

        # Color palette for different node types
        type_colors = {
            0: "#87CEEB",  # Sky blue
            1: "#90EE90",  # Light green
            2: "#FFB6C1",  # Light pink
            3: "#FFD700",  # Gold
        }

        for node_id in active_indices:
            node_id_int = int(node_id)
            node_type = int(state.node_types[node_id_int])
            colors[node_id_int] = type_colors.get(node_type, "#D3D3D3")  # Default gray

        return colors

    def get_node_labels(self, state: GraphState) -> Dict[int, str]:
        """Label nodes with their ID."""
        return {int(i): str(int(i)) for i in state.get_active_indices()}

    def get_edges(self, state: GraphState) -> List[Tuple[int, int]]:
        """Get edges from 'connections' adjacency matrix."""
        edges = []

        if "connections" in state.adj_matrices:
            network = state.adj_matrices["connections"]
            active_indices = state.get_active_indices()
            active_list = [int(i) for i in active_indices]

            for i in active_list:
                for j in active_list:
                    if i < j and network[i, j] > 0:  # Undirected, avoid duplicates
                        edges.append((i, j))

        return edges


class VisualGraphEditor:
    """
    Main visual graph editor application.

    Coordinates EditSession, EditMode, Toolbar, and TkinterRenderer.
    """

    def __init__(self, initial_capacity: int = 50):
        """
        Initialize visual graph editor.

        Args:
            initial_capacity: Maximum number of nodes (capacity mode)
        """
        # Create empty graph with capacity mode
        self.initial_state = create_padded_state(
            capacity=initial_capacity,
            initial_active=0,  # Start with empty graph
            node_attrs_init={
                "resources": jnp.array([], dtype=jnp.float32)
            },
            adj_matrices_init={
                "connections": jnp.zeros((0, 0), dtype=jnp.float32)
            },
            global_attrs={"round": 0}
        )

        # Create edit session
        self.session = EditSession(self.initial_state, max_history=100)

        # Create visualization config
        self.viz_config = SimpleVizConfig()

        # Create renderer
        self.renderer = TkinterRenderer(width=800, height=600)

        # Create edit mode (will wire up after renderer)
        self.edit_mode = EditMode(self.session, node_radius=0.05)

        # Create toolbar (will add to window after renderer initializes)
        self.toolbar = None

        # Wire up callbacks
        self._setup_callbacks()

        print("Visual Graph Editor initialized!")
        print("Controls:")
        print("  - Click to add nodes (in Add Node mode)")
        print("  - Drag between nodes to create edges (in Add Edge mode)")
        print("  - Click nodes to select (in Select mode)")
        print("  - Click nodes to delete (in Delete mode)")
        print("  - Ctrl+Z: Undo")
        print("  - Ctrl+Y: Redo")
        print("  - N: Add Node tool")
        print("  - E: Add Edge tool")
        print("  - S: Select tool")
        print("  - D: Delete tool")
        print("\nStarting editor...\n")

    def _setup_callbacks(self):
        """Wire up all callbacks between components."""

        # EditSession → Renderer: Update visualization on state change
        def on_state_changed(new_state: GraphState):
            self._update_renderer()
            if self.toolbar:
                self.toolbar.update()

        def on_metadata_changed(metadata):
            self._update_renderer()

        self.session.on_state_changed = on_state_changed
        self.session.on_metadata_changed = on_metadata_changed

        # Renderer → EditMode: Forward user interactions
        self.renderer.on_click = self.edit_mode.handle_click
        self.renderer.on_drag = self.edit_mode.handle_drag
        self.renderer.on_key = self.edit_mode.handle_key
        self.renderer.on_drag_start = self.edit_mode.handle_drag_start
        self.renderer.on_drag_move = self.edit_mode.handle_drag_move
        self.renderer.on_drag_end = self.edit_mode.handle_drag_end

        # EditMode → Renderer: Visual feedback for drag preview
        def on_drag_preview(preview):
            self.renderer.set_drag_preview(preview)

        self.edit_mode.on_drag_preview = on_drag_preview

    def _update_renderer(self):
        """Update renderer with current graph state."""
        state = self.session.current_state
        metadata = self.session.metadata

        # Get visualization data
        positions = self.viz_config.get_node_positions(state, metadata.node_positions)
        colors = self.viz_config.get_node_colors(state)
        labels = self.viz_config.get_node_labels(state)
        edges = self.viz_config.get_edges(state)

        # Update renderer
        self.renderer.render(positions, colors, labels, edges)

        # Update selection highlighting
        self.renderer.set_selection(metadata.selected_nodes)

    def run(self):
        """Start the visual editor."""
        # Create window and canvas first (without starting mainloop)
        state = self.session.current_state
        metadata = self.session.metadata

        # Get initial visualization data
        positions = self.viz_config.get_node_positions(state, metadata.node_positions)
        colors = self.viz_config.get_node_colors(state)
        labels = self.viz_config.get_node_labels(state)
        edges = self.viz_config.get_edges(state)

        # Initialize renderer window (this creates root and canvas)
        if self.renderer.root is None:
            self.renderer.root = tk.Tk()
            self.renderer.root.title("Mycorrhiza Studio - Visual Graph Editor")

            # Create toolbar at top
            self.toolbar = EditToolbar(self.renderer.root, self.edit_mode)
            self.toolbar.frame.pack(side=tk.TOP, fill=tk.X)

            # Create canvas below toolbar
            self.renderer.canvas = tk.Canvas(
                self.renderer.root,
                width=self.renderer.width,
                height=self.renderer.height,
                bg="white"
            )
            self.renderer.canvas.pack(side=tk.TOP)

            # Bind event handlers
            self.renderer.canvas.bind("<Button-1>", self.renderer._on_mouse_down)
            self.renderer.canvas.bind("<ButtonRelease-1>", self.renderer._on_mouse_up)
            self.renderer.canvas.bind("<B1-Motion>", self.renderer._on_mouse_drag)
            self.renderer.canvas.bind("<Motion>", self.renderer._on_mouse_move)
            self.renderer.root.bind("<KeyPress>", self.renderer._on_key_press)
            self.renderer.root.focus_set()

        # Store initial data and draw
        self.renderer._current_positions = positions
        self.renderer._current_colors = colors
        self.renderer._current_labels = labels
        self.renderer._current_edges = edges
        self.renderer._draw_graph()

        # Start main loop
        self.renderer._in_mainloop = True
        self.renderer.root.mainloop()


def main():
    """Run the visual graph editor demo."""
    print("\n" + "="*70)
    print("VISUAL GRAPH EDITOR DEMO")
    print("="*70)
    print("\nInteractive graph editing with EditSession pattern")
    print("Demonstrates: EditSession + EditMode + Toolbar + TkinterRenderer")
    print("\n" + "="*70 + "\n")

    # Create and run editor
    editor = VisualGraphEditor(initial_capacity=50)
    editor.run()

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nFinal graph state:")
    final_state = editor.session.export_final()
    print(f"  - Active nodes: {final_state.num_nodes}")
    print(f"  - Capacity: {final_state.capacity}")
    print(f"  - Node IDs: {list(final_state.get_active_indices())}")

    # Show undo/redo history
    print(f"\nEdit history:")
    print(f"  - Undo available: {editor.session.can_undo()}")
    print(f"  - History depth: {len(editor.session.history)}")
    print("\n")


if __name__ == "__main__":
    main()
