"""
GraphEditorScreen: Visual graph editing with drag-and-drop.

Integrates existing components:
- EditSession for state management
- EditMode for tool interactions
- EditToolbar for tool selection
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Tuple, List
import random

import jax.numpy as jnp

from studio.screens.base import Screen
from studio.edit_session import EditSession
from studio.edit_mode import EditMode, Tool
from studio.toolbar import EditToolbar
from studio.visualization import (
    VizConfig, GraphRenderer, get_node_positions, get_node_colors,
    get_node_shapes, get_node_labels, get_edges, normalize_to_screen,
    screen_to_normalized
)
from core.graph import create_padded_state


class GraphEditorScreen(Screen):
    """
    Graph initialization/editing screen.

    Integrates existing editing components into the screen system.
    """

    def __init__(self, manager, app_state):
        super().__init__(manager, app_state)
        self.edit_mode: Optional[EditMode] = None
        self.toolbar: Optional[EditToolbar] = None
        self.canvas: Optional[tk.Canvas] = None
        self.viz_config = VizConfig()
        self.renderer: Optional[GraphRenderer] = None
        self._drag_start: Optional[Tuple[int, int]] = None

    def on_enter(self, prev_screen: Optional[Screen] = None) -> None:
        self._create_base_frame()

        # Nav bar with custom buttons
        nav = self._create_nav_bar(title="Graph Editor", show_back=True)

        # Add "Run Simulation" button
        run_btn = ttk.Button(
            nav,
            text="Run Simulation >",
            command=self._start_simulation
        )
        run_btn.pack(side=tk.RIGHT)

        # Initialize or restore EditSession
        if self.app_state.edit_session is None:
            self._create_new_session()

        session = self.app_state.edit_session

        # Create EditMode
        self.edit_mode = EditMode(session, node_radius=0.05)

        # Create Toolbar
        self.toolbar = EditToolbar(self.frame, self.edit_mode)
        self.toolbar.frame.pack(side=tk.TOP, fill=tk.X)

        # Create Canvas
        self.canvas = tk.Canvas(
            self.frame,
            width=800,
            height=500,
            bg="white",
            relief="sunken",
            borderwidth=2
        )
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create renderer
        self.renderer = GraphRenderer(self.canvas, self.viz_config)

        # Wire up event handlers
        self._setup_event_handlers()

        # Wire up session callbacks
        session.on_state_changed = lambda s: self._render_graph()
        session.on_metadata_changed = lambda m: self._render_graph()

        # Wire up drag preview
        self.edit_mode.on_drag_preview = self._on_drag_preview

        # Initial render
        self._render_graph()

    def _create_new_session(self) -> None:
        """Create a new EditSession with initial graph."""
        num_agents = self.app_state.num_agents
        adversarial_ratio = self.app_state.adversarial_ratio
        network_density = self.app_state.custom_parameters.get("network_density", 0.3)

        # Calculate number of adversarial agents
        num_adversarial = int(num_agents * adversarial_ratio)
        num_normal = num_agents - num_adversarial

        # Create node types: 0 = normal (blue), 1 = adversarial (red)
        node_types = jnp.array(
            [0] * num_normal + [1] * num_adversarial,
            dtype=jnp.int32
        )

        # Create random connections matrix based on network_density
        connections = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)
        if network_density > 0:
            # Create random symmetric connections
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    if random.random() < network_density:
                        connections = connections.at[i, j].set(1.0)
                        connections = connections.at[j, i].set(1.0)

        # Create initial state with configured number of agents
        initial_state = create_padded_state(
            capacity=max(50, num_agents * 2),
            initial_active=num_agents,
            node_types_init=node_types,
            node_attrs_init={
                "resources": jnp.ones(num_agents, dtype=jnp.float32) * 100.0
            },
            adj_matrices_init={
                "connections": connections
            },
            global_attrs={"round": 0}
        )

        self.app_state.edit_session = EditSession(initial_state, max_history=100)

        # Initialize random positions for all nodes within a bounded area
        # Keep nodes within [0.15, 0.85] range to have margin from edges
        for i in range(num_agents):
            x = random.uniform(0.15, 0.85)
            y = random.uniform(0.15, 0.85)
            self.app_state.edit_session.set_node_position(i, x, y)

    def _setup_event_handlers(self) -> None:
        """Connect canvas events to EditMode."""
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.manager.root.bind("<KeyPress>", self._on_key_press)

    def _normalize_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Convert canvas coords to [0,1] normalized."""
        width = self.canvas.winfo_width() or 800
        height = self.canvas.winfo_height() or 500
        return screen_to_normalized(x, y, width, height, self.viz_config.margin)

    def _screen_coords(self, nx: float, ny: float) -> Tuple[int, int]:
        """Convert normalized [0,1] coords to screen coords."""
        width = self.canvas.winfo_width() or 800
        height = self.canvas.winfo_height() or 500
        return normalize_to_screen(nx, ny, width, height, self.viz_config.margin)

    def _on_mouse_down(self, event) -> None:
        x, y = self._normalize_coords(event.x, event.y)
        self.edit_mode.handle_drag_start(x, y)
        self._drag_start = (event.x, event.y)

    def _on_mouse_up(self, event) -> None:
        if self._drag_start is None:
            return

        start_x, start_y = self._drag_start
        end_x, end_y = event.x, event.y
        dist = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

        norm_end = self._normalize_coords(end_x, end_y)

        if dist < 5:  # Click (not drag)
            self.edit_mode.handle_click(*norm_end)
        else:  # Drag
            norm_start = self._normalize_coords(start_x, start_y)
            self.edit_mode.handle_drag(*norm_start, *norm_end)

        self.edit_mode.handle_drag_end(*norm_end)
        self._drag_start = None

    def _on_mouse_drag(self, event) -> None:
        x, y = self._normalize_coords(event.x, event.y)
        self.edit_mode.handle_drag_move(x, y)

    def _on_mouse_move(self, event) -> None:
        # Could add hover effects here
        pass

    def _on_key_press(self, event) -> None:
        ctrl = (event.state & 0x4) != 0
        self.edit_mode.handle_key(event.keysym.lower(), ctrl)

    def _on_drag_preview(self, preview) -> None:
        """Handle drag preview for edge creation."""
        self._render_graph()

        if preview is not None:
            node_id, end_x, end_y = preview
            session = self.app_state.edit_session
            start_pos = session.get_node_position(node_id)

            if start_pos:
                sx, sy = self._screen_coords(*start_pos)
                ex, ey = self._screen_coords(end_x, end_y)
                self.canvas.create_line(
                    sx, sy, ex, ey,
                    fill="blue",
                    dash=(4, 4),
                    width=2,
                    tags="preview"
                )

    def _render_graph(self) -> None:
        """Render current graph state to canvas."""
        if not self.canvas or not self.app_state.edit_session or not self.renderer:
            return

        session = self.app_state.edit_session
        state = session.current_state
        metadata = session.metadata

        # Use shared renderer
        self.renderer.render(
            state,
            custom_positions=metadata.node_positions,
            selected_nodes=metadata.selected_nodes,
            show_labels=True
        )

        # Update toolbar
        if self.toolbar:
            self.toolbar.update()

    def _start_simulation(self) -> None:
        """Save graph and navigate to simulation screen."""
        self.navigate_to("simulation")

    def on_exit(self, next_screen: Optional[Screen] = None) -> None:
        """Unbind events before leaving."""
        try:
            self.manager.root.unbind("<KeyPress>")
        except tk.TclError:
            pass  # Window might be closing

    def destroy(self) -> None:
        """Clean up resources."""
        try:
            self.manager.root.unbind("<KeyPress>")
        except (tk.TclError, AttributeError):
            pass
        super().destroy()
