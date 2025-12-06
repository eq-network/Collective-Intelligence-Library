"""
SimulationScreen: Run and visualize the simulation.

Shows graph evolving over time with playback controls and live metrics.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Tuple, List, Callable

from studio.screens.base import Screen
from studio.visualization import (
    VizConfig, get_node_positions, get_node_colors, get_edges,
    normalize_to_screen
)
from core.graph import GraphState


class SimulationScreen(Screen):
    """
    Simulation execution screen.

    Features:
        - Play/Pause/Step controls
        - Speed slider
        - Round counter
        - Graph visualization updating in real-time
        - Live metrics panel with charts
    """

    def __init__(self, manager, app_state):
        super().__init__(manager, app_state)
        self.canvas: Optional[tk.Canvas] = None
        self.metrics_canvas: Optional[tk.Canvas] = None
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        self._round_var: Optional[tk.StringVar] = None
        self._accumulated_time: float = 0.0
        self.play_btn: Optional[ttk.Button] = None
        self._round_transform: Optional[Callable[[GraphState], GraphState]] = None
        self._resource_label: Optional[ttk.Label] = None
        self._resource_var: Optional[tk.StringVar] = None
        self.viz_config = VizConfig()

        # Metrics history for charting
        self._metrics_history: List[Dict] = []

    def on_enter(self, prev_screen: Optional[Screen] = None) -> None:
        self._create_base_frame()

        nav = self._create_nav_bar(title="Simulation", show_back=True)

        # Export button
        export_btn = ttk.Button(nav, text="Export >", command=self._go_to_export)
        export_btn.pack(side=tk.RIGHT)

        # Header with round counter
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=20)

        self._round_var = tk.StringVar(
            value=f"Round: 0 / {self.app_state.num_rounds}"
        )
        round_lbl = ttk.Label(
            header,
            textvariable=self._round_var,
            font=("Arial", 14)
        )
        round_lbl.pack(side=tk.RIGHT)

        # Main content area - split into graph (left) and metrics (right)
        content = ttk.Frame(self.frame)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left side: Graph canvas
        graph_frame = ttk.LabelFrame(content, text="Network Graph", padding=5)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            graph_frame,
            width=500,
            height=350,
            bg="white",
            relief="sunken",
            borderwidth=2
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right side: Metrics panel
        metrics_frame = ttk.LabelFrame(content, text="Live Metrics", padding=5)
        metrics_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Metrics canvas for charts
        self.metrics_canvas = tk.Canvas(
            metrics_frame,
            width=280,
            height=350,
            bg="#f8f8f8",
            relief="sunken",
            borderwidth=1
        )
        self.metrics_canvas.pack(fill=tk.BOTH, expand=True)

        # Controls
        controls = ttk.Frame(self.frame)
        controls.pack(fill=tk.X, padx=20, pady=10)

        # Playback buttons
        self.play_btn = ttk.Button(
            controls,
            text="Play",
            command=self._toggle_play
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        step_btn = ttk.Button(
            controls,
            text="Step >",
            command=self._step_forward
        )
        step_btn.pack(side=tk.LEFT, padx=5)

        reset_btn = ttk.Button(
            controls,
            text="Reset",
            command=self._reset
        )
        reset_btn.pack(side=tk.LEFT, padx=5)

        # Speed slider
        ttk.Label(controls, text="Speed:").pack(side=tk.LEFT, padx=(30, 5))
        speed_slider = ttk.Scale(
            controls,
            from_=0.25,
            to=4.0,
            value=1.0,
            command=self._set_speed,
            length=150
        )
        speed_slider.pack(side=tk.LEFT)

        # Status label
        self._status_var = tk.StringVar(value="Ready")
        status_lbl = ttk.Label(
            controls,
            textvariable=self._status_var,
            font=("Arial", 10)
        )
        status_lbl.pack(side=tk.RIGHT, padx=10)

        # Resource display at bottom
        resource_frame = ttk.Frame(self.frame)
        resource_frame.pack(fill=tk.X, padx=20)

        self._resource_var = tk.StringVar(value="Total Resources: --")
        self._resource_label = ttk.Label(
            resource_frame,
            textvariable=self._resource_var,
            font=("Arial", 11)
        )
        self._resource_label.pack(side=tk.LEFT)

        # Initialize simulation
        self._initialize_simulation()
        self._render_current_state()
        self._update_resource_display()
        self._update_metrics_chart()

    def _initialize_simulation(self) -> None:
        """Setup simulation from graph editor state."""
        if self.app_state.edit_session:
            initial_state = self.app_state.edit_session.export_final()
            self.app_state.simulation_history = [initial_state]
            self.app_state.current_round = 0

            # Reset metrics history
            self._metrics_history = []
            self._collect_metrics(initial_state)

            # Create transforms from scenario config
            scenario = self.app_state.selected_scenario
            if scenario and scenario.create_transforms:
                self._round_transform = scenario.create_transforms(
                    num_agents=self.app_state.num_agents,
                    adversarial_ratio=self.app_state.adversarial_ratio
                )
            else:
                self._round_transform = None

    def _toggle_play(self) -> None:
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        if self.play_btn:
            self.play_btn.config(text="Pause" if self.is_playing else "Play")
        self._status_var.set("Running..." if self.is_playing else "Paused")

    def _step_forward(self) -> None:
        """Advance simulation by one round."""
        if self.app_state.current_round >= self.app_state.num_rounds:
            self._status_var.set("Simulation complete")
            return

        # Get current state
        current = self.app_state.simulation_history[-1]

        # Apply transforms from scenario
        if self._round_transform is not None:
            new_state = self._round_transform(current)
        else:
            # Fallback: just increment round counter
            new_state = current.update_global_attr(
                "round",
                self.app_state.current_round + 1
            )

        # Save to history
        self.app_state.simulation_history.append(new_state)
        self.app_state.current_round += 1

        # Collect metrics for charting
        self._collect_metrics(new_state)

        self._update_round_display()
        self._render_current_state()
        self._update_resource_display()
        self._update_metrics_chart()

    def _reset(self) -> None:
        """Reset to initial state."""
        if self.app_state.simulation_history:
            initial_state = self.app_state.simulation_history[0]
            self.app_state.simulation_history = [initial_state]
            self.app_state.current_round = 0
            self.is_playing = False

            # Reset metrics
            self._metrics_history = []
            self._collect_metrics(initial_state)

            if self.play_btn:
                self.play_btn.config(text="Play")
            self._update_round_display()
            self._render_current_state()
            self._update_resource_display()
            self._update_metrics_chart()
            self._status_var.set("Reset")

    def _update_resource_display(self) -> None:
        """Update resource display from current state."""
        if not self._resource_var or not self.app_state.simulation_history:
            return

        state = self.app_state.simulation_history[-1]
        import jax.numpy as jnp

        # Try to get resources from node_attrs
        total = 0.0
        found_resources = False

        # Check for "resources" key (simple case)
        if "resources" in state.node_attrs:
            values = state.node_attrs["resources"]
            if state.is_capacity_mode:
                active_mask = state.get_active_mask()
                total = float(jnp.sum(values * active_mask))
            else:
                total = float(jnp.sum(values))
            found_resources = True

        # Check for resource_types in global_attrs
        resource_types = state.global_attrs.get("resource_types", [])
        resource_totals = {}

        for rt in resource_types:
            rkey = f"resources_{rt}"
            if rkey in state.node_attrs:
                values = state.node_attrs[rkey]
                if state.is_capacity_mode:
                    active_mask = state.get_active_mask()
                    resource_totals[rt] = float(jnp.sum(values * active_mask))
                else:
                    resource_totals[rt] = float(jnp.sum(values))
                found_resources = True

        if resource_totals:
            # Multiple resource types
            parts = [f"{rt}: {val:.0f}" for rt, val in resource_totals.items()]
            self._resource_var.set(f"Total Resources: {', '.join(parts)}")
        elif found_resources:
            # Single resource type
            self._resource_var.set(f"Total Resources: {total:.0f}")
        else:
            self._resource_var.set("Total Resources: --")

    def _set_speed(self, val) -> None:
        """Update playback speed."""
        self.playback_speed = float(val)

    def _update_round_display(self) -> None:
        """Update round counter label."""
        if self._round_var:
            self._round_var.set(
                f"Round: {self.app_state.current_round} / {self.app_state.num_rounds}"
            )

    def _render_current_state(self) -> None:
        """Render current simulation state."""
        if not self.canvas or not self.app_state.simulation_history:
            return

        state = self.app_state.simulation_history[-1]
        active_indices = state.get_active_indices()

        # Get positions using shared function
        editor_positions = {}
        if self.app_state.edit_session:
            editor_positions = self.app_state.edit_session.metadata.node_positions
        positions = get_node_positions(state, editor_positions)

        # Get colors using shared function
        colors = get_node_colors(state, self.viz_config)

        # Get node sizes based on resources (simulation-specific feature)
        node_sizes = {}
        base_size = self.viz_config.node_radius
        if "resources" in state.node_attrs:
            resources = state.node_attrs["resources"]
            max_res = float(max(resources[int(i)] for i in active_indices)) if len(active_indices) > 0 else 100
            min_res = float(min(resources[int(i)] for i in active_indices)) if len(active_indices) > 0 else 0
            range_res = max(max_res - min_res, 1)

            for node_id in active_indices:
                node_id_int = int(node_id)
                res = float(resources[node_id_int])
                # Scale size from 15 to 30 based on resources
                scale = (res - min_res) / range_res
                node_sizes[node_id_int] = int(15 + scale * 15)
        else:
            for node_id in active_indices:
                node_sizes[int(node_id)] = base_size

        # Get edges using shared function
        edges = get_edges(state)

        # Draw
        self.canvas.delete("all")
        width = self.canvas.winfo_width() or 800
        height = self.canvas.winfo_height() or 450
        margin = self.viz_config.margin

        # Scale positions to screen coordinates
        scaled = {
            nid: normalize_to_screen(pos[0], pos[1], width, height, margin)
            for nid, pos in positions.items()
        }

        # Draw edges
        for from_id, to_id in edges:
            if from_id in scaled and to_id in scaled:
                x1, y1 = scaled[from_id]
                x2, y2 = scaled[to_id]
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill=self.viz_config.edge_color,
                    width=self.viz_config.edge_width
                )

        # Draw nodes with dynamic sizing based on resources
        for nid, (x, y) in scaled.items():
            color = colors.get(nid, self.viz_config.default_color)
            r = node_sizes.get(nid, base_size)
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="black", width=2
            )
            self.canvas.create_text(x, y, text=str(nid))

        # Draw round indicator
        self.canvas.create_text(
            width - 20, 20,
            text=f"T={self.app_state.current_round}",
            font=("Arial", 12, "bold"),
            anchor="ne"
        )

    def _collect_metrics(self, state: GraphState) -> None:
        """Collect metrics from current state for charting."""
        import jax.numpy as jnp

        metrics = {
            "round": self.app_state.current_round,
            "total_resources": 0.0,
            "normal_resources": 0.0,
            "adversarial_resources": 0.0,
            "gini": 0.0,
            "num_normal": 0,
            "num_adversarial": 0,
        }

        active_indices = state.get_active_indices()

        if "resources" in state.node_attrs and len(active_indices) > 0:
            resources = state.node_attrs["resources"]

            # Collect resources by agent type
            normal_resources = []
            adversarial_resources = []
            all_resources = []

            for node_id in active_indices:
                node_id_int = int(node_id)
                res = float(resources[node_id_int])
                all_resources.append(res)

                node_type = int(state.node_types[node_id_int])
                if node_type == 0:  # Normal
                    normal_resources.append(res)
                elif node_type == 1:  # Adversarial
                    adversarial_resources.append(res)

            metrics["total_resources"] = sum(all_resources)
            metrics["normal_resources"] = sum(normal_resources)
            metrics["adversarial_resources"] = sum(adversarial_resources)
            metrics["num_normal"] = len(normal_resources)
            metrics["num_adversarial"] = len(adversarial_resources)

            # Calculate Gini coefficient (inequality measure)
            if len(all_resources) > 1:
                sorted_res = sorted(all_resources)
                n = len(sorted_res)
                cumulative = 0
                for i, r in enumerate(sorted_res):
                    cumulative += (2 * (i + 1) - n - 1) * r
                total = sum(sorted_res)
                if total > 0:
                    metrics["gini"] = cumulative / (n * total)

        self._metrics_history.append(metrics)

    def _update_metrics_chart(self) -> None:
        """Draw live metrics charts on the metrics canvas."""
        if not self.metrics_canvas:
            return

        self.metrics_canvas.delete("all")

        width = self.metrics_canvas.winfo_width() or 280
        height = self.metrics_canvas.winfo_height() or 350

        if not self._metrics_history:
            self.metrics_canvas.create_text(
                width // 2, height // 2,
                text="No data yet",
                font=("Arial", 12),
                fill="gray"
            )
            return

        # Layout: 3 charts stacked vertically
        chart_height = (height - 60) // 3
        margin = 10
        chart_width = width - 2 * margin

        # Chart 1: Total Resources over time
        self._draw_line_chart(
            x=margin, y=10,
            w=chart_width, h=chart_height,
            data=[m["total_resources"] for m in self._metrics_history],
            title="Total Resources",
            color="#4CAF50"
        )

        # Chart 2: Normal vs Adversarial Resources
        self._draw_dual_line_chart(
            x=margin, y=chart_height + 25,
            w=chart_width, h=chart_height,
            data1=[m["normal_resources"] for m in self._metrics_history],
            data2=[m["adversarial_resources"] for m in self._metrics_history],
            title="By Agent Type",
            color1="#87CEEB",  # Blue for normal
            color2="#FF6B6B",  # Red for adversarial
            label1="Normal",
            label2="Adversarial"
        )

        # Chart 3: Gini Coefficient (inequality)
        self._draw_line_chart(
            x=margin, y=2 * chart_height + 40,
            w=chart_width, h=chart_height,
            data=[m["gini"] for m in self._metrics_history],
            title="Inequality (Gini)",
            color="#FF9800",
            y_range=(0, 1)
        )

    def _draw_line_chart(
        self,
        x: int, y: int, w: int, h: int,
        data: List[float],
        title: str,
        color: str,
        y_range: Optional[Tuple[float, float]] = None
    ) -> None:
        """Draw a simple line chart."""
        if not self.metrics_canvas or not data:
            return

        # Title
        self.metrics_canvas.create_text(
            x + w // 2, y,
            text=title,
            font=("Arial", 9, "bold"),
            anchor="n"
        )

        # Chart area
        chart_x = x + 5
        chart_y = y + 15
        chart_w = w - 10
        chart_h = h - 25

        # Background
        self.metrics_canvas.create_rectangle(
            chart_x, chart_y, chart_x + chart_w, chart_y + chart_h,
            fill="white", outline="#ddd"
        )

        if len(data) < 2:
            # Show single value as text
            self.metrics_canvas.create_text(
                chart_x + chart_w // 2, chart_y + chart_h // 2,
                text=f"{data[0]:.0f}",
                font=("Arial", 10)
            )
            return

        # Calculate Y range
        if y_range:
            y_min, y_max = y_range
        else:
            y_min = min(data)
            y_max = max(data)
            if y_max == y_min:
                y_max = y_min + 1

        # Draw line
        points = []
        for i, val in enumerate(data):
            px = chart_x + (i / (len(data) - 1)) * chart_w
            py = chart_y + chart_h - ((val - y_min) / (y_max - y_min)) * chart_h
            points.extend([px, py])

        if len(points) >= 4:
            self.metrics_canvas.create_line(points, fill=color, width=2)

        # Current value label
        self.metrics_canvas.create_text(
            chart_x + chart_w - 5, chart_y + 5,
            text=f"{data[-1]:.0f}",
            font=("Arial", 8),
            anchor="ne",
            fill=color
        )

    def _draw_dual_line_chart(
        self,
        x: int, y: int, w: int, h: int,
        data1: List[float],
        data2: List[float],
        title: str,
        color1: str,
        color2: str,
        label1: str,
        label2: str
    ) -> None:
        """Draw a chart with two lines."""
        if not self.metrics_canvas or not data1 or not data2:
            return

        # Title with legend
        self.metrics_canvas.create_text(
            x + w // 2, y,
            text=title,
            font=("Arial", 9, "bold"),
            anchor="n"
        )

        # Legend
        legend_y = y + 12
        self.metrics_canvas.create_line(x + 5, legend_y, x + 20, legend_y, fill=color1, width=2)
        self.metrics_canvas.create_text(x + 22, legend_y, text=label1, font=("Arial", 7), anchor="w")
        self.metrics_canvas.create_line(x + 70, legend_y, x + 85, legend_y, fill=color2, width=2)
        self.metrics_canvas.create_text(x + 87, legend_y, text=label2, font=("Arial", 7), anchor="w")

        # Chart area
        chart_x = x + 5
        chart_y = y + 25
        chart_w = w - 10
        chart_h = h - 35

        # Background
        self.metrics_canvas.create_rectangle(
            chart_x, chart_y, chart_x + chart_w, chart_y + chart_h,
            fill="white", outline="#ddd"
        )

        if len(data1) < 2:
            return

        # Calculate Y range from both datasets
        all_data = data1 + data2
        y_min = min(all_data)
        y_max = max(all_data)
        if y_max == y_min:
            y_max = y_min + 1

        # Draw both lines
        for data, color in [(data1, color1), (data2, color2)]:
            points = []
            for i, val in enumerate(data):
                px = chart_x + (i / (len(data) - 1)) * chart_w
                py = chart_y + chart_h - ((val - y_min) / (y_max - y_min)) * chart_h
                points.extend([px, py])

            if len(points) >= 4:
                self.metrics_canvas.create_line(points, fill=color, width=2)

        # Current values
        self.metrics_canvas.create_text(
            chart_x + chart_w - 5, chart_y + 5,
            text=f"{data1[-1]:.0f}",
            font=("Arial", 8),
            anchor="ne",
            fill=color1
        )
        self.metrics_canvas.create_text(
            chart_x + chart_w - 5, chart_y + 15,
            text=f"{data2[-1]:.0f}",
            font=("Arial", 8),
            anchor="ne",
            fill=color2
        )

    def on_update(self, dt: float) -> None:
        """Called each frame - advance simulation if playing."""
        if not self.is_playing:
            return

        self._accumulated_time += dt * self.playback_speed

        # Step every 0.5 seconds (adjusted by speed)
        if self._accumulated_time >= 0.5:
            self._accumulated_time = 0
            self._step_forward()

            # Stop at end
            if self.app_state.current_round >= self.app_state.num_rounds:
                self.is_playing = False
                if self.play_btn:
                    self.play_btn.config(text="Play")
                self._status_var.set("Simulation complete")

    def _go_to_export(self) -> None:
        """Navigate to export screen."""
        self.is_playing = False  # Stop playback
        self.navigate_to("export")

    def on_exit(self, next_screen: Optional[Screen] = None) -> None:
        """Stop playback when leaving."""
        self.is_playing = False
