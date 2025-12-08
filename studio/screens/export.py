"""
ExportScreen: Export visualizations and data.

Provides visualization mode selection and export options.
"""
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Optional

from studio.screens.base import Screen


class ExportScreen(Screen):
    """
    Export and visualization options screen.

    Visualization modes:
        1. Messages Being Passed
        2. Trust Updates
        3. Market View
        4. Metrics Over Time
    """

    def __init__(self, manager, app_state):
        super().__init__(manager, app_state)
        self._mode_var: Optional[tk.StringVar] = None
        self._format_var: Optional[tk.StringVar] = None
        self.preview_canvas: Optional[tk.Canvas] = None

    def on_enter(self, prev_screen: Optional[Screen] = None) -> None:
        self._create_base_frame()
        self._create_nav_bar(title="Export Visualizations", show_back=True)

        # Two-column layout
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Left column: Visualization mode
        left = ttk.LabelFrame(main, text="Visualization Mode", padding=15)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._mode_var = tk.StringVar(value="messages")

        modes = [
            ("messages", "1. Messages Being Passed"),
            ("trust", "2. Trust Updates"),
            ("market", "3. Market View"),
            ("metrics", "4. Metrics Over Time")
        ]

        for value, text in modes:
            rb = ttk.Radiobutton(
                left,
                text=text,
                value=value,
                variable=self._mode_var,
                command=self._update_preview
            )
            rb.pack(anchor=tk.W, pady=5)

        # Right column: Export options
        right = ttk.LabelFrame(main, text="Export Options", padding=15)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        ttk.Label(right, text="Format:").pack(anchor=tk.W, pady=(0, 5))

        self._format_var = tk.StringVar(value="png")

        formats = [
            ("png", "PNG Image"),
            ("gif", "GIF Animation"),
            ("csv", "CSV Data"),
            ("json", "JSON State")
        ]

        for value, text in formats:
            rb = ttk.Radiobutton(
                right,
                text=text,
                value=value,
                variable=self._format_var
            )
            rb.pack(anchor=tk.W, pady=2)

        # Export button
        export_btn = ttk.Button(
            right,
            text="Export...",
            command=self._export
        )
        export_btn.pack(pady=20)

        # Info label
        info = ttk.Label(
            right,
            text="(Export saves to console for now)",
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        info.pack()

        # Preview area
        preview_frame = ttk.LabelFrame(self.frame, text="Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(10, 20))

        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=600,
            height=250,
            bg="white",
            relief="sunken",
            borderwidth=1
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Initial preview
        self._update_preview()

    def _update_preview(self) -> None:
        """Update preview based on selected mode."""
        if not self.preview_canvas:
            return

        self.preview_canvas.delete("all")

        mode = self._mode_var.get() if self._mode_var else "messages"
        width = self.preview_canvas.winfo_width() or 600
        height = self.preview_canvas.winfo_height() or 250

        # Draw placeholder based on mode
        if mode == "messages":
            self._draw_messages_preview(width, height)
        elif mode == "trust":
            self._draw_trust_preview(width, height)
        elif mode == "market":
            self._draw_market_preview(width, height)
        elif mode == "metrics":
            self._draw_metrics_preview(width, height)

        # Mode label
        self.preview_canvas.create_text(
            width // 2, 20,
            text=f"Preview: {mode.title()} Visualization",
            font=("Arial", 12, "bold")
        )

    def _draw_messages_preview(self, width: int, height: int) -> None:
        """Draw placeholder for messages view."""
        # Draw some nodes with animated-looking arrows
        cx, cy = width // 2, height // 2 + 20

        # Draw nodes in a small network
        nodes = [
            (cx - 100, cy - 50),
            (cx + 100, cy - 50),
            (cx, cy + 50),
        ]

        # Draw edges with arrow markers
        self.preview_canvas.create_line(
            nodes[0][0], nodes[0][1], nodes[1][0], nodes[1][1],
            arrow=tk.LAST, fill="blue", width=2
        )
        self.preview_canvas.create_line(
            nodes[1][0], nodes[1][1], nodes[2][0], nodes[2][1],
            arrow=tk.LAST, fill="blue", width=2
        )

        # Draw nodes
        for x, y in nodes:
            self.preview_canvas.create_oval(
                x - 15, y - 15, x + 15, y + 15,
                fill="lightblue", outline="black"
            )

    def _draw_trust_preview(self, width: int, height: int) -> None:
        """Draw placeholder for trust view."""
        cx, cy = width // 2, height // 2 + 20

        nodes = [
            (cx - 100, cy),
            (cx, cy),
            (cx + 100, cy),
        ]

        # Draw edges with colors (blue=up, red=down)
        self.preview_canvas.create_line(
            nodes[0][0], nodes[0][1], nodes[1][0], nodes[1][1],
            fill="blue", width=3
        )
        self.preview_canvas.create_line(
            nodes[1][0], nodes[1][1], nodes[2][0], nodes[2][1],
            fill="red", width=3
        )

        # Draw nodes
        for x, y in nodes:
            self.preview_canvas.create_oval(
                x - 15, y - 15, x + 15, y + 15,
                fill="white", outline="black"
            )

        # Legend
        self.preview_canvas.create_text(
            50, height - 30,
            text="Blue = Trust Up, Red = Trust Down",
            font=("Arial", 9),
            anchor="w"
        )

    def _draw_market_preview(self, width: int, height: int) -> None:
        """Draw placeholder for market view."""
        cx, cy = width // 2, height // 2 + 20

        # Central market node
        self.preview_canvas.create_rectangle(
            cx - 20, cy - 20, cx + 20, cy + 20,
            fill="gold", outline="black", width=2
        )

        # Surrounding agents
        agents = [
            (cx - 100, cy - 60),
            (cx + 100, cy - 60),
            (cx - 100, cy + 60),
            (cx + 100, cy + 60),
        ]

        # Draw connections to market
        for ax, ay in agents:
            self.preview_canvas.create_line(
                ax, ay, cx, cy,
                fill="orange", dash=(4, 2)
            )
            self.preview_canvas.create_oval(
                ax - 12, ay - 12, ax + 12, ay + 12,
                fill="lightblue", outline="black"
            )

    def _draw_metrics_preview(self, width: int, height: int) -> None:
        """Draw placeholder for metrics chart."""
        # Draw simple line chart placeholder
        margin = 60
        chart_width = width - 2 * margin
        chart_height = height - 100

        # Axes
        self.preview_canvas.create_line(
            margin, 40, margin, 40 + chart_height,
            fill="black"
        )
        self.preview_canvas.create_line(
            margin, 40 + chart_height, margin + chart_width, 40 + chart_height,
            fill="black"
        )

        # Fake data lines
        import math
        points_resources = []
        points_trust = []
        for i in range(20):
            x = margin + (i / 19) * chart_width
            y1 = 40 + chart_height - (0.3 + 0.5 * (i / 19)) * chart_height
            y2 = 40 + chart_height - (0.6 - 0.2 * math.sin(i / 3)) * chart_height
            points_resources.extend([x, y1])
            points_trust.extend([x, y2])

        self.preview_canvas.create_line(
            *points_resources, fill="blue", width=2, smooth=True
        )
        self.preview_canvas.create_line(
            *points_trust, fill="orange", width=2, smooth=True
        )

        # Legend
        self.preview_canvas.create_text(
            margin + 20, height - 25,
            text="Resources",
            fill="blue",
            anchor="w"
        )
        self.preview_canvas.create_text(
            margin + 100, height - 25,
            text="Trust",
            fill="orange",
            anchor="w"
        )

    def _export(self) -> None:
        """Handle export action."""
        mode = self._mode_var.get() if self._mode_var else "messages"
        fmt = self._format_var.get() if self._format_var else "png"

        # For now, just print to console
        print(f"\n=== EXPORT ===")
        print(f"Mode: {mode}")
        print(f"Format: {fmt}")
        print(f"Rounds in history: {len(self.app_state.simulation_history)}")
        print(f"Current round: {self.app_state.current_round}")
        print("(Actual file export not yet implemented)")
        print("==============\n")

        # Show confirmation
        if hasattr(self, '_format_var'):
            # Could show a message box here
            pass
