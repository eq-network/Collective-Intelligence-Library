"""
ConfigurationScreen: Configure agents, rounds, and other parameters.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict

from studio.screens.base import Screen


class ConfigurationScreen(Screen):
    """
    Agent and simulation configuration screen.

    Shows sliders for:
        - Number of agents
        - Number of rounds
        - Adversarial agent ratio
    """

    def __init__(self, manager, app_state):
        super().__init__(manager, app_state)
        self._vars: Dict[str, tk.IntVar] = {}

    def on_enter(self, prev_screen: Optional[Screen] = None) -> None:
        self._create_base_frame()

        scenario = self.app_state.selected_scenario
        scenario_name = scenario.name if scenario else "Simulation"

        nav = self._create_nav_bar(title=f"Configure: {scenario_name}", show_back=True)

        # Add continue button to nav bar
        continue_btn = ttk.Button(
            nav,
            text="Continue >",
            command=self._continue_to_editor
        )
        continue_btn.pack(side=tk.RIGHT)

        # Configuration form
        form_frame = ttk.Frame(self.frame)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=100, pady=30)

        # Number of Agents slider
        self._create_slider(
            form_frame,
            "Number of Agents",
            "num_agents",
            min_val=2,
            max_val=50,
            default=self.app_state.num_agents
        )

        # Number of Rounds slider
        self._create_slider(
            form_frame,
            "Number of Rounds",
            "num_rounds",
            min_val=10,
            max_val=200,
            default=self.app_state.num_rounds
        )

        # Adversarial Ratio slider
        self._create_slider(
            form_frame,
            "Adversarial Agents (%)",
            "adversarial_pct",
            min_val=0,
            max_val=100,
            default=int(self.app_state.adversarial_ratio * 100),
            suffix="%"
        )

        # Network Density slider (for random connectivity)
        self._create_slider(
            form_frame,
            "Network Connectivity (%)",
            "network_density",
            min_val=0,
            max_val=100,
            default=int(self.app_state.custom_parameters.get("network_density", 0.3) * 100),
            suffix="%"
        )

        # Info label
        info = ttk.Label(
            self.frame,
            text="Adjust parameters and click Continue to proceed to the graph editor.",
            font=("Arial", 10)
        )
        info.pack(pady=20)

    def _create_slider(
        self,
        parent: tk.Widget,
        label: str,
        var_name: str,
        min_val: int,
        max_val: int,
        default: int,
        suffix: str = ""
    ) -> None:
        """Create a labeled slider with value display."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=20)

        # Label
        lbl = ttk.Label(
            row,
            text=label,
            font=("Arial", 12),
            width=25,
            anchor=tk.W
        )
        lbl.pack(side=tk.LEFT)

        # Value variable
        var = tk.IntVar(value=default)
        self._vars[var_name] = var

        # Value label (shows current value)
        val_lbl = ttk.Label(row, text=f"{default}{suffix}", width=8)
        val_lbl.pack(side=tk.RIGHT)

        # Slider
        def on_change(val):
            int_val = int(float(val))
            val_lbl.config(text=f"{int_val}{suffix}")

        slider = ttk.Scale(
            row,
            from_=min_val,
            to=max_val,
            variable=var,
            command=on_change,
            length=300
        )
        slider.pack(side=tk.RIGHT, padx=10)

    def _continue_to_editor(self) -> None:
        """Save configuration and navigate to graph editor."""
        self._save_values()
        self.navigate_to("editor")

    def _save_values(self) -> None:
        """Save slider values to app_state."""
        if "num_agents" in self._vars:
            self.app_state.num_agents = self._vars["num_agents"].get()
        if "num_rounds" in self._vars:
            self.app_state.num_rounds = self._vars["num_rounds"].get()
        if "adversarial_pct" in self._vars:
            self.app_state.adversarial_ratio = self._vars["adversarial_pct"].get() / 100.0
        if "network_density" in self._vars:
            self.app_state.custom_parameters["network_density"] = self._vars["network_density"].get() / 100.0

    def on_exit(self, next_screen: Optional[Screen] = None) -> None:
        """Save form state before leaving."""
        self._save_values()
