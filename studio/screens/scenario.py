"""
ScenarioSelectionScreen: Choose from available simulation scenarios.

Uses the registry to discover available environments.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, List

from studio.screens.base import Screen
from studio.app_state import ScenarioConfig
from studio.registry import ENVIRONMENT_TYPES, EnvironmentTypeInfo

from core.category import sequential


def create_farmers_market_transforms(num_agents: int, adversarial_ratio: float = 0.0):
    """
    Create transforms for Farmers Market simulation.

    Creates a simplified version without agent-driven trades for now,
    focusing on resource growth and consumption dynamics.

    Args:
        num_agents: Number of agents
        adversarial_ratio: Ratio of adversarial agents (higher consumption)

    Returns:
        A transform function that runs one round
    """
    num_adversarial = int(num_agents * adversarial_ratio)

    # Create a simple transform that:
    # 1. Grows resources
    # 2. Applies consumption (adversarial agents consume more)
    # 3. Tracks history
    # 4. Increments round

    def consumption_transform(state):
        """Simple consumption based on agent type."""
        import jax.numpy as jnp

        resource_types = state.global_attrs.get("resource_types", ["resources"])
        new_node_attrs = dict(state.node_attrs)

        for rt in resource_types:
            rkey = f"resources_{rt}"
            if rkey not in state.node_attrs:
                # Try simple "resources" key
                rkey = "resources"
                if rkey not in state.node_attrs:
                    continue

            current = state.node_attrs[rkey]

            # Consumption rates: normal = 5%, adversarial = 15%
            consumption_rates = jnp.where(
                state.node_types == 1,
                0.15,  # Adversarial consume more
                0.05   # Normal consumption
            )

            # Apply consumption
            new_values = current * (1.0 - consumption_rates)
            new_node_attrs[rkey] = new_values

        return state.replace(node_attrs=new_node_attrs)

    def simple_growth_transform(state):
        """Simple resource growth."""
        import jax.numpy as jnp

        resource_types = state.global_attrs.get("resource_types", ["resources"])
        new_node_attrs = dict(state.node_attrs)

        for rt in resource_types:
            rkey = f"resources_{rt}"
            gkey = f"growth_rate_{rt}"

            if rkey not in state.node_attrs:
                # Try simple "resources" key with fixed growth
                rkey = "resources"
                if rkey not in state.node_attrs:
                    continue
                # Fixed 8% growth for simple resources
                current = state.node_attrs[rkey]
                new_node_attrs[rkey] = current * 1.08
            elif gkey in state.node_attrs:
                # Use per-agent growth rates
                current = state.node_attrs[rkey]
                rates = state.node_attrs[gkey]
                new_node_attrs[rkey] = current * rates
            else:
                # Fixed 8% growth
                current = state.node_attrs[rkey]
                new_node_attrs[rkey] = current * 1.08

        return state.replace(node_attrs=new_node_attrs)

    def track_resources_transform(state):
        """Track total resources in history."""
        resource_types = state.global_attrs.get("resource_types", ["resources"])

        totals = {}
        for rt in resource_types:
            rkey = f"resources_{rt}"
            if rkey not in state.node_attrs:
                rkey = "resources"
            if rkey in state.node_attrs:
                import jax.numpy as jnp
                values = state.node_attrs[rkey]
                if state.is_capacity_mode:
                    active_mask = state.get_active_mask()
                    totals[rt] = float(jnp.sum(values * active_mask))
                else:
                    totals[rt] = float(jnp.sum(values))

        entry = {
            "round": state.global_attrs.get("round", 0),
            "total_resources": totals
        }

        new_global = dict(state.global_attrs)
        history = list(new_global.get("history", []))
        history.append(entry)
        new_global["history"] = history

        return state.replace(global_attrs=new_global)

    def increment_round_transform(state):
        """Increment round counter."""
        new_global = dict(state.global_attrs)
        new_global["round"] = state.global_attrs.get("round", 0) + 1
        return state.replace(global_attrs=new_global)

    # Compose transforms: growth -> consumption -> track -> increment
    return sequential(
        simple_growth_transform,
        consumption_transform,
        track_resources_transform,
        increment_round_transform
    )


# Transform factory mapping from environment IDs to transform creators
TRANSFORM_FACTORIES = {
    "farmers_market": create_farmers_market_transforms,
    "simple_resource": create_farmers_market_transforms,  # Reuse for now
    # Add more as environments are implemented
}


def _build_scenarios_from_registry() -> List[ScenarioConfig]:
    """Build scenario configs from the registry."""
    scenarios = []
    for env_id, env_info in ENVIRONMENT_TYPES.items():
        # Determine default agent count from parameters
        num_params = env_info.parameters
        default_agents = 10
        if "num_farmers" in num_params:
            default_agents = num_params["num_farmers"].get("default", 10)
        elif "num_agents" in num_params:
            default_agents = num_params["num_agents"].get("default", 10)
        elif "num_voters" in num_params:
            default_agents = num_params["num_voters"].get("default", 10)

        # Get transform factory if available
        transform_fn = TRANSFORM_FACTORIES.get(env_id)

        scenarios.append(ScenarioConfig(
            name=env_info.name,
            description=env_info.description,
            default_num_agents=default_agents,
            default_num_rounds=50,  # Default
            create_initial_graph=None,  # Uses editor-created graph
            create_transforms=transform_fn,
        ))
    return scenarios


# Build scenarios from registry
SCENARIOS: List[ScenarioConfig] = _build_scenarios_from_registry()


class ScenarioSelectionScreen(Screen):
    """
    Scenario selection screen.

    Shows list of available scenarios with descriptions.
    Only Farmers Market is enabled; others are grayed out.
    """

    def on_enter(self, prev_screen: Optional[Screen] = None) -> None:
        self._create_base_frame()
        self._create_nav_bar(title="Choose a Scenario", show_back=True)

        # Scenario list
        list_frame = ttk.Frame(self.frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=20)

        # Create scenario cards
        for idx, scenario in enumerate(SCENARIOS):
            self._create_scenario_card(list_frame, scenario, idx)

    def _create_scenario_card(
        self,
        parent: tk.Widget,
        scenario: ScenarioConfig,
        index: int
    ) -> None:
        """Create a clickable card for a scenario."""
        # Determine if scenario is implemented by checking for transforms
        is_implemented = scenario.create_transforms is not None

        # Card frame
        card = ttk.Frame(parent, relief="ridge", borderwidth=2)
        card.pack(fill=tk.X, pady=8)

        # Card content
        content = ttk.Frame(card, padding=15)
        content.pack(fill=tk.X)

        # Scenario name
        name_label = ttk.Label(
            content,
            text=f"{index + 1}. {scenario.name}",
            font=("Arial", 14, "bold")
        )
        name_label.pack(anchor=tk.W)

        # Gray out if not implemented
        if not is_implemented:
            name_label.config(foreground="gray")

        # Description
        desc_label = ttk.Label(
            content,
            text=scenario.description,
            wraplength=600
        )
        desc_label.pack(anchor=tk.W, pady=(5, 0))

        if not is_implemented:
            desc_label.config(foreground="gray")

        # Button row
        button_row = ttk.Frame(content)
        button_row.pack(fill=tk.X, pady=(10, 0))

        # Status label for unimplemented
        if not is_implemented:
            status = ttk.Label(
                button_row,
                text="(Coming soon)",
                foreground="gray",
                font=("Arial", 10, "italic")
            )
            status.pack(side=tk.LEFT)

        # Select button
        select_btn = ttk.Button(
            button_row,
            text="Select",
            command=lambda s=scenario: self._select_scenario(s)
        )
        select_btn.pack(side=tk.RIGHT)

        # Disable button if not implemented
        if not is_implemented:
            select_btn.config(state="disabled")

    def _select_scenario(self, scenario: ScenarioConfig) -> None:
        """Save selection and navigate to configuration."""
        self.app_state.selected_scenario = scenario
        self.app_state.num_agents = scenario.default_num_agents
        self.app_state.num_rounds = scenario.default_num_rounds
        self.navigate_to("configuration")
