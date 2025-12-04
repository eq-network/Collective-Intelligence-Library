"""
Visualization config for Farmer's Market environment.

Defines how to display farmers, resources, and trade networks.
"""
import jax.numpy as jnp
import math
from typing import Dict, List, Tuple
from core.graph import GraphState


class FarmersMarketVizConfig:
    """Visualization configuration for farmers_market environment."""

    def get_node_positions(self, state: GraphState) -> Dict[int, Tuple[float, float]]:
        """
        Position active farmers in a circle.

        Returns:
            {agent_id: (x, y)} with x, y in [0, 1]
        """
        active_indices = state.get_active_indices()
        num_farmers = len(active_indices)
        positions = {}

        for idx, agent_id in enumerate(active_indices):
            # Arrange in circle
            angle = 2 * math.pi * idx / num_farmers
            x = 0.5 + 0.4 * math.cos(angle)
            y = 0.5 + 0.4 * math.sin(angle)
            positions[int(agent_id)] = (x, y)

        return positions

    def get_node_colors(self, state: GraphState) -> Dict[int, str]:
        """
        Color active farmers by total resources.

        Green = high resources, Red = low resources.

        Returns:
            {agent_id: color_string}
        """
        colors = {}
        resource_types = state.global_attrs.get("resource_types", [])
        active_indices = state.get_active_indices()

        # Calculate total resources per active farmer
        totals = []
        for i in active_indices:
            i_int = int(i)
            total = sum(
                state.node_attrs[f"resources_{rt}"][i_int]
                for rt in resource_types
            )
            totals.append(total)

        # Normalize to [0, 1]
        min_total = min(totals) if totals else 0
        max_total = max(totals) if totals else 1
        range_total = max_total - min_total if max_total > min_total else 1

        for idx, agent_id in enumerate(active_indices):
            total = totals[idx]
            # Normalized value
            norm = (total - min_total) / range_total

            # Green (high) to Red (low)
            if norm > 0.5:
                # Green to Yellow
                r = int(255 * 2 * (1 - norm))
                g = 255
            else:
                # Yellow to Red
                r = 255
                g = int(255 * 2 * norm)

            colors[int(agent_id)] = f"#{r:02x}{g:02x}00"

        return colors

    def get_node_labels(self, state: GraphState) -> Dict[int, str]:
        """
        Label active farmers with their ID.

        Returns:
            {agent_id: label_string}
        """
        return {int(i): str(int(i)) for i in state.get_active_indices()}

    def get_edges(self, state: GraphState) -> List[Tuple[int, int]]:
        """
        Show trade network edges between active farmers.

        Returns:
            [(from_id, to_id), ...]
        """
        edges = []

        if "trade_network" in state.adj_matrices:
            network = state.adj_matrices["trade_network"]
            active_indices = state.get_active_indices()
            active_list = [int(i) for i in active_indices]

            for idx_i, i in enumerate(active_list):
                for j in active_list[idx_i + 1:]:
                    if network[i, j] > 0:
                        edges.append((i, j))

        return edges
