"""
Interactive graph editor demo.

Click on empty space to add nodes.
Drag from one node to another to add edges.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import jax.numpy as jnp
from core.graph import GraphState
from studio import Canvas
from studio.configs import FarmersMarketVizConfig


def create_empty_graph() -> GraphState:
    """Create empty graph with no nodes."""
    return GraphState(
        node_types=jnp.array([], dtype=jnp.int32),
        node_attrs={
            "resources_apples": jnp.array([]),
            "resources_wheat": jnp.array([]),
            "resources_corn": jnp.array([]),
            "growth_rate_apples": jnp.array([]),
            "growth_rate_wheat": jnp.array([]),
            "growth_rate_corn": jnp.array([]),
        },
        adj_matrices={
            "trade_network": jnp.zeros((0, 0))
        },
        global_attrs={
            "resource_types": ["apples", "wheat", "corn"]
        }
    )


def on_state_change(state: GraphState):
    """Callback when graph is edited."""
    print(f"Graph updated: {state.num_nodes} nodes, "
          f"{jnp.sum(state.adj_matrices['trade_network'])} edges")


if __name__ == "__main__":
    print("Starting interactive graph editor...")
    print("Click to add nodes, drag between nodes to add edges")
    print()

    # Create empty graph
    state = create_empty_graph()

    # Create canvas in edit mode
    canvas = Canvas(FarmersMarketVizConfig(), edit_mode=True)
    canvas.on_state_change = on_state_change

    # Open editor
    canvas.render(state)

    print("Editor closed.")
