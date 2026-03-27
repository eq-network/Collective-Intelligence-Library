"""
GraphState factory for the governed commons harvest experiment.

Maps CPR game state onto Mycorrhiza's GraphState abstraction so all
transforms operate on a single immutable pytree.
"""
import jax.numpy as jnp
import jax.random as jr

from core.graph import GraphState


# Default harvest levels: 0 through 5 (rebalanced from [0,2,4,6,8,10])
DEFAULT_HARVEST_LEVELS = jnp.array([0., 1., 2., 3., 4., 5.])


def create_initial_state(
    n_agents: int = 100,
    n_adversarial: int = 25,
    n_harvest_levels: int = 6,
    K: float = 5000.0,
    growth_rate: float = 0.4,
    initial_resource: float = 4000.0,
    max_steps: int = 200,
    learning_rate: float = 0.1,
    prediction_decay: float = 0.9,
    harvest_levels: jnp.ndarray = None,
    seed: int = 42,
) -> GraphState:
    """Create initial GraphState for the CPR game.

    Agent layout: agents [0, n_agents-n_adversarial) are cooperative,
    agents [n_agents-n_adversarial, n_agents) are adversarial.
    """
    if harvest_levels is None:
        harvest_levels = DEFAULT_HARVEST_LEVELS

    key = jr.PRNGKey(seed)

    # Node types: 0=cooperative, 1=adversarial
    node_types = jnp.zeros(n_agents, dtype=jnp.int32)
    node_types = node_types.at[n_agents - n_adversarial:].set(1)

    node_attrs = {
        "harvest_weights": jnp.ones((n_agents, n_harvest_levels)),
        "vote_weights": jnp.ones((n_agents, n_harvest_levels)),
        "cumulative_harvest": jnp.zeros(n_agents),
        "last_harvest": jnp.zeros(n_agents),
        "prediction_scores": jnp.ones(n_agents) * 0.5,
        "rewards": jnp.zeros(n_agents),
    }

    # No edges needed for this experiment
    adj_matrices = {"interaction": jnp.ones((n_agents, n_agents))}

    global_attrs = {
        "resource_level": jnp.array(initial_resource),
        "quota": harvest_levels[-1],  # start unconstrained
        "step": jnp.array(0, dtype=jnp.int32),
        "rng_key": key,
        # Static config (treated as static in pytree)
        "K": K,
        "growth_rate": growth_rate,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "prediction_decay": prediction_decay,
        "harvest_levels": harvest_levels,
        "n_harvest_levels": n_harvest_levels,
    }

    return GraphState(
        node_types=node_types,
        node_attrs=node_attrs,
        adj_matrices=adj_matrices,
        global_attrs=global_attrs,
    )
