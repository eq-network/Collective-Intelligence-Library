"""
Basin Stability Environment.

Composes through Mycorrhiza's core abstractions:
- GraphState for all state
- Transform pipeline for stepping
- Environment for lax.scan execution

Paper: "Basin Stability of Democratic Mechanisms Under Adversarial Pressure"
"""
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jr

from core.environment import Environment

from .state import create_initial_state
from .transforms import make_step_transform


class BasinStabilityEnv(Environment):
    """Proposal-selection resource game with swappable governance.

    The environment is fully defined by:
    1. Initial GraphState (from state.py)
    2. A composed step transform (from transforms.py)

    Resource collapses when R < collapse_threshold, handled via
    the alive flag in the transform pipeline.
    """

    def __init__(self, mechanism: str = "pdd", n_agents: int = 20,
                 n_adversarial: int = 0, seed: int = 42,
                 metrics: dict = None, **kwargs):
        state = create_initial_state(
            n_agents=n_agents,
            n_adversarial=n_adversarial,
            mechanism=mechanism,
            seed=seed,
            metrics=metrics,
            **kwargs,
        )

        step_transform = make_step_transform(mechanism=mechanism, metrics=metrics)

        super().__init__(initial_state=state, step_transform=step_transform)
        self.mechanism = mechanism


def run_batched(mechanism, n_agents, n_adversarial, master_key, n_seeds, T=200,
                metrics=None, **kwargs):
    """Run n_seeds episodes in parallel via jax.vmap.

    All seeds run as a single batched kernel — this is where GPU/TPU
    acceleration comes from. Returns a batched GraphState where every array
    has a leading (n_seeds,) dimension.

    Strategy: build one template state in Python (outside vmap), then inside
    vmap only swap the rng_key and re-randomize key-dependent arrays.  This
    avoids Python control flow inside the traced function.

    Args:
        mechanism: governance mechanism ("pdd", "prd", "pld")
        n_agents: number of agents per episode
        n_adversarial: number of adversarial agents
        master_key: JAX PRNGKey to split into per-seed keys
        n_seeds: number of parallel episodes
        T: number of timesteps
        metrics: dict of metric functions (optional)

    Returns:
        Batched final GraphState with shape (n_seeds, ...) on all arrays.
    """
    step_transform = make_step_transform(mechanism=mechanism, metrics=metrics)

    # Build template state once in Python (not traced)
    template = create_initial_state(
        n_agents=n_agents,
        n_adversarial=n_adversarial,
        mechanism=mechanism,
        seed=0,
        T=T,
        metrics=metrics,
        **kwargs,
    )

    keys = jr.split(master_key, n_seeds)
    K = template.global_attrs["K"]
    state_dim = template.global_attrs["state_dim"]
    n_actions = template.global_attrs["n_actions"]
    n_reps = template.global_attrs["n_reps"]

    def run_one(key):
        # Re-randomize only the key-dependent parts
        key, k1, k2, k_shuffle, k_reps = jr.split(key, 5)

        q_weights = jr.normal(k1, (n_agents, n_actions, state_dim)) * 0.01

        # Shuffle signal quality (same tier sizes, different assignment)
        signal_quality = template.node_attrs["signal_quality"]
        signal_quality = jr.permutation(k_shuffle, signal_quality)

        # Random initial representatives for PRD
        rep_indices = jr.choice(k_reps, n_agents, shape=(n_reps,), replace=False)
        rep_mask = jnp.zeros(n_agents, dtype=jnp.float32).at[rep_indices].set(1.0)

        # Swap into template
        state = template
        state = state.update_node_attrs("q_weights", q_weights)
        state = state.update_node_attrs("signal_quality", signal_quality)
        state = state.update_node_attrs("rep_mask", rep_mask)
        state = state.update_global_attr("rng_key", key)

        def scan_body(s, _):
            return step_transform(s), None

        final, _ = lax.scan(scan_body, state, None, length=T)
        return final

    return jax.vmap(run_one)(keys)
