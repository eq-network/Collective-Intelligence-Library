"""
Basin Stability Environment.

Composes through Mycorrhiza's core abstractions:
- GraphState for all state
- Transform pipeline for stepping
- Environment for lax.scan execution

Paper: "Basin Stability of Democratic Mechanisms Under Adversarial Pressure"
"""
import jax
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

    All seeds run as a single batched kernel — this is where GPU acceleration
    comes from. Returns a batched GraphState where every array has a leading
    (n_seeds,) dimension.

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
    keys = jr.split(master_key, n_seeds)

    def run_one(key):
        state = create_initial_state(
            n_agents=n_agents,
            n_adversarial=n_adversarial,
            mechanism=mechanism,
            key=key,
            T=T,
            metrics=metrics,
            **kwargs,
        )

        def scan_body(s, _):
            return step_transform(s), None

        final, _ = lax.scan(scan_body, state, None, length=T)
        return final

    return jax.vmap(run_one)(keys)
