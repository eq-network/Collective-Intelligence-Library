"""
GovSim-style metrics for common-pool-resource (commons) experiments.

These reproduce the standard evaluation suite from GovSim
(Piatti et al., "Cooperate or Collapse: Emergence of Sustainability Behaviors
of LLM Agents", NeurIPS 2024) so that results from Collective Intelligence Library commons
experiments are directly comparable to the LLM-agent cooperation literature.

Design stance: these are pure, representation-agnostic functions. Nothing here
imports GraphState. You feed them the harvest / resource histories produced by
*any* experiment (the dict-state lake model, a future GraphState fishing
commons, whatever) and they return scalars. Because they are written in jnp,
they vmap cleanly over a leading seed/batch axis — compute the whole metric
suite across 100 seeds in one compiled call.

Array conventions for a SINGLE run (T steps, N agents):
    resource_history : (T+1,)   stock at each step, including the initial stock
    harvests         : (T, N)   per-agent harvest taken at each step

For a BATCH of runs, prepend a seed axis (S, ...) and vmap any function here.
"""
import jax.numpy as jnp


# =============================================================================
# CORE METRICS  (the GovSim five)
# =============================================================================

def survival_time(resource_history, collapse_threshold: float = 1.0):
    """Number of steps the commons survived before collapse.

    GovSim "survival time": how many rounds the resource stays viable before
    the stock crashes below ``collapse_threshold``. Returns the full horizon T
    if the commons never collapses.

    Args:
        resource_history: (T+1,) stock at each step including the initial value.
        collapse_threshold: stock at/below which the commons is considered dead.

    Returns:
        Scalar number of surviving steps (jnp array, vmap-friendly).
    """
    alive = resource_history[1:] > collapse_threshold          # (T,)
    never_died = jnp.all(alive)
    first_dead = jnp.argmax(~alive)                            # 0 if step 0 dead
    return jnp.where(never_died, alive.shape[0], first_dead)


def total_gain(harvests):
    """Total resource harvested by the whole group over the run."""
    return jnp.sum(harvests)


def per_agent_gain(harvests):
    """Cumulative harvest per agent. Shape (N,)."""
    return jnp.sum(harvests, axis=0)


def efficiency(harvests, optimal_gain):
    """Achieved group gain as a fraction of optimal sustainable gain.

    GovSim efficiency = U / U*, where U* is the maximum total reward obtainable
    under optimal *sustainable* play over the same horizon. U* is regime- and
    growth-rule-specific, so you pass it in (see ``optimal_sustainable_gain``
    for the doubling-growth fishery, or compute it for logistic growth).

    Values near 1.0 mean the group extracted almost everything sustainably
    available; values >1.0 indicate unsustainable over-extraction (eating the
    stock), which usually coincides with a short ``survival_time``.
    """
    return total_gain(harvests) / optimal_gain


def gini(values):
    """Gini coefficient of a non-negative 1-D array. 0 = perfect equality.

    Uses the sorted-order formulation, which is exact and JIT-safe (no pairwise
    O(N^2) matrix needed for the typical small-N commons).
    """
    v = jnp.sort(values)
    n = v.shape[0]
    index = jnp.arange(1, n + 1)
    total = jnp.sum(v)
    numerator = 2.0 * jnp.sum(index * v) - (n + 1) * total
    denom = n * total
    return jnp.where(denom > 0, numerator / denom, 0.0)


def equality(harvests):
    """1 - Gini over per-agent cumulative gains. 1.0 = perfectly equal split."""
    return 1.0 - gini(per_agent_gain(harvests))


def over_usage(harvests, sustainable_per_agent):
    """Fraction of harvest actions that exceeded the sustainable level.

    GovSim "over-usage": how often agents take more than the commons can
    regenerate. ``sustainable_per_agent`` may be a scalar, or a (T,) / (T, N)
    array if the per-step threshold depends on the current stock.

    Returns a value in [0, 1].
    """
    over = harvests > sustainable_per_agent
    return jnp.mean(over.astype(jnp.float32))


# =============================================================================
# GROWTH-RULE HELPERS  (doubling / GovSim fishery)
# =============================================================================

def sustainable_total(stock):
    """Max total group harvest this step that still lets a doubling commons recover.

    GovSim fishery: after harvest, survivors double (capped at K). Taking half
    the stock leaves half, which doubles back to the original level. So the
    sustainable *group* harvest is ``stock / 2``; divide by N for per-agent.
    """
    return stock / 2.0


def optimal_sustainable_gain(carrying_capacity, n_steps):
    """Optimal total group gain under doubling growth held at carrying capacity.

    The reward-maximising sustainable policy holds the stock at K and harvests
    the regrowth (K/2) every step: U* = n_steps * K / 2.
    """
    return n_steps * carrying_capacity / 2.0


# =============================================================================
# CONVENIENCE
# =============================================================================

def summary(resource_history, harvests, optimal_gain,
            sustainable_per_agent, collapse_threshold: float = 1.0) -> dict:
    """Bundle the full GovSim metric suite for one run into a dict.

    For batched runs, vmap the individual metric functions instead — this
    convenience wrapper returns python floats and is meant for a single run /
    reporting, not for the compiled hot path.
    """
    return {
        "survival_time": float(survival_time(resource_history, collapse_threshold)),
        "total_gain": float(total_gain(harvests)),
        "efficiency": float(efficiency(harvests, optimal_gain)),
        "equality": float(equality(harvests)),
        "over_usage": float(over_usage(harvests, sustainable_per_agent)),
    }
