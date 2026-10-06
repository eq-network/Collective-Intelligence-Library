"""
Reducer catalog — the fold shapes this library's metrics actually use.

Built by surveying every metric across the eight environments. The result was
narrower than expected: **not one of them needs the joint ``(T, N)`` array.**
They are four shapes, and each has an exact streaming form:

    metric shape                        example                              reducer
    ----------------------------------  -----------------------------------  ---------------------
    final state only                    ``trace["output"][-1]``              (read ``finals``)
    scalar per step, full series         ``late_log_output`` regressions      (keep in ``trace_fn``)
    per-agent vector reduced over time  ``gini_of(sum(reward, axis=0))``     ``per_agent_total``
    scalar reduced over both axes        ``human_origin_share``               ``late_mean``

So the ``(T, N)`` materialization every ``default_trace`` performs is not paying
for anything a metric reads — it is an artifact of tracing raw and reducing
afterwards. These reducers close that gap: same arithmetic, O(1) or O(N) carry.

Each entry takes a ``readout: GraphState -> value`` and returns a
``core.reduce.Reducer``. Readouts must be pure and branch-free (they run inside
``lax.scan``).

    from cilib.core.reduce import run_scan_reduce
    from cilib.metrics.reducers import late_mean, per_agent_total

    reducers = {"share": late_mean(lambda s: jnp.mean(1.0 - s.node_attrs["culture"]))}
    finals, trace, reduced = run_scan_reduce(round_fn, s0, 2000, key, reducers)

**Exactness.** A streaming mean sums in a different order than ``jnp.mean`` over
a materialized block, so reducer results match their trace-based twins to
float32 rounding (~1e-7 relative), NOT bit-for-bit. Unlike the sparse-adjacency
swap, ``allclose`` is the honest assertion here. Nothing downstream thresholds
these values, so the difference does not amplify.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.core.reduce import Reducer

Readout = Callable[[GraphState], Any]


def _window_start(n_steps: int, frac: float) -> int:
    """First step index of the trailing ``frac`` of a run.

    Reproduces the ``_late`` idiom used across the environments' metrics —
    ``series[3 * T // 4:]`` — exactly for ``frac=0.25``, at every ``T``.
    """
    return int(n_steps * (1.0 - frac))


# --- whole-run folds -------------------------------------------------------------

def last(readout: Readout) -> Reducer:
    """The readout's final value. Mostly redundant — ``finals`` already carries
    the last state — but useful when the readout is a derived quantity you would
    otherwise recompute at the call site."""
    return Reducer(
        init=lambda s, T: readout(s),
        update=lambda acc, s, t, T: readout(s),
        finalize=lambda acc, T: acc,
    )


def total(readout: Readout) -> Reducer:
    """Running sum over all steps. ``total(harvest)`` replaces
    ``jnp.sum(trace["harvest"])`` with an O(1) carry."""
    return Reducer(
        init=lambda s, T: jnp.zeros_like(readout(s)),
        update=lambda acc, s, t, T: acc + readout(s),
        finalize=lambda acc, T: acc,
    )


def running_mean(readout: Readout) -> Reducer:
    """Mean over the whole run."""
    return Reducer(
        init=lambda s, T: jnp.zeros_like(readout(s)),
        update=lambda acc, s, t, T: acc + readout(s),
        finalize=lambda acc, T: acc / T,
    )


def late_mean(readout: Readout, frac: float = 0.25) -> Reducer:
    """Mean over the trailing ``frac`` of the run — the streaming ``_late`` mean.

    The window gate is ``t >= start`` with ``start`` a static int, so this is
    branch-free in the traced step index. ``count`` is accumulated rather than
    assumed so the divisor matches the gate exactly at any ``T``.
    """
    def init(s, T):
        return (jnp.zeros_like(readout(s)), jnp.zeros((), jnp.float32))

    def update(acc, s, t, T):
        total_, count = acc
        gate = (t >= _window_start(T, frac)).astype(jnp.float32)
        return (total_ + gate * readout(s), count + gate)

    def finalize(acc, T):
        total_, count = acc
        return total_ / jnp.maximum(count, 1.0)

    return Reducer(init=init, update=update, finalize=finalize)


def at_step(readout: Readout, step_key: str) -> Reducer:
    """The readout just before step ``global_attrs[step_key]`` runs: the state after
    round ``t_on - 1``, i.e. the last state a mechanism switching on at ``t_on`` has
    not yet touched (``scheduled(..., onset=step_key)``).

    The step is read from the state, so it can be a ``run_grid`` dial. ``t`` is the
    scan index, which equals the ``step`` global when the run starts at step 0.

    Exception to the identity-init rule in ``core/reduce.py``: ``init`` is
    ``readout(initial_state)``, because for ``t_on = 0`` the state before the
    mechanism first runs IS the initial state, which the scan never visits. A
    ``t_on`` at or beyond the run length leaves that initial value in place.
    """
    return Reducer(
        init=lambda s, T: readout(s),
        update=lambda acc, s, t, T: jnp.where(t == s.global_attrs[step_key] - 1,
                                              readout(s), acc),
        finalize=lambda acc, T: acc,
    )


def rate(predicate: Readout, frac: float = 1.0) -> Reducer:
    """Fraction of the run (optionally its trailing ``frac``) where a boolean
    readout holds. ``predicate`` may return a per-agent boolean array — it is
    averaged, so this covers ``compliance_rate``-style "mean over agents AND
    steps" metrics in one O(1) accumulator."""
    return late_mean(lambda s: jnp.mean(predicate(s).astype(jnp.float32)), frac)


def _extremum_identity(x, most_negative: bool):
    """The identity element for max/min over ``x``'s dtype.

    Seeding these with ``readout(initial_state)`` instead would fold over T+1
    states while every other reducer folds over T — the reducer would include
    the pre-run state that the trace never records, and `running_min` would
    silently return the initial value on any rising series.
    """
    if jnp.issubdtype(jnp.asarray(x).dtype, jnp.integer):
        info = jnp.iinfo(jnp.asarray(x).dtype)
        fill = info.min if most_negative else info.max
    else:
        fill = -jnp.inf if most_negative else jnp.inf
    return jnp.full_like(x, fill)


def running_max(readout: Readout) -> Reducer:
    """Elementwise running maximum (peaks: worst drawdown, highest concentration)."""
    return Reducer(
        init=lambda s, T: _extremum_identity(readout(s), most_negative=True),
        update=lambda acc, s, t, T: jnp.maximum(acc, readout(s)),
        finalize=lambda acc, T: acc,
    )


def running_min(readout: Readout) -> Reducer:
    """Elementwise running minimum (troughs: closest approach to collapse)."""
    return Reducer(
        init=lambda s, T: _extremum_identity(readout(s), most_negative=False),
        update=lambda acc, s, t, T: jnp.minimum(acc, readout(s)),
        finalize=lambda acc, T: acc,
    )


def running_var(readout: Readout) -> Reducer:
    """Population variance of a scalar readout over the run, via Welford.

    Welford rather than sum-of-squares: the naive form subtracts two large
    nearly-equal numbers and loses most of its significant digits in float32
    when the mean is far from zero, which is the normal case for a stock or a
    wage. Carry is three scalars either way.
    """
    def init(s, T):
        z = jnp.zeros_like(readout(s))
        return (jnp.zeros((), jnp.float32), z, z)      # (count, mean, m2)

    def update(acc, s, t, T):
        count, mean, m2 = acc
        x = readout(s)
        count = count + 1.0
        delta = x - mean
        mean = mean + delta / count
        return (count, mean, m2 + delta * (x - mean))

    def finalize(acc, T):
        count, _, m2 = acc
        return m2 / jnp.maximum(count, 1.0)

    return Reducer(init=init, update=update, finalize=finalize)


# --- per-agent folds (O(N) carry, not O(T*N)) ------------------------------------

def per_agent_total(readout: Readout,
                    then: Optional[Callable[[Any], Any]] = None) -> Reducer:
    """Sum a per-agent readout over time, then optionally reduce the ``(N,)``
    result. This is the shape behind every concentration metric in the repo:

        gini_of(jnp.sum(trace["last_reward"], axis=0))        # (T, N) materialized
        per_agent_total(reward_readout, then=gini_of)          # (N,) carry

    ``then`` runs once at finalize, so an O(N log N) or O(N^2) reduction like a
    Gini or an HHI costs nothing per step.
    """
    return Reducer(
        init=lambda s, T: jnp.zeros_like(readout(s)),
        update=lambda acc, s, t, T: acc + readout(s),
        finalize=lambda acc, T: acc if then is None else then(acc),
    )


def per_agent_late_mean(readout: Readout, frac: float = 0.25,
                        then: Optional[Callable[[Any], Any]] = None) -> Reducer:
    """Per-agent mean over the trailing ``frac``, then optionally reduce.

    Covers the ``gini_of(jnp.mean(_late(trace["influence"]), axis=0))`` family:
    the late window per agent, collapsed to a scalar at the end.
    """
    inner = late_mean(readout, frac)
    return Reducer(
        init=inner.init,
        update=inner.update,
        finalize=lambda acc, T: (lambda v: v if then is None else then(v))(
            inner.finalize(acc, T)),
    )


REDUCERS = {
    "last": last,
    "total": total,
    "running_mean": running_mean,
    "late_mean": late_mean,
    "at_step": at_step,
    "rate": rate,
    "running_max": running_max,
    "running_min": running_min,
    "running_var": running_var,
    "per_agent_total": per_agent_total,
    "per_agent_late_mean": per_agent_late_mean,
}
