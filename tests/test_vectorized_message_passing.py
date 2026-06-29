"""
Tests for vectorized (scan-tier) message passing = information-form fusion.

Validates:
  1. weighted_aggregate reproduces hand-computed trust-weighted averages.
  2. It fuses arbitrary trailing shapes (h:(N,d), Pi:(N,d,d)) via einsum.
  3. row_normalize is safe on zero rows (no NaN).
  4. Composed into core.scan.run_scan, repeated fusion CONTRACTS disagreement to
     consensus on a connected graph — the V5 "averaging collapses diversity"
     signature, and proof the primitive runs compiled + vmaps.

Run: python -m pytest tests/test_vectorized_message_passing.py -q
"""
import jax
import jax.numpy as jnp
from jax import random

from cilib.core.graph import GraphState
from cilib.core.scan import run_scan, run_scan_batch
from cilib.transformations.vectorized_message_passing import (
    weighted_aggregate, row_normalize,
)


def _state(x, W, extra=None):
    n = x.shape[0]
    attrs = {"x": x}
    if extra:
        attrs.update(extra)
    return GraphState(
        node_types=jnp.zeros(n, dtype=jnp.int32),
        node_attrs=attrs,
        adj_matrices={"trust": W},
        global_attrs={},
    )


def test_weighted_average_matches_hand_computation():
    x = jnp.array([0.0, 10.0, 20.0])          # (3,)
    W = jnp.array([[1.0, 1.0, 0.0],            # node 0 averages itself + node 1
                   [0.0, 1.0, 0.0],            # node 1 keeps itself (self-loop)
                   [0.0, 0.0, 1.0]])           # node 2 keeps itself
    out = weighted_aggregate("trust", ["x"], normalize_rows=True)(_state(x, W))
    assert jnp.allclose(out.node_attrs["x"], jnp.array([5.0, 10.0, 20.0]))


def test_fuses_vector_and_matrix_attrs():
    n, d = 4, 3
    key = random.PRNGKey(0)
    kh, kP, kW = random.split(key, 3)
    h = random.normal(kh, (n, d))
    Pi = random.normal(kP, (n, d, d))
    W = jnp.abs(random.normal(kW, (n, n))) + jnp.eye(n)  # positive, self-loops

    out = weighted_aggregate("trust", ["h", "Pi"], normalize_rows=True)(
        _state(jnp.zeros(n), W, extra={"h": h, "Pi": Pi})
    )
    Wn = row_normalize(W)
    assert jnp.allclose(out.node_attrs["h"], jnp.einsum('ij,jd->id', Wn, h), atol=1e-5)
    assert jnp.allclose(out.node_attrs["Pi"], jnp.einsum('ij,jde->ide', Wn, Pi), atol=1e-5)


def test_row_normalize_safe_on_zero_rows():
    W = jnp.array([[0.0, 0.0], [2.0, 2.0]])
    Wn = row_normalize(W)
    assert not jnp.any(jnp.isnan(Wn))
    assert jnp.allclose(Wn[1], jnp.array([0.5, 0.5]))   # normalized
    assert jnp.allclose(Wn[0], jnp.array([0.0, 0.0]))   # left as-is, no NaN


def test_fusion_contracts_to_consensus_in_scan():
    """Repeated normalized fusion on a connected graph -> all nodes converge to
    the mean (the V5 'contraction'). A ring graph mixes gradually, so the
    monotonic decay of disagreement is visible. Also proves the primitive runs
    under scan."""
    n = 6
    x0 = jnp.arange(n, dtype=jnp.float32) * 3.0     # distinct opinions
    eye = jnp.eye(n)
    W = eye + jnp.roll(eye, 1, axis=1) + jnp.roll(eye, -1, axis=1)  # ring + self-loop
    init = _state(x0, W)

    fuse = weighted_aggregate("trust", ["x"], normalize_rows=True)

    def round_fn(state, t, key):
        return fuse(state)

    spread0 = float(jnp.std(x0))
    final, trace = run_scan(round_fn, init, 40, random.PRNGKey(0),
                            trace_fn=lambda s: jnp.std(s.node_attrs["x"]))
    spread_final = float(jnp.std(final.node_attrs["x"]))

    assert spread0 > 1.0
    assert spread_final < 1e-3, f"expected consensus, spread={spread_final}"
    # mean is conserved by averaging
    assert jnp.allclose(jnp.mean(final.node_attrs["x"]), jnp.mean(x0), atol=1e-4)
    # disagreement decays monotonically (a contraction): non-increasing over time
    assert jnp.all(trace[1:] <= trace[:-1] + 1e-6)
    assert trace[0] > trace[-1]


def test_fusion_vmaps_over_seeds():
    """Batched fusion: random initial opinions per seed, all contract; one program."""
    n, B, T = 5, 16, 12
    W = jnp.ones((n, n))
    fuse = weighted_aggregate("trust", ["x"], normalize_rows=True)

    def init_fn(key):
        return _state(random.normal(key, (n,)) * 5.0, W)

    def round_fn(state, t, key):
        return fuse(state)

    keys = random.split(random.PRNGKey(0), B)
    finals, _ = run_scan_batch(round_fn, init_fn, T, keys)
    spreads = jnp.std(finals.node_attrs["x"], axis=1)   # per-seed final spread
    assert spreads.shape == (B,)
    assert jnp.all(spreads < 1e-3)


if __name__ == "__main__":
    test_weighted_average_matches_hand_computation()
    test_fuses_vector_and_matrix_attrs()
    test_row_normalize_safe_on_zero_rows()
    test_fusion_contracts_to_consensus_in_scan()
    test_fusion_vmaps_over_seeds()
    print("all vectorized message passing tests passed")
