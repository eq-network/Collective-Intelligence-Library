"""
Spectral metrics: graph-Laplacian structure readouts (minimal C0 slice).

The cultural register (docs/cultural-register-design.md §8, §11) pulls a
spectral family forward as phase C0: generic readouts for any environment that
carries an agent-agent adjacency, the *measurement dual* of the separation dial
— you set homophily on the generator, you read partition alignment here. This
module is the minimal slice: ``fiedler_partition_alignment`` only, the detector
for "is the graph's primary fault line the human/AI boundary?". The rest of the
family (spectral-gap ratio, current-flow mediation shares) lands with A4.

Eigenvector SIGNS are arbitrary (any eigensolver may flip them), so alignment
compares induced *partitions*, never vectors: the |phi coefficient| (absolute
Matthews correlation) between the Fiedler sign-split of ``L = D − W`` and the
node-type split. Chance baseline ≈ 0 even under heavy type imbalance (a
matching-fraction score would baseline at ``max(n_h, n_ai)/n``); 1 exactly when
the splits coincide up to relabeling; 0 on degenerate one-sided splits (the
denominator guard). Disconnected graphs have λ₂ = 0 and no meaningful Fiedler
partition — keep generator homophily below 1.0.

All functions stay in jnp (``jnp.linalg.eigh`` traces and vmaps).
"""
import jax.numpy as jnp

from cilib.core.graph import GraphState
from cilib.metrics.identity import kind


def fiedler_partition_alignment_of(W, types):
    """|phi| between the Fiedler sign-bipartition of ``L = D − W`` and the binary
    ``types`` partition. 1 = the graph's primary fault line IS the type boundary;
    ~0 = unrelated. Sign-invariant by construction (|phi| is unchanged under
    flipping either labeling).

    A sparse (BCOO) adjacency is densified on entry: ``eigh`` is O(N³) on a dense
    factorization regardless, so there is nothing to save — and this is an
    offline readout, not an in-loop op. Without it a sparse ``friendship``
    (``value_contagion``'s ``sparse_friendship``) would fail obscurely inside
    ``jnp.sum``."""
    if hasattr(W, "todense"):
        W = W.todense()
    L = jnp.diag(jnp.sum(W, axis=-1)) - W
    _, vecs = jnp.linalg.eigh(L)                      # ascending: column 1 = Fiedler
    s = (vecs[:, 1] >= 0.0).astype(jnp.float32)
    t = (types > 0).astype(jnp.float32)
    n11 = jnp.sum(s * t)
    n10 = jnp.sum(s * (1.0 - t))
    n01 = jnp.sum((1.0 - s) * t)
    n00 = jnp.sum((1.0 - s) * (1.0 - t))
    den = jnp.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return jnp.abs(n11 * n00 - n10 * n01) / jnp.maximum(den, 1e-8)


@kind("index")
def fiedler_partition_alignment(state: GraphState, adj_key: str = "friendship"):
    """GraphState binding: alignment of the agent graph's Fiedler split with
    ``node_types`` (the human/AI boundary). ``friendship`` is the agent-graph
    adjacency convention (value_contagion); pass ``adj_key`` for others."""
    return fiedler_partition_alignment_of(state.adj_matrices[adj_key],
                                          state.node_types)


SPECTRAL_METRICS = {
    "fiedler_partition_alignment": fiedler_partition_alignment,
}
