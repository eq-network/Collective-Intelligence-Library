"""
Vectorized message passing — the scan-tier counterpart to message_passing.py.

``message_passing.create_message_passing_transform`` is the EAGER form: a Python
loop over ``int(i)`` node indices, building ``Dict[str, Any]`` messages. It is
flexible (heterogeneous, typed packets) but host-syncs per node, so it cannot run
inside ``jit``/``lax.scan``.

This module is the PURE form for the scan tier: message passing expressed as
matrix ops over a named adjacency, so it compiles and ``vmap``s. The workhorse is
trust-weighted aggregation, which is exactly information-form belief fusion
("precision-weighted averaging on a trust graph", DeGroot in information form):

    new[i] = sum_j  W[i, j] * attr[j]          # W = adj_matrices[connection_type]

implemented as ``jnp.einsum('ij,j...->i...', W, attr)`` so it works for any
trailing attribute shape: a potential ``h`` of shape ``(N, d)``, a precision
``Pi`` of shape ``(N, d, d)``, candidate stacks ``(N, K, d, d)``, etc.

Memory is the self-loop: if ``W`` has a positive diagonal, each node mixes in its
own previous value, so an isolated node (only its self-edge) simply carries its
belief forward.
"""
from typing import List

import jax.numpy as jnp

from core.graph import GraphState
from core.category import Transform


def row_normalize(W: jnp.ndarray) -> jnp.ndarray:
    """Row-normalize an adjacency so each row sums to 1 (trust-weighted averaging).

    Rows that sum to zero are left untouched (no division), so a node with no
    incoming edges is not turned into NaN. Give nodes a self-loop (positive
    diagonal = "memory") to guarantee every row sums to > 0.
    """
    row_sums = W.sum(axis=1, keepdims=True)
    safe = jnp.where(row_sums > 0, row_sums, 1.0)
    return jnp.where(row_sums > 0, W / safe, W)


def weighted_aggregate(
    connection_type: str,
    attr_names: List[str],
    *,
    normalize_rows: bool = True,
) -> Transform:
    """Aggregate node attributes across a named adjacency — the fusion step.

    For each ``name`` in ``attr_names``::

        new_attr[i] = sum_j W[i, j] * attr[j]

    where ``W = state.adj_matrices[connection_type]`` (optionally row-normalized).
    This is information-form fusion: pass ``Pi`` (precisions) and ``h``
    (potentials) as ``attr_names`` to fuse beliefs over the trust graph.

    Args:
        connection_type: Key into ``adj_matrices`` (e.g. ``"trust"``).
        attr_names: Node attributes to aggregate (each has leading axis = N).
        normalize_rows: If True, row-normalize ``W`` first (weighted *average*,
            a contraction — the DeGroot / opinion-pooling regime). If False, use
            ``W`` as given (weighted *sum* — additive information accumulation).

    Returns:
        A pure, jit/scan-safe ``Transform``. Returns the state unchanged if the
        connection type is absent (mirrors the eager transform's tolerance).
    """
    def transform(state: GraphState) -> GraphState:
        W = state.adj_matrices.get(connection_type)
        if W is None:
            return state
        if normalize_rows:
            W = row_normalize(W)

        new_node_attrs = dict(state.node_attrs)
        for name in attr_names:
            attr = state.node_attrs[name]
            new_node_attrs[name] = jnp.einsum('ij,j...->i...', W, attr)
        return state.replace(node_attrs=new_node_attrs)

    return transform
