"""
Democracy transforms: voting mechanisms as pure graph transformations.

"""

import jax.numpy as jnp
import jax.random as jrandom
from typing import Dict, Any, Optional, Callable
from core.graph import GraphState
from core.category import Transform


def _ensure_row_stochastic(W: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize rows to sum to 1 (make row-stochastic)."""
    rs = W.sum(axis=1, keepdims=True)
    rs = jnp.where(rs < eps, 1.0, rs)
    return W / rs


def _onehot_from_prefs(vote_pref: jnp.ndarray, m: int) -> jnp.ndarray:
    """Convert integer preferences to one-hot utility matrix."""
    vp = vote_pref.astype(jnp.int32).ravel()
    n = vp.shape[0]
    U = jnp.zeros((n, m), dtype=float)
    vp = jnp.clip(vp, 0, m - 1)
    U = U.at[jnp.arange(n), vp].set(1.0)
    return U


def _apply_liars(
    U: jnp.ndarray, p: float, targeted_mask: Optional[jnp.ndarray]
) -> jnp.ndarray:
    """Apply strategic misrepresentation (liars) to utility matrix."""
    if p <= 0.0:
        return U
    n = U.shape[0]

    key = jrandom.PRNGKey(0)
    rand_vals = jrandom.uniform(key, shape=(n,))

    mask = (
        (rand_vals < p)
        if targeted_mask is None
        else (targeted_mask.astype(bool) & (rand_vals < p))
    )
    Ui = jnp.where(mask[:, jnp.newaxis], -U, U)
    return Ui


def _deliberate(
    U: jnp.ndarray, W: jnp.ndarray, self_weight: float, steps: int
) -> jnp.ndarray:
    """Apply network-based deliberation to utilities."""
    Wn = _ensure_row_stochastic(W)
    A = self_weight * jnp.eye(Wn.shape[0]) + (1.0 - self_weight) * Wn
    Up = U.copy()
    for _ in range(max(steps, 0)):
        Up = A @ Up
    return Up


def _plurality(U: jnp.ndarray) -> Dict[str, Any]:
    """Plurality (first-past-the-post) voting."""
    votes = jnp.argmax(U, axis=1)
    M = U.shape[1]
    counts = jnp.zeros(M, dtype=jnp.int32)
    counts = counts.at[votes].add(1)
    winner = int(jnp.argmax(counts))
    return {"rule": "plurality", "winner": winner, "counts": counts.tolist()}


def _approval(U: jnp.ndarray, k: int) -> Dict[str, Any]:
    """Approval voting: vote for top k alternatives."""
    N, M = U.shape
    k = max(1, min(k, M))
    from jax import lax

    top_k_indices = lax.top_k(U, k)[1]  # [N, k]

    counts = jnp.zeros(M, dtype=jnp.int32)
    for i in range(k):
        counts = counts.at[top_k_indices[:, i]].add(1)

    winner = int(jnp.argmax(counts))
    return {"rule": f"approval@{k}", "winner": winner, "counts": counts.tolist()}


def _irv(U: jnp.ndarray) -> Dict[str, Any]:
    """Instant runoff voting (ranked choice)."""
    N, M = U.shape
    ranks = jnp.argsort(-U, axis=1)  # [N, M] descending preference
    eliminated = jnp.zeros(M, dtype=bool)

    for _ in range(M - 1):
        elim_mask = eliminated[ranks]  # [N, M]
        first_valid_idx = jnp.argmax(~elim_mask, axis=1)  # [N]
        first_choice = ranks[jnp.arange(N), first_valid_idx]

        counts = jnp.zeros(M, dtype=jnp.int32)
        counts = counts.at[first_choice].add(1)

        counts_masked = jnp.where(eliminated, jnp.iinfo(jnp.int32).max, counts)

        if eliminated.sum() >= M - 1:
            break

        loser = jnp.argmin(counts_masked)
        eliminated = eliminated.at[loser].set(True)

    winner = int(jnp.where(~eliminated)[0][0])
    final_counts = jnp.zeros(M, dtype=jnp.int32)
    final_counts = final_counts.at[winner].set(N)

    return {"rule": "irv", "winner": winner, "counts": final_counts.tolist()}


def _spectral_health(
    W: jnp.ndarray, signal: jnp.ndarray, hf_start: int = 5
) -> Dict[str, float]:
    """Compute spectral properties of the network with respect to signal."""
    Wn = _ensure_row_stochastic(W)
    S = 0.5 * (Wn + Wn.T)
    D = jnp.diag(S.sum(axis=1))
    L = D - S

    evals, evecs = jnp.linalg.eigh(L)

    coeffs = evecs.T @ signal
    energy = coeffs**2
    total = float(energy.sum() + 1e-12)
    gap = float(evals[1]) if evals.shape[0] > 1 else 0.0
    hf_ratio = (
        float(energy[hf_start:].sum() / total) if energy.shape[0] > hf_start else 0.0
    )
    p = energy / total
    eig_entropy = float(-jnp.sum(p * jnp.log(p + 1e-12)))
    return {
        "spectral_gap": gap,
        "spectral_hf_ratio": hf_ratio,
        "spectral_eig_entropy": eig_entropy,
    }


def create_democracy_transform(
    vote_attr: str = "vote_preference",
    decision_attr: str = "policy",
    region_attr: str = "region",
    prefs_key: str = "policy_preferences",
    m_alternatives: int = 3,
    liar_p: float = 0.0,
    targeted: bool = False,
    target_mask_key: str = "liar_target_mask",
    deliberation_enabled: bool = False,
    self_weight: float = 0.6,
    delib_steps: int = 1,
    rule: str = "plurality",
    approval_k: int = 2,
    spectral_enabled: bool = False,
    spectral_adj_key: str = "social",
    spectral_signal: str = "winner",
    spectral_col: int = 0,
    spectral_hf_start: int = 5,
    outcome_key: str = "democracy_outcome",
) -> Transform:
    """
    Create a democracy transform with specified parameters.

    Args:
        vote_attr: Node attribute containing integer votes
        decision_attr: Global attribute to store winning alternative
        region_attr: Node attribute for regional voting (optional)
        prefs_key: Node attribute with preference matrix [N, M]
        m_alternatives: Number of alternatives to choose from
        liar_p: Probability of strategic misrepresentation
        targeted: Whether liars target specific agents
        target_mask_key: Node attribute masking liar targets
        deliberation_enabled: Whether to apply network deliberation
        self_weight: Weight on own preferences during deliberation
        delib_steps: Number of deliberation rounds
        rule: Voting rule ('plurality', 'approval', 'irv')
        approval_k: Number of approvals for approval voting
        spectral_enabled: Whether to compute spectral metrics
        spectral_adj_key: Adjacency matrix key for spectral analysis
        spectral_signal: Signal to analyze ('winner' or column index)
        spectral_col: Column index if spectral_signal not 'winner'
        spectral_hf_start: Start index for high-frequency components
        outcome_key: Global attribute to store full outcome dict

    Returns:
        Transform function that executes voting
    """

    def democracy_transform(state: GraphState) -> GraphState:
        # Build utility matrix from preferences or votes
        if prefs_key and prefs_key in state.node_attrs:
            U = jnp.asarray(state.node_attrs[prefs_key])
            M = U.shape[1]
        else:
            if vote_attr not in state.node_attrs:
                raise KeyError(f"Missing required attribute: {vote_attr}")
            vp = jnp.asarray(state.node_attrs[vote_attr])
            M = int(m_alternatives) if m_alternatives else int(vp.max() + 1)
            U = _onehot_from_prefs(vp, M)

        # Apply strategic misrepresentation
        tgt_mask = (
            jnp.asarray(state.node_attrs.get(target_mask_key))
            if targeted and target_mask_key in state.node_attrs
            else None
        )
        U1 = _apply_liars(U, p=liar_p, targeted_mask=tgt_mask)

        # Deliberation (if enabled)
        U2 = U1
        if deliberation_enabled:
            W = jnp.asarray(state.adj_matrices[spectral_adj_key])
            U2 = _deliberate(U1, W, self_weight=self_weight, steps=delib_steps)

        # Execute voting rule
        if rule == "plurality":
            outcome = _plurality(U2)
        elif rule == "approval":
            outcome = _approval(U2, k=approval_k)
        elif rule == "irv":
            outcome = _irv(U2)
        else:
            raise ValueError(f"Unknown voting rule: {rule}")

        # Regional decisions (if regions exist)
        regions = (
            jnp.asarray(state.node_attrs.get(region_attr))
            if region_attr in state.node_attrs
            else None
        )
        st = state
        if regions is not None:
            r = regions.astype(int)
            for rid in jnp.unique(r):
                idx = r == rid
                sub = (
                    _plurality(U2[idx])
                    if rule == "plurality"
                    else _approval(U2[idx], k=approval_k)
                )
                st = st.update_global_attr(
                    f"{decision_attr}_{int(rid)}", float(sub["winner"])
                )

        # Set global decision
        st = st.update_global_attr(decision_attr, float(outcome["winner"]))

        # Track which agents voted for winner
        if vote_attr in st.node_attrs:
            votes_vec = jnp.asarray(st.node_attrs[vote_attr]).reshape(-1)
        else:
            votes_vec = jnp.argmax(U2, axis=1).astype(jnp.int32)

        voted_for_winner = (votes_vec == int(outcome["winner"])).astype(jnp.float32)
        st = st.update_node_attrs("voted_for_winner", voted_for_winner)

        # Spectral diagnostics (if enabled)
        if spectral_enabled:
            W = jnp.asarray(state.adj_matrices[spectral_adj_key])
            sig = (
                U2[:, outcome["winner"]]
                if spectral_signal == "winner"
                else U2[:, spectral_col]
            )
            spec = _spectral_health(W, sig, hf_start=spectral_hf_start)
            for k, v in spec.items():
                st = st.update_global_attr(k, v)

        # Store outcome and updated preferences
        if outcome_key:
            st = st.update_global_attr(outcome_key, outcome)
        if prefs_key:
            st = st.update_node_attrs(prefs_key, U2)

        return st

    return democracy_transform


def ensure_democracy_inputs(
    prefs_key: str = "policy_preferences",
    vote_attr: str = "vote_preference",
    m_alternatives: int = 3,
) -> Transform:
    """
    Create a transform that ensures democracy inputs exist.

    If preferences or votes are missing, this initializes them randomly.
    Should be applied before democracy_transform in the schedule.
    """

    def init_transform(state: GraphState) -> GraphState:
        st, sub = state.split_rng()
        n = st.num_nodes

        # Initialize preferences if missing
        if prefs_key not in st.node_attrs:
            choices = jrandom.randint(sub, (n,), 0, m_alternatives)
            prefs = jnp.eye(m_alternatives)[choices]
            st = st.update_node_attrs(prefs_key, prefs)

        # Initialize votes if missing
        if vote_attr not in st.node_attrs:
            votes = jnp.argmax(st.node_attrs[prefs_key], axis=1).astype(jnp.int32)
            st = st.update_node_attrs(vote_attr, votes)

        return st

    return init_transform
