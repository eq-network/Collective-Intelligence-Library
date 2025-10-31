# Streamlit UI for a graph resource game using market transform.

import streamlit as st
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from core.graph import GraphState
from mechanisms.market import create_double_auction_transform

# ---------- helpers


def make_ring_with_chords(N, chord=5):
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        A[i, (i + 1) % N] = 1
        A[i, (i + chord) % N] = 1
    return jnp.array(A)


def init_state(N: int, seed: int, r0: float, b0: float, sigma: float) -> GraphState:
    key = jrandom.PRNGKey(seed)
    key, k1, k2 = jrandom.split(key, 3)

    rice = jnp.clip(jrandom.normal(k1, (N,)) * sigma + r0, 0, None)
    beans = jnp.clip(jrandom.normal(k2, (N,)) * sigma + b0, 0, None)

    node_attrs = {"rice": rice, "beans": beans}
    adj_matrices = {"trade": make_ring_with_chords(N, chord=5)}
    global_attrs = {"tick": 0, "rng_key": key, "price_rice_beans": jnp.array(1.0)}

    return GraphState(
        node_types=jnp.zeros(N, dtype=jnp.int32),
        node_attrs=node_attrs,
        adj_matrices=adj_matrices,
        global_attrs=global_attrs,
    )


@st.cache_resource(show_spinner=False)
def build_layout_positions(adj_numpy, seed=42):
    G = nx.from_numpy_array(adj_numpy.astype(int), create_using=nx.DiGraph)
    return G, nx.spring_layout(G, seed=seed)


def has_converged(
    state_prev: GraphState, state_cur: GraphState, tol: float = 1e-2
) -> bool:
    """True if per-agent deltas for both resources are below tol."""
    r0 = np.array(state_prev.node_attrs["rice"])
    r1 = np.array(state_cur.node_attrs["rice"])
    b0 = np.array(state_prev.node_attrs["beans"])
    b1 = np.array(state_cur.node_attrs["beans"])
    return np.all(np.abs(r1 - r0) < tol) and np.all(np.abs(b1 - b0) < tol)


def draw_graph(
    state_prev: GraphState,
    state_cur: GraphState,
    resource_key: str = "rice",
    pos=None,
    node_scale: float = 0.8,
    figsize=(6, 4),
    edge_alpha: float = 0.25,
    tol: float = 1e-2,
):
    adj_np = np.array(state_cur.adj_matrices["trade"])
    G = nx.from_numpy_array(adj_np.astype(int), create_using=nx.DiGraph)

    r0 = np.array(state_prev.node_attrs["rice"])
    b0 = np.array(state_prev.node_attrs["beans"])
    r1 = np.array(state_cur.node_attrs["rice"])
    b1 = np.array(state_cur.node_attrs["beans"])

    d_res = r1 - r0 if resource_key == "rice" else b1 - b0
    wealth = r1 + b1

    sizes = (wealth - wealth.min()) / (wealth.max() - wealth.min() + 1e-9) * 700 + 200
    sizes = sizes * node_scale

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        G, pos, node_size=sizes, node_color=d_res, cmap="coolwarm", ax=ax
    )
    nx.draw_networkx_edges(G, pos, arrows=False, alpha=edge_alpha, ax=ax)

    tick = int(state_cur.global_attrs.get("tick", 0))
    ax.set_title(f"Tick {tick} | size=wealth  color=Δ{resource_key}", fontsize=11)
    ax.axis("off")

    # --- Equilibrium annotation ---
    r_change = np.abs(r1 - r0).max()
    b_change = np.abs(b1 - b0).max()
    if max(r_change, b_change) < tol:
        ax.text(
            0.5,
            -0.08,
            f"System reached equilibrium (|Δ| < {tol})",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            color="green",
            fontweight="bold",
        )

    return fig


def conservation(state: GraphState):
    r = jnp.sum(state.node_attrs["rice"])
    b = jnp.sum(state.node_attrs["beans"])
    return float(r), float(b)


def step_once():
    # increment tick manually (no scheduler here)
    cur_tick = int(st.session_state.state.global_attrs.get("tick", 0))
    st.session_state.state = st.session_state.state.update_global_attr(
        "tick", cur_tick + 1
    )
    st.session_state.prev = st.session_state.state
    st.session_state.state = st.session_state.market(st.session_state.state)


# ---------- UI

st.set_page_config(page_title="Graph Market", layout="wide")
st.title("Graph Resource Market: Rice ↔ Beans")

with st.sidebar:
    st.subheader("Initialization")
    N = st.slider("Agents", 10, 300, 50, 5)
    seed = st.number_input("Seed", value=0, step=1)
    r0 = st.number_input("Mean rice", value=8.0)
    b0 = st.number_input("Mean beans", value=8.0)
    sigma = st.number_input("Init std", value=2.0)
    trade_rate = st.slider("Trade rate", 0.0, 1.0, 0.2, 0.05)
    vis_key = st.selectbox("Color by delta of", ["rice", "beans"])
    node_scale = st.slider("Node size scale", 0.2, 2.0, 0.8, 0.05)
    fig_w = st.slider("Figure width (in)", 3.0, 12.0, 6.0, 0.5)
    fig_h = st.slider("Figure height (in)", 2.0, 9.0, 4.0, 0.5)
    tol = st.number_input(
        "Equilibrium tolerance |Δ| <",
        value=0.01,
        min_value=0.0,
        step=0.005,
        format="%.3f",
    )

    if st.button("Reset state", type="primary"):
        st.session_state.state = init_state(N, seed, r0, b0, sigma)
        st.session_state.prev = st.session_state.state
        st.session_state.init = st.session_state.state  # freeze initial
        st.session_state.market = create_double_auction_transform(
            resource_pair=("rice", "beans"),
            price_key="price_rice_beans",
            trade_rate=trade_rate,
        )
        # compute and store a stable layout for side-by-side views
        adj_np0 = np.array(st.session_state.init.adj_matrices["trade"])
        _, st.session_state.POS0 = build_layout_positions(adj_np0)

# first-time init (after sidebar vars exist)
if "state" not in st.session_state or st.session_state.state is None:
    st.session_state.state = init_state(N, seed, r0, b0, sigma)
    st.session_state.prev = st.session_state.state
    st.session_state.init = st.session_state.state  # baseline for left plot
    st.session_state.market = create_double_auction_transform(
        resource_pair=("rice", "beans"),
        price_key="price_rice_beans",
        trade_rate=trade_rate,
    )
    adj_np0 = np.array(st.session_state.init.adj_matrices["trade"])
    _, st.session_state.POS0 = build_layout_positions(adj_np0)

# keep transform in sync with sidebar
if ("market" in st.session_state) and (
    trade_rate != st.session_state.get("trade_rate_cached")
):
    st.session_state.market = create_double_auction_transform(
        resource_pair=("rice", "beans"),
        price_key="price_rice_beans",
        trade_rate=trade_rate,
    )
st.session_state["trade_rate_cached"] = trade_rate

# actions row
colA, colB, colC, colD = st.columns([1, 1, 1, 2])
with colA:
    if st.button("Step 1 tick"):
        step_once()
with colB:
    steps = st.number_input("Run N ticks", value=0, min_value=0, step=10)
with colC:
    if st.button("Run"):
        for _ in range(int(steps)):
            step_once()

state0 = st.session_state.prev
state1 = st.session_state.state
state_init = st.session_state.init
POS0 = st.session_state.POS0

# metrics
r_sum0, b_sum0 = conservation(state0)
r_sum1, b_sum1 = conservation(state1)
price = float(state1.global_attrs["price_rice_beans"])

m1, m2, m3 = st.columns(3)
m1.metric("Total rice", f"{r_sum1:.2f}", f"{r_sum1 - r_sum0:+.2f}")
m2.metric("Total beans", f"{b_sum1:.2f}", f"{b_sum1 - b_sum0:+.2f}")
m3.metric("Price rice/beans", f"{price:.3f}")

# equilibrium banner
tick = int(state1.global_attrs.get("tick", 0))
if has_converged(state0, state1, tol):
    st.success(f"System reached equilibrium — |Δ| < {tol} at tick {tick}.")
else:
    st.info("Market still adjusting — trades above tolerance.")


# graphs side-by-side: Initial vs Updated (same node positions)
gcol1, gcol2 = st.columns(2)

with gcol1:
    st.subheader("Initial")
    fig0 = draw_graph(
        state_init,
        state_init,
        resource_key=vis_key,
        pos=POS0,
        node_scale=node_scale,
        figsize=(fig_w, fig_h),
        tol=tol,
    )
    st.pyplot(fig0, clear_figure=True, use_container_width=True)

with gcol2:
    st.subheader(f"Updated (tick {int(state1.global_attrs.get('tick', 0))})")
    fig1 = draw_graph(
        state0,
        state1,
        resource_key=vis_key,
        pos=POS0,
        node_scale=node_scale,
        figsize=(fig_w, fig_h),
        tol=tol,
    )
    st.pyplot(fig1, clear_figure=True, use_container_width=True)

# tabular deltas
r0_arr = np.array(state0.node_attrs["rice"])
b0_arr = np.array(state0.node_attrs["beans"])
r1_arr = np.array(state1.node_attrs["rice"])
b1_arr = np.array(state1.node_attrs["beans"])

df = pd.DataFrame(
    {
        "agent": np.arange(r1_arr.shape[0]),
        "rice0": r0_arr,
        "rice1": r1_arr,
        "Δrice": r1_arr - r0_arr,
        "beans0": b0_arr,
        "beans1": b1_arr,
        "Δbeans": b1_arr - b0_arr,
        "wealth1": r1_arr + b1_arr,
    }
).sort_values(by="wealth1", ascending=False)

st.subheader("Agent changes")
st.dataframe(df.head(20), use_container_width=True)

# CSV export
st.download_button(
    "Download all agent data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"agents_tick_{int(state1.global_attrs.get('tick', 0))}.csv",
    mime="text/csv",
)
