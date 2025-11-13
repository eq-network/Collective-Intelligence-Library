# Enhanced app.py - WITH SCHEDULER INTEGRATION

import streamlit as st
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from core.graph import GraphState
from mechanisms.market import create_double_auction_transform
from core.scheduler import MechanismSchedule, MultiMechanismRunner

# ---------- Helpers ----------


def make_ring_with_chords(N, chord=5):
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        A[i, (i + 1) % N] = 1
        A[i, (i + chord) % N] = 1
    return jnp.array(A)


def init_state(N, seed, res_a, res_b, r0, b0, sigma):
    key = jrandom.PRNGKey(seed)
    key, k1, k2 = jrandom.split(key, 3)

    vals_a = jnp.clip(jrandom.normal(k1, (N,)) * sigma + r0, 0, None)
    vals_b = jnp.clip(jrandom.normal(k2, (N,)) * sigma + b0, 0, None)

    node_attrs = {res_a: vals_a, res_b: vals_b}
    adj_matrices = {"trade": make_ring_with_chords(N, chord=5)}
    price_key = f"price_{res_a}_{res_b}"
    global_attrs = {"tick": 0, "rng_key": key, price_key: jnp.array(1.0)}

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


def has_converged(state_prev, state_cur, res_a, res_b, tol=1e-2):
    """Check if ALL agents have deltas below tolerance."""
    a0 = np.array(state_prev.node_attrs[res_a])
    a1 = np.array(state_cur.node_attrs[res_a])
    b0 = np.array(state_prev.node_attrs[res_b])
    b1 = np.array(state_cur.node_attrs[res_b])

    max_delta_a = np.abs(a1 - a0).max()
    max_delta_b = np.abs(b1 - b0).max()

    return max_delta_a < tol and max_delta_b < tol


def gini_coefficient(x):
    """
    CORRECT Gini coefficient formula.
    Returns value in [0, 1] where 0=perfect equality, 1=total inequality.
    """
    x = np.array(x)
    if len(x) == 0 or np.sum(x) <= 0:
        return 0.0

    x_sorted = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n)


def conservation(state, res_a, res_b):
    a = jnp.sum(state.node_attrs[res_a])
    b = jnp.sum(state.node_attrs[res_b])
    return float(a), float(b)


def draw_graph(
    state_prev,
    state_cur,
    res_a,
    res_b,
    resource_key,
    pos=None,
    node_scale=0.8,
    figsize=(6, 4),
    edge_alpha=0.25,
    tol=1e-2,
):
    adj_np = np.array(state_cur.adj_matrices["trade"])
    G = nx.from_numpy_array(adj_np.astype(int), create_using=nx.DiGraph)

    a0 = np.array(state_prev.node_attrs[res_a])
    b0 = np.array(state_prev.node_attrs[res_b])
    a1 = np.array(state_cur.node_attrs[res_a])
    b1 = np.array(state_cur.node_attrs[res_b])

    d_res = a1 - a0 if resource_key == res_a else b1 - b0
    wealth = a1 + b1

    sizes = (wealth - wealth.min()) / (wealth.max() - wealth.min() + 1e-9) * 700 + 200
    sizes = sizes * node_scale

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        G, pos, node_size=sizes, node_color=d_res, cmap="coolwarm", ax=ax
    )
    nx.draw_networkx_edges(G, pos, arrows=False, alpha=edge_alpha, ax=ax)

    tick = int(state_cur.global_attrs.get("tick", 0))
    # Display as 1-indexed for human readability
    display_tick = tick if tick == 0 else tick + 1
    ax.set_title(
        f"Tick {display_tick} | size=wealth  color=Δ{resource_key}", fontsize=11
    )
    ax.axis("off")

    max_delta = np.abs(d_res).max()
    a_change = np.abs(a1 - a0).max()
    b_change = np.abs(b1 - b0).max()

    if max(a_change, b_change) < tol:
        ax.text(
            0.5,
            -0.08,
            f"✓ Equilibrium (max |Δ| = {max_delta:.4f} < {tol})",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            color="green",
            fontweight="bold",
        )
    else:
        ax.text(
            0.5,
            -0.08,
            f"Still adjusting (max |Δ| = {max(a_change, b_change):.4f})",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="orange",
        )

    return fig


def step_once():
    """Execute one tick using the scheduler."""
    # SCHEDULER: Use runner.step() instead of direct transform call
    st.session_state.state = st.session_state.runner.step()

    # Track histories
    price_key = f"price_{st.session_state.res_a}_{st.session_state.res_b}"
    price = float(st.session_state.state.global_attrs[price_key])
    st.session_state.price_history.append(price)

    res_a = st.session_state.res_a
    res_b = st.session_state.res_b
    wealth = np.array(st.session_state.state.node_attrs[res_a]) + np.array(
        st.session_state.state.node_attrs[res_b]
    )
    gini = gini_coefficient(wealth)
    st.session_state.gini_history.append(gini)


# ---------- UI ----------

st.set_page_config(page_title="Graph Market", layout="wide")
st.title("🌐 Graph Resource Market (Scheduler-Based)")

# Sidebar
with st.sidebar:
    st.subheader("🧬 Resources")
    res_a = st.text_input("Resource A", value="apples")
    res_b = st.text_input("Resource B", value="oranges")

    st.subheader("Initialization")
    N = st.slider("Agents", 10, 300, 50, 5)
    seed = st.number_input("Seed", value=42, step=1)
    r0 = st.number_input(f"Mean {res_a}", value=8.0)
    b0 = st.number_input(f"Mean {res_b}", value=8.0)
    sigma = st.number_input("Init std", value=2.0)
    trade_rate = st.slider("Trade rate", 0.0, 1.0, 0.2, 0.05)
    vis_key = st.selectbox("Color by delta of", [res_a, res_b])
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
        st.session_state.res_a = res_a
        st.session_state.res_b = res_b

        # Initialize state
        state = init_state(N, seed, res_a, res_b, r0, b0, sigma)
        st.session_state.state = state
        st.session_state.prev = state
        st.session_state.init = state

        # Create market transform
        price_key = f"price_{res_a}_{res_b}"
        market_transform = create_double_auction_transform(
            resource_pair=(res_a, res_b),
            price_key=price_key,
            trade_rate=trade_rate,
        )

        # SCHEDULER: Create mechanism schedule
        market_schedule = MechanismSchedule(
            name="double_auction",
            transform=market_transform,
            frequency=1,  # Run every tick
            phase=0,
        )

        # SCHEDULER: Create runner
        st.session_state.runner = MultiMechanismRunner(
            initial_state=state, mechanisms=[market_schedule], watch_keys=[res_a, res_b]
        )

        # Initialize histories
        st.session_state.price_history = [1.0]
        wealth_init = np.array(state.node_attrs[res_a]) + np.array(
            state.node_attrs[res_b]
        )
        st.session_state.gini_history = [gini_coefficient(wealth_init)]

        # Build layout
        adj_np0 = np.array(state.adj_matrices["trade"])
        _, st.session_state.POS0 = build_layout_positions(adj_np0)

        st.rerun()

# First-time init
if "state" not in st.session_state or st.session_state.state is None:
    st.session_state.res_a = res_a
    st.session_state.res_b = res_b

    state = init_state(N, seed, res_a, res_b, r0, b0, sigma)
    st.session_state.state = state
    st.session_state.prev = state
    st.session_state.init = state

    price_key = f"price_{res_a}_{res_b}"
    market_transform = create_double_auction_transform(
        resource_pair=(res_a, res_b),
        price_key=price_key,
        trade_rate=trade_rate,
    )

    # SCHEDULER: Create mechanism schedule
    market_schedule = MechanismSchedule(
        name="double_auction", transform=market_transform, frequency=1, phase=0
    )

    # SCHEDULER: Create runner
    st.session_state.runner = MultiMechanismRunner(
        initial_state=state, mechanisms=[market_schedule], watch_keys=[res_a, res_b]
    )

    st.session_state.price_history = [1.0]
    wealth_init = np.array(state.node_attrs[res_a]) + np.array(state.node_attrs[res_b])
    st.session_state.gini_history = [gini_coefficient(wealth_init)]

    adj_np0 = np.array(state.adj_matrices["trade"])
    _, st.session_state.POS0 = build_layout_positions(adj_np0)

# Keep transform in sync when trade_rate changes
if ("runner" in st.session_state) and (
    trade_rate != st.session_state.get("trade_rate_cached")
):
    price_key = f"price_{st.session_state.res_a}_{st.session_state.res_b}"
    market_transform = create_double_auction_transform(
        resource_pair=(st.session_state.res_a, st.session_state.res_b),
        price_key=price_key,
        trade_rate=trade_rate,
    )

    # SCHEDULER: Recreate runner with new transform
    market_schedule = MechanismSchedule(
        name="double_auction", transform=market_transform, frequency=1, phase=0
    )

    st.session_state.runner = MultiMechanismRunner(
        initial_state=st.session_state.state,
        mechanisms=[market_schedule],
        watch_keys=[st.session_state.res_a, st.session_state.res_b],
    )

st.session_state["trade_rate_cached"] = trade_rate

# Tabs
tab1, tab2, tab3 = st.tabs(["🎮 Interactive", "📊 Convergence Test", "📈 Time Series"])

# TAB 1: Interactive
with tab1:
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
    res_a = st.session_state.res_a
    res_b = st.session_state.res_b

    # Metrics
    r_sum0, b_sum0 = conservation(state0, res_a, res_b)
    r_sum1, b_sum1 = conservation(state1, res_a, res_b)
    price_key = f"price_{res_a}_{res_b}"
    price = float(state1.global_attrs[price_key])

    wealth0 = np.array(state_init.node_attrs[res_a]) + np.array(
        state_init.node_attrs[res_b]
    )
    wealth1 = np.array(state1.node_attrs[res_a]) + np.array(state1.node_attrs[res_b])
    gini_init = gini_coefficient(wealth0)
    gini_curr = gini_coefficient(wealth1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Total {res_a}", f"{r_sum1:.2f}", f"{r_sum1 - r_sum0:+.2f}")
    m2.metric(f"Total {res_b}", f"{b_sum1:.2f}", f"{b_sum1 - b_sum0:+.2f}")
    m3.metric(f"Price {res_a}/{res_b}", f"{price:.3f}")
    m4.metric("Gini coefficient", f"{gini_curr:.3f}", f"{gini_curr - gini_init:+.3f}")

    tick = int(state1.global_attrs.get("tick", 0))
    # Display as 1-indexed for human readability
    display_tick = tick if tick == 0 else tick + 1

    # Better equilibrium reporting
    a0 = np.array(state0.node_attrs[res_a])
    a1 = np.array(state1.node_attrs[res_a])
    b0 = np.array(state0.node_attrs[res_b])
    b1 = np.array(state1.node_attrs[res_b])
    max_delta_a = np.abs(a1 - a0).max()
    max_delta_b = np.abs(b1 - b0).max()
    max_delta = max(max_delta_a, max_delta_b)

    if has_converged(state0, state1, res_a, res_b, tol):
        st.success(
            f"✅ System reached equilibrium at tick {display_tick} (max |Δ| = {max_delta:.4f} < {tol})"
        )
    else:
        st.info(
            f"⚙️ Market still adjusting (max |Δ| = {max_delta:.4f}, target < {tol}) — tick {display_tick}"
        )

    # Side-by-side graphs
    gcol1, gcol2 = st.columns(2)

    with gcol1:
        st.subheader("Initial")
        fig0 = draw_graph(
            state_init,
            state_init,
            res_a,
            res_b,
            vis_key,
            pos=POS0,
            node_scale=node_scale,
            figsize=(fig_w, fig_h),
            tol=tol,
        )
        st.pyplot(fig0, clear_figure=True, use_container_width=True)

    with gcol2:
        st.subheader(f"Updated (tick {display_tick})")
        fig1 = draw_graph(
            state0,
            state1,
            res_a,
            res_b,
            vis_key,
            pos=POS0,
            node_scale=node_scale,
            figsize=(fig_w, fig_h),
            tol=tol,
        )
        st.pyplot(fig1, clear_figure=True, use_container_width=True)

    # Agent table
    a0_arr = np.array(state0.node_attrs[res_a])
    b0_arr = np.array(state0.node_attrs[res_b])
    a1_arr = np.array(state1.node_attrs[res_a])
    b1_arr = np.array(state1.node_attrs[res_b])

    df = pd.DataFrame(
        {
            "agent": np.arange(a1_arr.shape[0]),
            f"{res_a}₀": a0_arr,
            f"{res_a}₁": a1_arr,
            f"Δ{res_a}": a1_arr - a0_arr,
            f"{res_b}₀": b0_arr,
            f"{res_b}₁": b1_arr,
            f"Δ{res_b}": b1_arr - b0_arr,
            "wealth": a1_arr + b1_arr,
        }
    ).sort_values(by="wealth", ascending=False)

    st.subheader("Agent changes")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button(
        "Download all agent data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"agents_tick_{display_tick}.csv",
        mime="text/csv",
    )

# TAB 2: Convergence Test
with tab2:
    st.subheader("🔬 Multi-Seed Convergence Analysis")
    st.markdown(
        "Run the market from multiple random initial conditions to measure convergence speed."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        num_seeds = st.number_input(
            "Number of seeds", value=20, min_value=5, max_value=100, step=5
        )
    with col_b:
        max_ticks_test = st.number_input(
            "Max ticks per run", value=100, min_value=20, max_value=500, step=10
        )

    if st.button("🚀 Run Convergence Test", type="primary"):
        with st.spinner(f"Running {num_seeds} simulations..."):
            results = []
            progress_bar = st.progress(0)

            for idx, s in enumerate(range(num_seeds)):
                state = init_state(
                    N, seed=s, res_a=res_a, res_b=res_b, r0=r0, b0=b0, sigma=sigma
                )
                prev = state
                price_key = f"price_{res_a}_{res_b}"

                # Create transform and scheduler for this seed
                market_transform = create_double_auction_transform(
                    (res_a, res_b), price_key=price_key, trade_rate=trade_rate
                )

                market_schedule = MechanismSchedule(
                    name="double_auction",
                    transform=market_transform,
                    frequency=1,
                    phase=0,
                )

                runner = MultiMechanismRunner(
                    initial_state=state,
                    mechanisms=[market_schedule],
                    watch_keys=[res_a, res_b],
                )

                converged = False
                for tick in range(max_ticks_test):
                    state = runner.step()

                    if has_converged(prev, state, res_a, res_b, tol):
                        wealth_final = np.array(state.node_attrs[res_a]) + np.array(
                            state.node_attrs[res_b]
                        )
                        results.append(
                            {
                                "seed": s,
                                "ticks_to_converge": tick + 1,
                                "final_price": float(state.global_attrs[price_key]),
                                "final_gini": gini_coefficient(wealth_final),
                            }
                        )
                        converged = True
                        break
                    prev = state

                if not converged:
                    wealth_final = np.array(state.node_attrs[res_a]) + np.array(
                        state.node_attrs[res_b]
                    )
                    results.append(
                        {
                            "seed": s,
                            "ticks_to_converge": max_ticks_test,
                            "final_price": float(state.global_attrs[price_key]),
                            "final_gini": gini_coefficient(wealth_final),
                        }
                    )

                progress_bar.progress((idx + 1) / num_seeds)

            progress_bar.empty()
            st.session_state.convergence_results = pd.DataFrame(results)

    if "convergence_results" in st.session_state:
        df_conv = st.session_state.convergence_results
        mean_conv = df_conv["ticks_to_converge"].mean()
        std_conv = df_conv["ticks_to_converge"].std()
        median_conv = df_conv["ticks_to_converge"].median()

        st.success(
            f"**Mean convergence: {mean_conv:.1f} ± {std_conv:.1f} ticks** (median: {median_conv:.0f})"
        )

        # Visualization
        fig_conv, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.bar(
            df_conv["seed"], df_conv["ticks_to_converge"], color="steelblue", alpha=0.7
        )
        ax1.axhline(
            mean_conv,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_conv:.1f}",
        )
        ax1.set_xlabel("Seed")
        ax1.set_ylabel("Ticks to Converge")
        ax1.set_title("Convergence Time by Seed")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        ax2.hist(
            df_conv["ticks_to_converge"],
            bins=15,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax2.axvline(
            mean_conv,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_conv:.1f}",
        )
        ax2.set_xlabel("Ticks to Converge")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Convergence Times")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_conv)

        st.subheader("📊 Detailed Results")
        st.dataframe(df_conv, use_container_width=True)

        st.download_button(
            "Download convergence data (CSV)",
            data=df_conv.to_csv(index=False).encode("utf-8"),
            file_name="convergence_analysis.csv",
            mime="text/csv",
        )

# TAB 3: Time Series
with tab3:
    st.subheader("📈 Price & Inequality Evolution")

    if len(st.session_state.price_history) > 1:
        fig_time, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        ticks = np.arange(len(st.session_state.price_history))

        # Price
        ax1.plot(ticks, st.session_state.price_history, "o-", linewidth=2)
        ax1.set_xlabel("Tick")
        ax1.set_ylabel(f"Price ({res_a}/{res_b})")
        ax1.set_title("Price Discovery Dynamics")
        ax1.grid(alpha=0.3)
        ax1.axhline(
            st.session_state.price_history[0],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Initial",
        )
        ax1.legend()

        # Gini
        ax2.plot(
            ticks, st.session_state.gini_history, "o-", color="darkorange", linewidth=2
        )
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Gini Coefficient")
        ax2.set_title("Wealth Inequality Evolution")
        ax2.grid(alpha=0.3)
        ax2.axhline(
            st.session_state.gini_history[0],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Initial",
        )
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig_time)

        # Summary stats
        col1, col2 = st.columns(2)
        with col1:
            price_range = max(st.session_state.price_history) - min(
                st.session_state.price_history
            )
            st.metric(
                "Price Range",
                f"{min(st.session_state.price_history):.3f} - {max(st.session_state.price_history):.3f}",
                f"Δ {price_range:.3f}",
            )
        with col2:
            gini_change = (
                st.session_state.gini_history[-1] - st.session_state.gini_history[0]
            )
            st.metric(
                "Gini Change",
                f"{gini_change:+.3f}",
                f"{st.session_state.gini_history[0]:.3f} → {st.session_state.gini_history[-1]:.3f}",
            )
    else:
        st.info("Run the simulation for multiple ticks to see time series data.")
