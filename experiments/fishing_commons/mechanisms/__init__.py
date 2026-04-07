"""
Mechanism Library — Composable mechanism modules for CI-Lib.

Each mechanism is a Transform (GraphState -> GraphState) with a declared
type contract specifying its read/write sets. Mechanisms compose via
core.category.sequential() and must not have side effects.

The library metaphor: CI-Lib is a shelf of mechanism books. Each book is a
typed, composable Transform. You pull books off the shelf, compose them, and
the framework guarantees interoperability (all operate on GraphState) and
comparability (universal metrics).

Type Contracts
--------------
Each mechanism type declares which GraphState fields it reads and writes.
Disjoint write sets enable parallel composition — market writes allocations,
network writes received_signals, democracy writes policy parameters. No
conflicts when composed via sequential().
"""

# ---------------------------------------------------------------------------
# Market Type
# ---------------------------------------------------------------------------
# What: Centralised exchange of typed resources. Agents trade money/budget
#        for goods/harvest rights.
# Information topology: Global, centralised — all agents interact through a
#        single hub. Price signal reaches everyone simultaneously.
#
# Variants (planned):
#   - posted_price: Agent posts (quantity, price), others accept. Bilateral.
#                   Cite: basic market microstructure.
#   - double_auction: Buyers and sellers both post. Trades where bid >= ask.
#                     Cite: Friedman (1993), Smith (1962).
#   - sealed_bid: All bids simultaneous, highest bidders win.
#                 Cite: Krishna (2002), Vickrey (1961).

MARKET_CONTRACT = {
    "reads": {
        "node_attrs": ["budget", "market_weights"],
        "adj_matrices": [],
        "global_attrs": ["resource_level"],
    },
    "writes": {
        "node_attrs": ["allocations", "budget"],
        "adj_matrices": [],
        "global_attrs": ["clearing_price"],
    },
}

# ---------------------------------------------------------------------------
# Network Type
# ---------------------------------------------------------------------------
# What: Local information sharing along edges. Agents send and receive typed
#        signals through a sparse graph topology.
# Information topology: Local, distributed — signals propagate slowly. Each
#        agent sees only its neighbours.
#
# Important: The network is a *pipe* — it moves data along edges. Trust and
# credibility tracking is a *separate capability* layered on top. A network
# without trust is naive (agents believe everything). Trust is an overlay.
#
# Variants (planned):
#   - broadcast: Each agent shares one signal with all neighbours. No filtering.
#                Cite: basic graph diffusion.
#   - trust_weighted: Adds credibility overlay (EMA of |announced - actual|).
#                     Cite: Ostrom (1990) principle 4 (monitoring).
#   - gossip: Probabilistic forwarding. Information percolates stochastically.
#             Cite: Shah (2009), epidemic protocols.

NETWORK_CONTRACT = {
    "reads": {
        "node_attrs": ["signal"],
        "adj_matrices": ["network"],
        "global_attrs": [],
    },
    "writes": {
        "node_attrs": ["received_signals"],
        "adj_matrices": [],
        "global_attrs": [],
    },
}

# ---------------------------------------------------------------------------
# Democracy Type
# ---------------------------------------------------------------------------
# What: Many-to-one aggregation of preferences, then one-to-many enforcement
#        of policy. The only mechanism type that *imposes constraints on other
#        mechanisms*.
# Information topology: All-to-one then one-to-all. Aggregation collapses
#        many preferences into one decision. Enforcement broadcasts it back.
#
# Democracy configures other mechanisms: the democratic decision can set the
# market's quota cap, the network's transparency requirements, or the harvest
# penalty. It is a meta-mechanism.
#
# Key variable across variants: delegation update cadence.
#   PDD = everyone votes. PRD = delegation locked for T ticks.
#   PLD = delegation revocable every tick.
#
# Variants (planned):
#   - pdd (direct): All vote, median/mean aggregation.
#                   Cite: Condorcet (1785), Arrow (1951).
#   - prd (representative): Fixed representatives elected every T ticks.
#                           Cite: Madison, representative democracy theory.
#   - pld (liquid): Delegated votes, revocable any time.
#                   Cite: Bloembergen et al. (2019), Ford (2002).

DEMOCRACY_CONTRACT = {
    "reads": {
        "node_attrs": ["vote_weights", "last_harvest"],
        "adj_matrices": [],
        "global_attrs": [],
    },
    "writes": {
        "node_attrs": [],
        "adj_matrices": [],
        "global_attrs": ["harvest_target", "penalty_lambda"],
    },
}

# ---------------------------------------------------------------------------
# Composability Note
# ---------------------------------------------------------------------------
# The write sets are disjoint:
#   Market  writes: allocations, budget, clearing_price
#   Network writes: received_signals
#   Democracy writes: harvest_target, penalty_lambda
#
# This means sequential(market, network, democracy) has no write conflicts.
# Each mechanism can be developed, tested, and swapped independently.
