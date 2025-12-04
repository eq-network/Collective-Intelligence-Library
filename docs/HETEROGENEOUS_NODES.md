# Heterogeneous Node Types

The Collective Intelligence Library supports multiple node types in the same graph. This document describes how different nodes (Agents, Markets, Democracies) coexist and interact.

---

## Visual Representation

From the system diagrams:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│        ○ ○ ○           ■           ◇                               │
│       Agent          Market      Democracy                          │
│      (ellipse)      (square)    (diamond)                          │
│                                                                     │
│   Input/Output     Aggregation   Voting                            │
│    Operators       Mechanisms    Mechanisms                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

All three are **nodes in the same graph**. The `node_types` array distinguishes them:

```python
# Example: 10 agents, 2 markets, 1 democracy
state = GraphState(
    node_types=jnp.array([0,0,0,0,0,0,0,0,0,0, 1,1, 2]),
    #                    └─────Agents─────────┘ └M┘ └D┘
    ...
)
```

---

## Node Type Definitions

### Type 0: Agent Nodes

**Visual**: Ellipse (○)

**Purpose**: Input/output operators that receive messages and produce actions

**Behavior**:
```
Messages from neighbors → Agent → Messages to neighbors
```

**Typical Attributes**:
```python
{
    "resources": float,     # What the agent owns
    "beliefs": float,       # What the agent believes
    "reputation": float,    # How trusted by others
}
```

**Example Processing**:
```python
def agent_process_messages(state, node_id, incoming_messages):
    # Aggregate information from messages
    avg_belief = mean([m["belief"] for m in incoming_messages])

    # Update own state based on messages
    return {
        "beliefs": 0.9 * current_belief + 0.1 * avg_belief
    }
```

---

### Type 1: Market Nodes

**Visual**: Square (■)

**Purpose**: Aggregation mechanisms that collect bids and produce prices/allocations

**Behavior**:
```
Bids from agents → Market → Prices, Allocations
```

**Typical Attributes**:
```python
{
    "price": float,           # Current clearing price
    "volume": float,          # Trading volume
    "last_update": int,       # Round of last update
}
```

**Example Processing**:
```python
def market_process_bids(state, node_id, incoming_messages):
    # Incoming messages are bids
    bids = [m for m in incoming_messages if m["type"] == "bid"]

    # Market clearing logic
    buy_orders = [b for b in bids if b["side"] == "buy"]
    sell_orders = [b for b in bids if b["side"] == "sell"]

    clearing_price = find_equilibrium(buy_orders, sell_orders)

    return {
        "price": clearing_price,
        "volume": matched_volume
    }
```

---

### Type 2: Democracy Nodes

**Visual**: Diamond (◇)

**Purpose**: Voting mechanisms that aggregate preferences into collective decisions

**Behavior**:
```
Votes from agents → Democracy → Decisions, Policies
```

**Typical Attributes**:
```python
{
    "current_policy": int,      # Active policy ID
    "vote_threshold": float,    # Required majority
    "last_decision": int,       # Round of last decision
}
```

**Example Processing**:
```python
def democracy_process_votes(state, node_id, incoming_messages):
    # Incoming messages are votes
    votes = [m for m in incoming_messages if m["type"] == "vote"]

    # Tally and decide
    vote_counts = Counter([v["choice"] for v in votes])
    winner = max(vote_counts, key=vote_counts.get)

    return {
        "current_policy": winner,
        "last_decision": state.global_attrs["round"]
    }
```

---

## Message Types on Edges

Edges carry typed messages. The adjacency matrices define connectivity, and message content is determined by the transform:

| Message Type | From | To | Content |
|--------------|------|-----|---------|
| `TrustUpdate` | Agent | Agent | `{trust_delta: float}` |
| `Bid` | Agent | Market | `{side: str, quantity: float, price: float}` |
| `Vote` | Agent | Democracy | `{choice: int, weight: float}` |
| `Price` | Market | Agent | `{price: float, your_allocation: float}` |
| `Policy` | Democracy | Agent | `{new_policy: int, effective_round: int}` |

---

## Sub-Networks

The diagrams show hierarchical groupings:

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────────────┐         ┌─────────────────┐                │
│  │   Country 1     │         │   Country 2     │                │
│  │  ┌───┐ ┌───┐   │         │   ┌───┐ ┌───┐   │                │
│  │  │ A │─│ A │   │         │   │ A │─│ A │   │                │
│  │  └─┬─┘ └─┬─┘   │         │   └─┬─┘ └─┬─┘   │                │
│  │    └──┬──┘     │         │     └──┬──┘     │                │
│  │       │        │         │        │        │                │
│  │     ┌─┴─┐      │         │      ┌─┴─┐      │                │
│  │     │ M │      │ ◄──────►│      │ M │      │                │
│  │     └───┘      │         │      └───┘      │                │
│  └─────────────────┘         └─────────────────┘                │
│                                                                  │
│                    ┌───────────────┐                            │
│                    │  Regulator    │                            │
│                    │    ┌───┐      │                            │
│                    │    │ D │      │◄── Democracy node          │
│                    │    └───┘      │                            │
│                    └───────────────┘                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

Sub-networks are represented via:
1. **Sparse adjacency matrices**: Nodes only connect within their group (dense local, sparse global)
2. **Node attributes**: A `"subnetwork_id"` attribute could mark membership
3. **Multiple edge types**: Different adjacency matrices for intra-group vs inter-group connections

---

## Implementing NodeBehavior

To cleanly handle heterogeneous nodes, define a protocol:

```python
from typing import Protocol, Dict, Any, List

class NodeBehavior(Protocol):
    """Protocol for node-type-specific message processing."""

    def generate_message(self, state: GraphState, node_id: int) -> Message:
        """Generate outgoing message from this node."""
        ...

    def process_messages(
        self,
        state: GraphState,
        node_id: int,
        messages: List[Message]
    ) -> Dict[str, Any]:
        """Process incoming messages and return attribute updates."""
        ...
```

Then dispatch by node type:

```python
def create_heterogeneous_transform(
    behaviors: Dict[int, NodeBehavior]  # node_type → behavior
) -> Transform:
    def transform(state: GraphState) -> GraphState:
        updates = []
        for i in range(state.num_nodes):
            node_type = int(state.node_types[i])
            behavior = behaviors[node_type]

            # Get incoming messages for this node
            incoming = get_messages_for_node(state, i)

            # Dispatch to type-specific behavior
            update = behavior.process_messages(state, i, incoming)
            updates.append(update)

        return apply_updates(state, updates)

    return transform
```

---

## Evolution Over Time

From the simulation diagrams (T=1 → T=4):

```
T=1                    T=2                    T=3                    T=4
┌────────────┐        ┌────────────┐        ┌────────────┐        ┌────────────┐
│ ○   ○   ○  │        │ ○   ●   ○  │        │ ○   ●   ●  │        │ ●   ●   ●  │
│   ■        │   →    │   ■        │   →    │   ■        │   →    │   ■        │
│ ○   ○      │        │ ○   ○      │        │ ●   ○      │        │ ●   ●      │
└────────────┘        └────────────┘        └────────────┘        └────────────┘

● = Adversarial/changed state
```

Key observations:
1. **Node states evolve**: Agent beliefs, resources, trust levels change
2. **Edges can change**: Trust networks rewire based on interactions
3. **Adversarial nodes appear**: Some agents may become adversarial (shown in red)
4. **Sub-network boundaries shift**: The "Regulator system" boundary changes over time

All of this is captured in the GraphState:
- Node state changes → `node_attrs` updates
- Edge changes → `adj_matrices` updates
- Time tracking → `global_attrs["round"]`

---

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Where nodes fit in the layers
- [EXTERNAL_API.md](./EXTERNAL_API.md) - How to implement custom node behaviors
