"""
Market Transform for Graph Architecture - Fully Resource-Agnostic
"""

import jax.numpy as jnp
from typing import Optional, Tuple
from core.graph import GraphState
from core.category import Transform


def create_double_auction_transform(
    resource_pair: Tuple[str, str],
    price_key: Optional[str] = None,
    trade_rate: float = 0.2,
) -> Transform:
    """
    Build a double-auction transform for ANY two resources.

    Args:
        resource_pair: (res_a, res_b) traded in the auction (A priced in B).
        price_key: Optional global attribute name to read/write the market price.
                  If None, price = mean(valuations) each tick (not persisted).
        trade_rate: Fraction of each agent's inventory they are willing to trade
                   when they are on the supply/demand side (0..1).

    Returns:
        Transform: Pure function GraphState -> GraphState

    Example:
        market = create_double_auction_transform(
            resource_pair=("apples", "oranges"),
            price_key="price_apples_oranges",
            trade_rate=0.2
        )
        new_state = market(old_state)
    """
    res_a, res_b = resource_pair

    def auction_transform(state: GraphState) -> GraphState:
        # Skip if resources missing
        if res_a not in state.node_attrs or res_b not in state.node_attrs:
            return state

        # Resource holdings per agent
        holdings_a = jnp.asarray(state.node_attrs[res_a])
        holdings_b = jnp.asarray(state.node_attrs[res_b])

        # Scarcity-based valuation: agents value A by relative abundance of B
        total_a = jnp.sum(holdings_a)
        total_b = jnp.sum(holdings_b)
        base_price = (total_b + 1.0) / (total_a + 1.0)
        valuations = base_price * (holdings_b + 1.0) / (holdings_a + 1.0)

        # Current market price (from storage or fresh calculation)
        market_price = (
            state.global_attrs.get(price_key, jnp.mean(valuations))
            if price_key
            else jnp.mean(valuations)
        )

        # FIXED: Classify agents correctly
        # High valuation → agent values apples highly → wants to BUY apples
        # Low valuation → agent values apples lowly → wants to SELL apples
        wants_a = valuations > market_price  # Buyers (want to acquire A)
        wants_b = valuations < market_price  # Sellers (want to dispose A)

        # Aggregate supply and demand
        supply_a = jnp.where(wants_b, holdings_a * trade_rate, 0.0)
        demand_a = jnp.where(wants_a, holdings_b / market_price * trade_rate, 0.0)

        total_supply = jnp.sum(supply_a)
        total_demand = jnp.sum(demand_a)

        # Market clearing
        if (total_supply > 0) & (total_demand > 0):
            traded_quantity = jnp.minimum(total_supply, total_demand)
            sell_ratio = traded_quantity / jnp.maximum(total_supply, 1e-12)
            buy_ratio = traded_quantity / jnp.maximum(total_demand, 1e-12)

            actual_sales = supply_a * sell_ratio
            actual_purchases = demand_a * buy_ratio

            # Update holdings (with floor at zero)
            new_holdings_a = jnp.maximum(
                holdings_a - actual_sales + actual_purchases, 0.0
            )
            new_holdings_b = jnp.maximum(
                holdings_b
                + actual_sales * market_price
                - actual_purchases * market_price,
                0.0,
            )

            # Price adjustment: excess demand pushes price up
            excess = (total_demand - total_supply) / (
                total_demand + total_supply + 1e-2
            )
            new_price = market_price * (1.0 + 0.1 * excess)
        else:
            # No trade possible: hold steady
            new_holdings_a = holdings_a
            new_holdings_b = holdings_b
            new_price = market_price
            traded_quantity = 0.0

        # Update state immutably
        new_state = state.update_node_attrs(res_a, new_holdings_a)
        new_state = new_state.update_node_attrs(res_b, new_holdings_b)

        # Store price if key provided
        if price_key:
            new_state = new_state.update_global_attr(price_key, float(new_price))

        # Diagnostics (for UI/logging)
        new_state = new_state.update_global_attr(
            "last_trade_volume", float(traded_quantity)
        )
        new_state = new_state.update_global_attr("last_price", float(new_price))

        return new_state

    return auction_transform
