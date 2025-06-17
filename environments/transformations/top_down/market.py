# transformations/market.py
"""
Pure market transformation for state-to-state resource exchange.
"""
import jax.numpy as jnp
from typing import Callable, Dict, Any, Tuple, List

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from algebra.graph import GraphState
from algebra.property import Property, attach_properties

# Type aliases for clarity
TradePair = Tuple[int, int, float]  # (agent1, agent2, trade_value)
TradeInfo = Dict[str, Any]  # Trade execution details
TradeMatcherFn = Callable[[jnp.ndarray, jnp.ndarray], List[TradePair]]
TradeExecutorFn = Callable[[int, int, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, TradeInfo]]

def create_market_transform(
    preference_attr: str,
    resource_attr: str,
) -> Callable[[GraphState], GraphState]:
    """
    Creates a pure transformation for market-based resource exchange.
    
    This transform requires state to contain:
    - trade_matcher: A function in global_attrs that matches trading pairs
    - trade_executor: A function in global_attrs that executes trades
    - {preference_attr}: Node attributes containing preferences
    - {resource_attr}: Node attributes containing resources
    
    The execution environment is responsible for injecting matcher/executor functions.
    """
    def transform(state: GraphState) -> GraphState:
        # Return unchanged state if required attributes missing
        if (preference_attr not in state.node_attrs or 
            resource_attr not in state.node_attrs or
            "trade_matcher" not in state.global_attrs or
            "trade_executor" not in state.global_attrs):
            return state
        
        # Extract required components from state
        preferences = state.node_attrs[preference_attr]
        resources = state.node_attrs[resource_attr]
        trade_matcher = state.global_attrs["trade_matcher"]
        trade_executor = state.global_attrs["trade_executor"]
        
        # Find potential trading pairs
        trade_pairs = trade_matcher(preferences, resources)
        
        # Execute trades
        new_resources = resources
        executed_trades = []
        
        for i, j, _ in trade_pairs:
            updated_resources, trade_info = trade_executor(i, j, preferences, new_resources)
            if trade_info:
                new_resources = updated_resources
                executed_trades.append(trade_info)
        
        # Update state
        new_global_attrs = dict(state.global_attrs)
        trade_history = new_global_attrs.get("trade_history", [])
        trade_history.append(executed_trades)
        new_global_attrs["trade_history"] = trade_history
        
        return state.replace(
            node_attrs={**state.node_attrs, resource_attr: new_resources},
            global_attrs=new_global_attrs
        )
    
    return transform