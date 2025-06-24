# environments/democracy/stable/initialization.py
import jax.numpy as jnp
import importlib
from typing import List, Dict, Tuple
from core.agents import Agent
from dataclasses import dataclass, field

from .configuration import StableDemocracyConfig
from core.graph import GraphState
from services.llm import LLMService

def initialize_environment(config: StableDemocracyConfig, llm_service: LLMService) -> Tuple[GraphState, List[Agent]]:
    """Initializes the heterogeneous graph and instantiates agent objects."""
    
    # 1. Instantiate Agent Objects
    agents = []
    agent_node_indices = []
    node_type_map = {}
    
    current_index = 0
    for component in config.graph_components:
        node_type_map[component.node_type_id] = component.name
        if component.agent_class_path:
            AgentClass = _import_class(component.agent_class_path)
            for _ in range(component.count):
                agent_id = current_index
                agents.append(AgentClass(agent_id=agent_id, llm_service=llm_service, role=component.name))
                agent_node_indices.append(agent_id)
                current_index += 1
        else:
            # For non-agent nodes
            current_index += component.count
            
    num_total_nodes = current_index
    
    # 2. Build the GraphState (the SoA representation)
    node_types = []
    initial_attrs = {
        # Initialize all potential attributes with correctly shaped empty arrays
        'is_adversarial': jnp.zeros(num_total_nodes, dtype=jnp.bool_),
        'portfolio_votes': jnp.zeros((num_total_nodes, len(config.portfolios)), dtype=jnp.int32),
        'decision': jnp.array([-1] * num_total_nodes, dtype=jnp.int32),
        'signals': jnp.zeros((num_total_nodes, len(config.portfolios)), dtype=jnp.float32),
    }

    node_idx_offset = 0
    for comp in config.graph_components:
        for i in range(comp.count):
            node_types.append(comp.node_type_id)
            # Set initial attributes for this component
            for attr, value in comp.initial_attrs.items():
                initial_attrs[attr] = initial_attrs[attr].at[node_idx_offset + i].set(value)
        node_idx_offset += comp.count

    # 3. Define Adjacency Matrices for message passing
    adj_matrices = {
        'market_to_agents': jnp.zeros((num_total_nodes, num_total_nodes)),
        'agents_to_aggregator': jnp.zeros((num_total_nodes, num_total_nodes)),
    }
    
    # Find node indices by type
    market_nodes = [i for i, t in enumerate(node_types) if t == 1] # Assuming 1 = Market
    aggregator_nodes = [i for i, t in enumerate(node_types) if t == 2] # Assuming 2 = Aggregator
    
    # Connect market to agents
    for market_idx in market_nodes:
        adj_matrices['market_to_agents'] = adj_matrices['market_to_agents'].at[market_idx, agent_node_indices].set(1)
        
    # Connect agents to aggregator
    for agg_idx in aggregator_nodes:
        adj_matrices['agents_to_aggregator'] = adj_matrices['agents_to_aggregator'].at[agent_node_indices, agg_idx].set(1)
        
    # Create the final GraphState
    initial_state = GraphState(
        node_types=jnp.array(node_types),
        node_attrs=initial_attrs,
        adj_matrices=adj_matrices,
        global_attrs={"resources": config.initial_resources, ...}
    )

    return initial_state, agents

def _import_class(class_path: str):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)