"""
Registry: Database of available agent types, environment types, and mechanisms.

Provides a centralized registry for the studio to discover and configure
available components from the engine.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional, Type
from enum import Enum


class AgentCategory(Enum):
    """Categories of agents."""
    GENERAL = "general"           # Generic, reusable agents
    ALIGNED = "aligned"           # Cooperative/aligned agents
    ADVERSARIAL = "adversarial"   # Adversarial/red team agents
    ENVIRONMENT = "environment"   # Environment-specific agents


class EnvironmentCategory(Enum):
    """Categories of environments/scenarios."""
    MARKET = "market"             # Market/trading scenarios
    DEMOCRACY = "democracy"       # Voting/governance scenarios
    RESOURCE = "resource"         # Resource management scenarios
    GAME = "game"                 # Game theory scenarios


@dataclass
class AgentTypeInfo:
    """Information about an available agent type."""
    id: str
    name: str
    description: str
    category: AgentCategory
    module_path: str
    class_name: Optional[str] = None
    factory_fn: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    default_attrs: Dict[str, Any] = field(default_factory=dict)
    compatible_environments: List[str] = field(default_factory=list)


@dataclass
class EnvironmentTypeInfo:
    """Information about an available environment type."""
    id: str
    name: str
    description: str
    category: EnvironmentCategory
    module_path: str
    factory_fn: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    default_config: Dict[str, Any] = field(default_factory=dict)
    node_types: List[Dict[str, Any]] = field(default_factory=list)
    resource_types: List[str] = field(default_factory=list)


@dataclass
class MechanismInfo:
    """Information about an available mechanism/transform."""
    id: str
    name: str
    description: str
    module_path: str
    factory_fn: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_attrs: List[str] = field(default_factory=list)
    output_attrs: List[str] = field(default_factory=list)


# =============================================================================
# AGENT REGISTRY
# =============================================================================

AGENT_TYPES: Dict[str, AgentTypeInfo] = {
    # General-Purpose Agents
    "random": AgentTypeInfo(
        id="random",
        name="Random Agent",
        description="Makes random decisions from available actions",
        category=AgentCategory.GENERAL,
        module_path="engine.agents.random_agent",
        class_name="RandomAgent",
        parameters={
            "seed": {"type": "int", "default": 42, "description": "Random seed"}
        },
        compatible_environments=["*"]
    ),
    "greedy": AgentTypeInfo(
        id="greedy",
        name="Greedy Agent",
        description="Selects action with highest score from candidates",
        category=AgentCategory.GENERAL,
        module_path="engine.agents.greedy_agent",
        class_name="GreedyAgent",
        parameters={
            "score_fn": {"type": "callable", "description": "Function to score actions"},
            "action_generator": {"type": "callable", "description": "Function to generate candidate actions"}
        },
        compatible_environments=["*"]
    ),
    "llm": AgentTypeInfo(
        id="llm",
        name="LLM Agent",
        description="Uses a language model for decision making",
        category=AgentCategory.GENERAL,
        module_path="engine.agents.llm_agent",
        class_name="LLMAgent",
        parameters={
            "llm_service": {"type": "service", "description": "LLM service instance"}
        },
        compatible_environments=["*"]
    ),

    # Democracy-Specific Agents
    "hardcoded_aligned": AgentTypeInfo(
        id="hardcoded_aligned",
        name="Hardcoded Aligned Agent",
        description="Always selects highest-yield option (baseline)",
        category=AgentCategory.ALIGNED,
        module_path="engine.agents.democracy.hardcoded",
        class_name="HardcodedAlignedAgent",
        parameters={
            "role": {"type": "str", "default": "Voter"},
            "mechanism": {"type": "str", "default": "PLD"}
        },
        compatible_environments=["democracy"]
    ),
    "hardcoded_adversarial": AgentTypeInfo(
        id="hardcoded_adversarial",
        name="Hardcoded Adversarial Agent",
        description="Always selects lowest-yield option (baseline)",
        category=AgentCategory.ADVERSARIAL,
        module_path="engine.agents.democracy.hardcoded",
        class_name="HardcodedAdversarialAgent",
        parameters={
            "role": {"type": "str", "default": "Voter"},
            "mechanism": {"type": "str", "default": "PLD"}
        },
        compatible_environments=["democracy"]
    ),
    "red_team": AgentTypeInfo(
        id="red_team",
        name="Red Team Agent",
        description="LLM-driven adversarial agent with red team framing",
        category=AgentCategory.ADVERSARIAL,
        module_path="engine.agents.democracy.llm_agents",
        class_name="RedTeamAgent",
        parameters={
            "llm_service": {"type": "service", "description": "LLM service instance"},
            "role": {"type": "str", "default": "Voter"}
        },
        compatible_environments=["democracy"]
    ),
    "aligned_heuristic": AgentTypeInfo(
        id="aligned_heuristic",
        name="Aligned Heuristic Agent",
        description="LLM-driven agent with cooperative heuristics",
        category=AgentCategory.ALIGNED,
        module_path="engine.agents.democracy.llm_agents",
        class_name="AlignedHeuristicAgent",
        parameters={
            "llm_service": {"type": "service", "description": "LLM service instance"},
            "role": {"type": "str", "default": "Voter"}
        },
        compatible_environments=["democracy"]
    ),

    # Farmers Market Agents
    "diversity_farmer": AgentTypeInfo(
        id="diversity_farmer",
        name="Diversity Farmer",
        description="Seeks balanced resource portfolio (minimizes variance)",
        category=AgentCategory.ENVIRONMENT,
        module_path="engine.environments.farmers_market.agent_configs",
        factory_fn="create_diversity_farmer",
        default_attrs={"resources": 100.0},
        compatible_environments=["farmers_market"]
    ),
    "accumulator_farmer": AgentTypeInfo(
        id="accumulator_farmer",
        name="Accumulator Farmer",
        description="Maximizes total resources (greedy accumulation)",
        category=AgentCategory.ENVIRONMENT,
        module_path="engine.environments.farmers_market.agent_configs",
        factory_fn="create_accumulator_farmer",
        default_attrs={"resources": 100.0},
        compatible_environments=["farmers_market"]
    ),
    "trader_farmer": AgentTypeInfo(
        id="trader_farmer",
        name="Trader Farmer",
        description="Evaluates trade benefits, trades surplus resources",
        category=AgentCategory.ENVIRONMENT,
        module_path="engine.environments.farmers_market.agent_configs",
        factory_fn="create_trader_farmer",
        default_attrs={"resources": 100.0},
        compatible_environments=["farmers_market"]
    ),
    "random_farmer": AgentTypeInfo(
        id="random_farmer",
        name="Random Farmer",
        description="Makes random trading and consumption decisions",
        category=AgentCategory.ENVIRONMENT,
        module_path="engine.environments.farmers_market.agent_configs",
        factory_fn="create_random_farmer",
        parameters={
            "seed": {"type": "int", "default": 42}
        },
        default_attrs={"resources": 100.0},
        compatible_environments=["farmers_market"]
    ),
}


# =============================================================================
# ENVIRONMENT REGISTRY
# =============================================================================

ENVIRONMENT_TYPES: Dict[str, EnvironmentTypeInfo] = {
    "farmers_market": EnvironmentTypeInfo(
        id="farmers_market",
        name="Farmers Market",
        description="Agents trade resources in a local market. Explores price discovery and resource allocation.",
        category=EnvironmentCategory.MARKET,
        module_path="engine.environments.farmers_market.state",
        factory_fn="create_farmers_market_state",
        parameters={
            "num_farmers": {"type": "int", "default": 10, "min": 2, "max": 100},
            "network_density": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
            "seed": {"type": "int", "default": 42}
        },
        default_config={
            "resource_types": ["apples", "wheat", "corn"],
            "initial_resources_per_farmer": {"apples": 100.0, "wheat": 100.0, "corn": 100.0}
        },
        node_types=[
            {"id": 0, "name": "Farmer", "shape": "circle", "color": "#87CEEB"},
        ],
        resource_types=["apples", "wheat", "corn"]
    ),
    "simple_resource": EnvironmentTypeInfo(
        id="simple_resource",
        name="Simple Resource Game",
        description="Basic resource sharing game for testing agent behaviors",
        category=EnvironmentCategory.RESOURCE,
        module_path="examples.clean_architecture_demo",
        factory_fn="SimpleResourceGame.create_initial_state",
        parameters={
            "num_agents": {"type": "int", "default": 5, "min": 2, "max": 50},
            "initial_resources": {"type": "float", "default": 100.0},
            "seed": {"type": "int", "default": 42}
        },
        default_config={},
        node_types=[
            {"id": 0, "name": "Agent", "shape": "circle", "color": "#87CEEB"},
        ],
        resource_types=["resources"]
    ),
    "tragedy_commons": EnvironmentTypeInfo(
        id="tragedy_commons",
        name="Tragedy of the Commons",
        description="Shared resource depletion. Explores cooperation vs defection dynamics.",
        category=EnvironmentCategory.RESOURCE,
        module_path="",  # Not yet implemented
        factory_fn="",
        parameters={
            "num_agents": {"type": "int", "default": 8, "min": 2, "max": 30},
            "shared_resource": {"type": "float", "default": 1000.0},
            "regeneration_rate": {"type": "float", "default": 0.1}
        },
        default_config={},
        node_types=[
            {"id": 0, "name": "Agent", "shape": "circle", "color": "#87CEEB"},
            {"id": 1, "name": "Commons", "shape": "diamond", "color": "#90EE90"},
        ],
        resource_types=["commons"]
    ),
    "democracy_pld": EnvironmentTypeInfo(
        id="democracy_pld",
        name="Prediction Liquid Democracy",
        description="Voting mechanism with prediction market and delegation",
        category=EnvironmentCategory.DEMOCRACY,
        module_path="",  # Partially implemented
        factory_fn="",
        parameters={
            "num_voters": {"type": "int", "default": 10, "min": 2, "max": 100},
            "num_portfolios": {"type": "int", "default": 5},
            "cognitive_resource_mean": {"type": "float", "default": 50.0}
        },
        default_config={},
        node_types=[
            {"id": 0, "name": "Voter", "shape": "circle", "color": "#87CEEB"},
            {"id": 1, "name": "Delegate", "shape": "circle", "color": "#FFD700"},
        ],
        resource_types=["cognitive_resources", "voting_power"]
    ),
}


# =============================================================================
# MECHANISM REGISTRY
# =============================================================================

MECHANISM_TYPES: Dict[str, MechanismInfo] = {
    "resource_growth": MechanismInfo(
        id="resource_growth",
        name="Resource Growth",
        description="Multiplies resources by growth rate each round",
        module_path="engine.environments.farmers_market.agent_transforms",
        factory_fn="create_resource_growth_transform",
        parameters={},
        input_attrs=["resources", "growth_rate"],
        output_attrs=["resources"]
    ),
    "resource_consumption": MechanismInfo(
        id="resource_consumption",
        name="Resource Consumption",
        description="Agents consume portion of their resources",
        module_path="studio.screens.scenario",
        factory_fn="consumption_transform",
        parameters={
            "normal_rate": {"type": "float", "default": 0.05},
            "adversarial_rate": {"type": "float", "default": 0.15}
        },
        input_attrs=["resources", "node_types"],
        output_attrs=["resources"]
    ),
    "trade": MechanismInfo(
        id="trade",
        name="Agent Trading",
        description="Agents exchange resources based on trade offers",
        module_path="engine.environments.farmers_market.agent_transforms",
        factory_fn="create_agent_driven_trade_transform",
        parameters={},
        input_attrs=["resources", "trade_network"],
        output_attrs=["resources", "total_trades"]
    ),
    "message_passing": MechanismInfo(
        id="message_passing",
        name="Message Passing",
        description="Information diffusion along network edges",
        module_path="engine.transformations.bottom_up.message_passing",
        factory_fn="create_message_passing_transform",
        parameters={
            "connection_type": {"type": "str", "default": "connections"}
        },
        input_attrs=["*"],
        output_attrs=["*"]
    ),
    "prediction_market": MechanismInfo(
        id="prediction_market",
        name="Prediction Market",
        description="Injects market signals with agent-specific noise",
        module_path="engine.transformations.bottom_up.prediction_market",
        factory_fn="create_prediction_market_transform",
        parameters={
            "base_noise": {"type": "float", "default": 0.1}
        },
        input_attrs=["predictions", "cognitive_resources"],
        output_attrs=["market_signals"]
    ),
}


# =============================================================================
# NODE TYPE REGISTRY
# =============================================================================

NODE_TYPES: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "Agent",
        "shape": "circle",
        "color": "#87CEEB",
        "description": "Regular agent node"
    },
    1: {
        "name": "Adversarial",
        "shape": "circle",
        "color": "#FF6B6B",
        "description": "Adversarial agent (higher consumption)"
    },
    2: {
        "name": "Market",
        "shape": "square",
        "color": "#FFA500",
        "description": "Market mechanism node"
    },
    3: {
        "name": "Resource",
        "shape": "diamond",
        "color": "#90EE90",
        "description": "Resource depot node"
    },
}


# =============================================================================
# REGISTRY ACCESS FUNCTIONS
# =============================================================================

def get_agent_types(category: Optional[AgentCategory] = None) -> List[AgentTypeInfo]:
    """Get all agent types, optionally filtered by category."""
    types = list(AGENT_TYPES.values())
    if category:
        types = [t for t in types if t.category == category]
    return types


def get_environment_types(category: Optional[EnvironmentCategory] = None) -> List[EnvironmentTypeInfo]:
    """Get all environment types, optionally filtered by category."""
    types = list(ENVIRONMENT_TYPES.values())
    if category:
        types = [t for t in types if t.category == category]
    return types


def get_compatible_agents(environment_id: str) -> List[AgentTypeInfo]:
    """Get agent types compatible with a specific environment."""
    return [
        agent for agent in AGENT_TYPES.values()
        if "*" in agent.compatible_environments or environment_id in agent.compatible_environments
    ]


def get_mechanism_types() -> List[MechanismInfo]:
    """Get all available mechanism types."""
    return list(MECHANISM_TYPES.values())


def get_node_type_info(type_id: int) -> Optional[Dict[str, Any]]:
    """Get information about a node type."""
    return NODE_TYPES.get(type_id)


# =============================================================================
# REGISTRY SUMMARY (for debugging/display)
# =============================================================================

def print_registry_summary():
    """Print a summary of the registry contents."""
    print("=" * 60)
    print("MYCORRHIZA REGISTRY SUMMARY")
    print("=" * 60)

    print(f"\nAgent Types ({len(AGENT_TYPES)}):")
    for cat in AgentCategory:
        agents = get_agent_types(cat)
        if agents:
            print(f"  {cat.value.title()} ({len(agents)}):")
            for a in agents:
                print(f"    - {a.name}: {a.description[:50]}...")

    print(f"\nEnvironment Types ({len(ENVIRONMENT_TYPES)}):")
    for cat in EnvironmentCategory:
        envs = get_environment_types(cat)
        if envs:
            print(f"  {cat.value.title()} ({len(envs)}):")
            for e in envs:
                status = "[x]" if e.factory_fn else "[ ]"
                print(f"    {status} {e.name}: {e.description[:40]}...")

    print(f"\nMechanism Types ({len(MECHANISM_TYPES)}):")
    for m in MECHANISM_TYPES.values():
        print(f"  - {m.name}: {m.description[:50]}...")

    print(f"\nNode Types ({len(NODE_TYPES)}):")
    for type_id, info in NODE_TYPES.items():
        print(f"  {type_id}: {info['name']} ({info['shape']}, {info['color']})")


if __name__ == "__main__":
    print_registry_summary()
