# GraphTransform Framework

A principled approach to multi-agent simulation using category theory, functional programming, and JAX acceleration.

You can see the diagrams here: https://excalidraw.com/#room=f4116b0ba2d8d5095d85,zSDwGDuqMZI4uxu4CTQuHg

## Quick Start

### Running Experiments

The main entry point for running simulations is `experiments/run_portfolio_experiment.py`. This script defines and executes complete experimental suites:

```bash
cd experiments
python run_portfolio_experiment.py
```

This will run the predefined experiments, which test different democratic mechanisms (PDD, PRD, PLD) across various conditions.

### Basic Usage Pattern

1. **Define Experiments** - Configure what you want to test
2. **Run Simulations** - Execute experiments in parallel
3. **Analyze Results** - Generate visualizations and statistics

Here's how the pieces fit together:

```python
# 1. Define an experiment in run_portfolio_experiment.py
experiment = ExperimentDefinition(
    name="MyExperiment",
    config_factory_func_name="create_stable_democracy_config",
    mechanisms_to_test=["PDD", "PRD", "PLD"],
    adversarial_proportions_to_sweep=[0.0, 0.2, 0.4],
    num_replications_per_setting=5,
    base_seed_for_experiment=42
)

# 2. The system automatically:
# - Generates all parameter combinations
# - Runs simulations in parallel
# - Collects timeline data from each simulation
# - Aggregates results and generates analysis
```

## System Architecture

### Environments

The system supports different simulation environments, each with its own configuration and mechanisms:

#### `environments/stable_democracy/`
- **Purpose**: Deterministic simulations with perfect information
- **Configuration**: `StablePortfolioDemocracyConfig`
- **Factory**: `create_stable_democracy_config()`
- **Features**: Participation constraints, multiple adversarial framings

#### `environments/noise_democracy/`
- **Purpose**: Realistic simulations with information noise and cognitive constraints
- **Configuration**: `PortfolioDemocracyConfig` 
- **Factories**: `create_thesis_baseline_config()`, `create_thesis_highvariance_config()`
- **Features**: Cognitive resource modeling, prediction market noise

### Democratic Mechanisms

Each environment implements three core democratic mechanisms:

- **PDD (Predictive Direct Democracy)**: One-agent-one-vote with prediction market information
- **PRD (Predictive Representative Democracy)**: Elected representatives make decisions
- **PLD (Predictive Liquid Democracy)**: Agents can delegate their voting power to others

### Experiment Framework

```
experiments/
├── run_portfolio_experiment.py    # Main entry point - define experiments here
├── experiment_config.py           # Experiment definition structures
├── runner.py                     # Parallel execution engine
├── worker.py                     # Individual simulation execution
├── results.py                    # Result aggregation and storage
├── analysis.py                   # Visualization and statistical analysis
└── progress_tracker.py           # Real-time progress monitoring
```

## Customizing Experiments

### 1. Modify Existing Experiments

Edit `run_portfolio_experiment.py` to change experiment parameters:

```python
def define_all_experiments() -> List[ExperimentDefinition]:
    experiments = []
    
    # Customize this experiment
    my_experiment = ExperimentDefinition(
        name="CustomTest",
        config_factory_func_name="create_stable_democracy_config",
        mechanisms_to_test=["PLD"],  # Test only liquid democracy
        adversarial_proportions_to_sweep=[0.1, 0.3, 0.5],  # Custom proportions
        num_replications_per_setting=10,  # More replications
        base_seed_for_experiment=12345,
        llm_model='openai/gpt-4o-mini',  # Specify LLM model
        adversarial_framing="competitive"  # Adversarial agent framing
    )
    experiments.append(my_experiment)
    
    return experiments
```

### 2. Create New Configurations

Add new configuration variants in the appropriate environment:

```python
# In environments/stable_democracy/configuration.py
def create_my_custom_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    adversarial_proportion_total: float,
    seed: int = 42,
    # Your custom parameters
    custom_parameter: float = 1.0
) -> StablePortfolioDemocracyConfig:
    # Your custom configuration logic
    return create_stable_democracy_config(
        mechanism=mechanism,
        adversarial_proportion_total=adversarial_proportion_total,
        seed=seed,
        # Apply your customizations
        num_agents=20,  # Different agent count
        delegate_participation_rate=0.8,  # Custom participation
        # etc.
    )
```

Then register it in `experiments/worker.py`:

```python
CONFIG_FACTORIES = {
    "create_stable_democracy_config": create_stable_democracy_config,
    "create_my_custom_config": create_my_custom_config,  # Add your factory
    # ... other factories
}
```

### 3. Add LLM Integration

The system supports optional LLM integration for agent decision-making. Configure by setting the `llm_model` parameter and ensuring you have the appropriate API key:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

Supported model formats (via OpenRouter):
- `'openai/gpt-4o-mini'`
- `'google/gemini-2.5-flash-preview-05-20'`
- `'anthropic/claude-3.5-haiku'`

## Output and Analysis

### Generated Files

When you run experiments, the system creates:

```
experiment_outputs/
└── TimelinePortfolioDemocracySuite_YYYYMMDD_HHMMSS/
    ├── ExperimentName_TIMESTAMP/
    │   ├── aggregated_results_timeline_data_TIMESTAMP.csv.gz    # Raw timeline data
    │   ├── aggregated_results_metadata_TIMESTAMP.csv           # Simulation metadata
    │   ├── aggregated_results_anomaly_logs_TIMESTAMP.csv.gz    # Behavioral anomalies
    │   ├── individual_trajectories_TIMESTAMP.png              # Sample trajectories
    │   ├── aggregated_trajectories_TIMESTAMP.png              # Mechanism comparison
    │   ├── resource_change_distributions_TIMESTAMP.png        # Round-to-round changes
    │   └── timeline_summary_stats_TIMESTAMP.csv               # Statistical summary
    └── [Additional experiments...]
```

### Key Metrics

The system tracks comprehensive metrics including:

- **Resource Trajectories**: Round-by-round resource levels for each simulation
- **Decision Quality**: Optimality of portfolio choices relative to available information
- **Mechanism Performance**: Comparative effectiveness across democratic systems
- **Behavioral Anomalies**: Detection of unexpected agent behaviors
- **Participation Patterns**: Agent engagement and delegation dynamics

## Understanding Results

### Timeline Data Structure

Each simulation generates timeline data with one row per round:

```csv
run_id,round,resources_after,chosen_portfolio_idx,mechanism,adversarial_proportion_total,...
1,0,105.2,2,PLD,0.2,...
1,1,98.7,1,PLD,0.2,...
1,2,103.1,0,PLD,0.2,...
```

### Visualization Types

1. **Individual Trajectories**: Show resource progression for sample simulations
2. **Aggregated Trajectories**: Compare mechanism performance with confidence intervals
3. **Distribution Analysis**: Examine round-to-round resource change patterns
4. **Mechanism Comparison**: Performance across different adversarial conditions

## Core Principles

GraphTransform is built on foundational mathematical and computational principles that provide a rigorous basis for modeling complex multi-agent systems.

### From Category Theory to Code

At its heart, GraphTransform implements category theory concepts directly in code:

- **Morphisms as Pure Functions**: Transformations are morphisms in the category of graph states
- **Composition as a First-Class Operation**: Sequential and parallel composition of transformations
- **Invariant Preservation**: Transformations can be characterized by the properties they preserve
- **Type Safety**: Mathematical properties encoded in the type system

This category-theoretic foundation enables us to reason about transformations mathematically while implementing them computationally.

### Functional Paradigm

The framework embraces functional programming principles:

- **Immutability**: Graph states are immutable, transformations produce new states
- **Pure Functions**: Transformations have no side effects
- **Function Composition**: Complex behaviors built from simple composable parts
- **Higher-Order Functions**: Transformations that operate on other transformations
- **Referential Transparency**: Identical inputs always produce identical outputs

### The Two-Layer Architecture

GraphTransform separates **what** happens from **how** it happens through a clean two-layer architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Process Layer (Mathematical Definition)                             │
│                                                                     │
│  • Graph transformations as typed, composable operations            │
│  • Mathematical properties encoded and verified                     │
│  • Algebraic laws governing composition                             │
│  • Scale-independent, platform-independent definitions              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Execution Layer (Computational Implementation)                      │
│                                                                     │
│  • Optimization of computational resources                          │
│  • Hardware-specific acceleration (JAX)                             │
│  • Service integration (LLMs, storage)                              │
│  • Performance monitoring and adaptation                            │
└─────────────────────────────────────────────────────────────────────┘
```

This separation ensures mathematical rigor while enabling computational efficiency.

## Conceptual Framework

### Bottom-Up vs. Top-Down Processes

The framework distinguishes between two fundamental classes of transformations:

**Bottom-Up Communication**:
- Agent-to-agent interactions
- Information generation and exchange
- Belief updating through local interactions
- Emergent patterns from local rules

**Top-Down Regularization**:
- Global coordination mechanisms
- Constraint enforcement
- Collective decision-making
- Resource allocation systems

This distinction mirrors how complex systems in nature operate: local interactions produce emergent behaviors, while global constraints shape the overall system dynamics.

### Graph Monads

The core data structure is the `GraphState`, which functions as a monad in the category-theoretic sense:

- It encapsulates a complete system state
- It provides operations for transformation
- It maintains immutability
- It enables composition of operations

This monad-based approach gives us a mathematically sound way to represent and transform complex system states.

## Installation

```bash
pip install graph-transform
```

## Further Reading

- [Category Theory for Programmers](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers-the-preface/)
- [Functional Programming in Python](https://docs.python.org/3/howto/functional.html)
- [JAX Documentation](https://jax.readthedocs.io/)
- [A Process-Centric Multi-Agent Simulation Manifesto](/Manifesto.md)

## License

MIT License
