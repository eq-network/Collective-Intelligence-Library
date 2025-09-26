"""
Command-line interface for running democratic mechanism experiments.

This module provides the primary entry point for running democratic mechanism
experiments with the graph transformation framework. It orchestrates the complete
experiment lifecycle, from configuration to results analysis.
"""
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr
import argparse
import json
import os
from datetime import datetime

from core.graph import GraphState
from core.category import sequential, compose, attach_properties, jit_transform
from core.property import ConservesSum, Property
from environments.democracy.initialization import initialize_democratic_graph_state
from execution.simulation import run_simulation
from analysis.democracy import compute_metrics, aggregate_results

# Type aliases for clean type signatures
Transform = Callable[[GraphState], GraphState]
RandomKey = jnp.ndarray


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for democratic mechanism experiments."""
    # Mechanism configuration
    mechanism_type: str  # "PDD", "PRD", or "PLD"
    
    # Population parameters
    num_agents: int
    adversarial_proportion: float
    adversarial_introduction: str  # "immediate" or "gradual"
    
    # Environment parameters
    num_rounds: int
    initial_resources: int
    resource_min_threshold: int
    crops: List[str]
    yield_volatility: str  # "stable" or "variable"
    
    # Execution parameters
    random_seed: int
    num_trials: int
    jit_compile: bool = True
    
    # Analysis flags
    track_delegation_metrics: bool = True
    track_resource_metrics: bool = True
    track_information_metrics: bool = True


def create_initial_state(config: ExperimentConfig, key: RandomKey) -> GraphState:
    """
    Map from experiment configuration to initialization parameters.
    
    This function bridges the ExperimentConfig structure to the domain-specific
    initialization functions in the democracy module.
    
    Args:
        config: Experiment configuration
        key: Random key for initialization
        
    Returns:
        Initialized graph state
    """
    # Map configuration to parameter dictionaries
    adversarial_params = {
        "proportion": config.adversarial_proportion,
        "introduction": config.adversarial_introduction
    }
    
    crop_params = {
        "volatility_params": {
            config.yield_volatility: {"alpha": (5.0, 7.0), "beta": (3.0, 5.0)},
            "other_volatility": {"alpha": (2.0, 5.0), "beta": (1.0, 3.0)}
        }
    }
    
    # Initialize graph state using domain-specific initialization
    return initialize_democratic_graph_state(
        num_agents=config.num_agents,
        crops=config.crops,
        initial_resources=config.initial_resources,
        resource_min_threshold=config.resource_min_threshold,
        adversarial_proportion=config.adversarial_proportion,
        adversarial_introduction=config.adversarial_introduction,
        yield_volatility=config.yield_volatility,
        key=key
    )


def run_democratic_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a complete democratic mechanism experiment.
    
    This function orchestrates the full experiment pipeline:
    1. Mechanism construction
    2. Initial state creation
    3. Simulation execution
    4. Results analysis
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dict containing structured experiment results and analysis metrics
    """
    # Initialize master random key from seed
    master_key = jr.PRNGKey(config.random_seed)
    
    # Import mechanism construction function
    from graph_transformation.transformations.bottom_up.mechanisms import construct_mechanism
    
    # Prepare transformation pipeline based on mechanism type
    transform_pipeline = construct_mechanism(config.mechanism_type)
    
    # Apply optimization if requested
    if config.jit_compile:
        transform_pipeline = jit_transform(transform_pipeline)
    
    # Execute trials with different random subkeys
    trial_results = []
    
    for trial in range(config.num_trials):
        # Generate trial-specific key
        master_key, trial_key = jr.split(master_key)
        
        # Initialize graph state for this trial
        initial_state = create_initial_state(config, trial_key)
        
        # Execute simulation for this trial
        final_state, state_history = run_simulation(
            initial_state=initial_state,
            transform=transform_pipeline,
            num_rounds=config.num_rounds,
            key=trial_key
        )
        
        # Compute trial metrics
        metrics = compute_metrics(
            config=config,
            final_state=final_state,
            state_history=state_history
        )
        
        trial_results.append(metrics)
    
    # Aggregate results across trials
    aggregated_results = aggregate_results(trial_results)
    
    # Return structured results with appropriate metadata
    return {
        "config": config,
        "aggregated_results": aggregated_results,
        "trial_results": trial_results
    }


def run_comparative_analysis(base_config: ExperimentConfig, mechanisms: List[str]) -> Dict[str, Any]:
    """
    Run comparative analysis across multiple democratic mechanisms.
    
    Args:
        base_config: Base experiment configuration
        mechanisms: List of mechanism types to compare
        
    Returns:
        Dict containing comparative analysis results
    """
    # Run experiments for each mechanism
    mechanism_results = {}
    
    for mechanism in mechanisms:
        # Create mechanism-specific configuration
        mechanism_config = ExperimentConfig(
            mechanism_type=mechanism,
            num_agents=base_config.num_agents,
            adversarial_proportion=base_config.adversarial_proportion,
            adversarial_introduction=base_config.adversarial_introduction,
            num_rounds=base_config.num_rounds,
            initial_resources=base_config.initial_resources,
            resource_min_threshold=base_config.resource_min_threshold,
            crops=base_config.crops,
            yield_volatility=base_config.yield_volatility,
            random_seed=base_config.random_seed,
            num_trials=base_config.num_trials,
            jit_compile=base_config.jit_compile
        )
        
        # Run experiment
        results = run_democratic_experiment(mechanism_config)
        mechanism_results[mechanism] = results
    
    # Generate comparative analysis
    from analysis.democracy.visualization import generate_mechanism_comparison_report
    comparison_report = generate_mechanism_comparison_report(mechanism_results)
    
    return {
        "mechanism_results": mechanism_results,
        "comparison_report": comparison_report
    }


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return ExperimentConfig(**config_dict)


def save_results(results: Dict[str, Any], output_dir: str, experiment_name: str = None):
    """Save experiment results to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Save results as JSON
    results_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")


def main():
    """Command-line entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run democratic mechanism experiments")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to experiment configuration file")
    parser.add_argument("--mechanism", type=str, choices=["PDD", "PRD", "PLD"], 
                        help="Democratic mechanism to simulate")
    parser.add_argument("--comparative", action="store_true", 
                        help="Run comparative analysis across all mechanisms")
    parser.add_argument("--output-dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--experiment-name", type=str, 
                        help="Name for the experiment (used in output filenames)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        # Default configuration
        config = ExperimentConfig(
            mechanism_type=args.mechanism or "PLD",
            num_agents=100,
            adversarial_proportion=0.3,
            adversarial_introduction="gradual",
            num_rounds=30,
            initial_resources=1000,
            resource_min_threshold=500,
            crops=["Wheat", "Corn", "Rice", "Potatoes", "Soybeans"],
            yield_volatility="variable",
            random_seed=42,
            num_trials=10,
            jit_compile=True,
            track_delegation_metrics=True,
            track_resource_metrics=True,
            track_information_metrics=True
        )
    
    # Run experiment
    if args.comparative:
        results = run_comparative_analysis(config, ["PDD", "PRD", "PLD"])
    else:
        results = run_democratic_experiment(config)
    
    # Save results
    save_results(results, args.output_dir, args.experiment_name)


if __name__ == "__main__":
    main()