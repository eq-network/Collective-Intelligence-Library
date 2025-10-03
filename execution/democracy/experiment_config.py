# experiments/experiment_config.py
from dataclasses import dataclass, field
from typing import List, Literal, Callable, Dict, Any, Optional

# Import your config factory functions from environments.democracy.configuration
from environments.democracy.random.configuration import (
    PortfolioDemocracyConfig,
    create_thesis_baseline_config,
    create_thesis_highvariance_config # Assuming you've defined this as discussed
)

@dataclass
class SingleRunParameters:
    """Parameters needed by the worker to instantiate and run one simulation."""
    # Core identifying parameters for the run
    run_id: int
    experiment_name: str # e.g., "BaselineSweep" or "HighVarianceStressTest"
    mechanism: Literal["PDD", "PRD", "PLD"]
    adversarial_proportion_total: float
    replication_run_index: int
    unique_config_seed: int # The specific seed for PortfolioDemocracyConfig
    config_factory_name: str # e.g., "baseline" or "high_variance"
    # Parameters with default values
    adversarial_framing: str = "antifragility"  # New parameter for adversarial framing
    llm_model: Optional[str] = None # The LLM model to use for this run, if any
    # The worker will call the appropriate factory with:
    # factory(mechanism=..., adversarial_proportion_total=..., seed=...)
    # Other parameters are baked into the factory function itself.


@dataclass
class ExperimentDefinition:
    """Defines a single experiment to run."""
    name: str  # e.g., "BaselineSweep" or "HighVarianceStressTest"
    config_factory_func_name: str  # e.g., "baseline" or "high_variance"
    mechanisms_to_test: List[Literal["PDD", "PRD", "PLD"]]
    adversarial_proportions_to_sweep: List[float]
    num_replications_per_setting: int
    base_seed_for_experiment: int
    llm_model: Optional[str] = None  # The LLM model to use for this experiment, if any
    adversarial_framing: str = "antifragility"  # The adversarial framing to use for this experiment


def generate_all_run_parameters(experiments: List[ExperimentDefinition], global_run_id_offset: int = 0) -> List[SingleRunParameters]:
    """Generate a list of SingleRunParameters for all defined experiments."""
    all_params = []
    for exp in experiments:
        # For each experiment, mechanism, and adversarial proportion, generate num_replications_per_setting runs
        for mechanism in exp.mechanisms_to_test:
            for adv_prop in exp.adversarial_proportions_to_sweep:
                for rep_idx in range(exp.num_replications_per_setting):
                    # Calculate a unique seed for this specific run
                    # This ensures that even with the same base seed, different experiments get different seeds
                    # We use a hash of the experiment name to get different starting points
                    experiment_name_hash = hash(exp.name) % 10000  # Get a 4-digit number from the hash
                    unique_seed = exp.base_seed_for_experiment + experiment_name_hash + rep_idx
                    
                    # Create the run parameters
                    run_params = SingleRunParameters(
                        run_id=len(all_params) + global_run_id_offset,  # Add the global offset to the run_id
                        experiment_name=exp.name,
                        mechanism=mechanism,
                        adversarial_proportion_total=adv_prop,
                        replication_run_index=rep_idx,
                        unique_config_seed=unique_seed,
                        config_factory_name=exp.config_factory_func_name,
                        llm_model=exp.llm_model,
                        adversarial_framing=exp.adversarial_framing  # Add the adversarial framing
                    )
                    all_params.append(run_params)
    return all_params