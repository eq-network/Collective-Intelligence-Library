# experiments/experiment_config.py
from dataclasses import dataclass, field
from typing import List, Literal, Callable, Dict, Any, Optional

# Import your config factory functions from environments.democracy.configuration
from environments.democracy.configuration import (
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

    # Name of the factory function to call to get the PortfolioDemocracyConfig
    # This allows us to choose between create_thesis_baseline_config, create_thesis_highvariance_config, etc.
    config_factory_name: str # e.g., "baseline" or "high_variance"
    llm_model: Optional[str] = None # The LLM model to use for this run, if any
    # The worker will call the appropriate factory with:
    # factory(mechanism=..., adversarial_proportion_total=..., seed=...)
    # Other parameters are baked into the factory function itself.


@dataclass
class ExperimentDefinition:
    """Defines a single type of experiment to sweep over."""
    name: str # e.g., "Baseline_Experiment" or "HighVariance_Experiment"
    
    # The function from configuration.py that will create the PortfolioDemocracyConfig
    # e.g., create_thesis_baseline_config or create_thesis_highvariance_config
    config_factory_func_name: str # Store the name of the function

    # Parameters to sweep for this specific experiment definition
    mechanisms_to_test: List[Literal["PDD", "PRD", "PLD"]]
    adversarial_proportions_to_sweep: List[float]
    
    num_replications_per_setting: int
    base_seed_for_experiment: int # Master seed for this specific experiment definition
    llm_model: Optional[str] = None # LLM model for this entire experiment definition


def generate_all_run_parameters(
    experiment_definitions: List[ExperimentDefinition],
    global_run_id_offset: int = 0
) -> List[SingleRunParameters]:
    """
    Generates a flat list of all SingleRunParameters for all defined experiments.
    """
    all_single_run_params_list = []
    current_overall_run_id_counter = global_run_id_offset

    for exp_def in experiment_definitions:
        num_settings_in_exp = len(exp_def.mechanisms_to_test) * len(exp_def.adversarial_proportions_to_sweep)
        
        for i_mech, mech in enumerate(exp_def.mechanisms_to_test):
            for i_adv, adv_prop_total in enumerate(exp_def.adversarial_proportions_to_sweep):
                for i_rep in range(exp_def.num_replications_per_setting):
                    # Create a unique seed for this specific run configuration
                    # Based on experiment's base seed and the specific setting
                    setting_offset = (i_mech * len(exp_def.adversarial_proportions_to_sweep) + i_adv) * exp_def.num_replications_per_setting
                    unique_run_config_seed = exp_def.base_seed_for_experiment + setting_offset + i_rep

                    params = SingleRunParameters(
                        run_id=current_overall_run_id_counter,
                        experiment_name=exp_def.name, # To identify which experiment this run belongs to
                        mechanism=mech,
                        adversarial_proportion_total=adv_prop_total,
                        replication_run_index=i_rep,
                        unique_config_seed=unique_run_config_seed,
                        config_factory_name=exp_def.config_factory_func_name, # Worker will use this
                        llm_model=exp_def.llm_model
                    )
                    all_single_run_params_list.append(params)
                    current_overall_run_id_counter += 1
                    
    return all_single_run_params_list