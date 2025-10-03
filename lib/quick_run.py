# lib/quick_run.py - NEW FILE  
from experiments.runner import EnhancedParallelExperimentRunner
from experiments.democracy.experiment_config import generate_all_run_parameters
from typing import Dict, Any

def quick_experiment(simulation_name: str, **kwargs) -> Dict[str, Any]:
    """Simple interface for running a single experiment"""
    
    # Use existing registry to get experiment definition
    from . import registry
    exp_def = registry.make(simulation_name, **kwargs)
    
    # Use your existing machinery unchanged
    run_params = generate_all_run_parameters([exp_def])
    runner = EnhancedParallelExperimentRunner(
        output_dir=kwargs.get("output_dir", "./results"),
        suite_timestamp=kwargs.get("timestamp", "quick_run"),
        max_workers=kwargs.get("max_workers", 4)
    )
    
    # Return your existing results aggregator
    return runner.run_experiment_grid(run_params)