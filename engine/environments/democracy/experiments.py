# environments/democracy/experiments.py
"""
Defines the specific experimental suites for the Portfolio Democracy simulation.
This file acts as a "menu" of experiments that the main runner can execute.
"""
from typing import List, Dict, Any

def define_democracy_experiments() -> List[Dict[str, Any]]:
    """
    Returns a list of experiment definitions for the democracy simulation.
    Each dictionary is a complete specification for a suite of runs.
    """
    experiments = []

    # --- Experiment 1: Stable Baseline with LLM Agents ---
    # This directly maps to the experiment from your original file.
    stable_llm_exp = {
        "name": "StableDemocracy_LLM_Agents_Sweep",
        "simulation_class_path": "environments.democracy.stable.simulation.StableDemocracySimulation",
        "num_replications_per_setting": 10,
        "base_seed": 20240905,
        
        # Parameters to pass to the `create_stable_democracy_config` factory
        "config_factory_path": "environments.democracy.stable.configuration.create_stable_democracy_config",
        "config_sweeps": {
            "mechanism": ["PDD", "PLD", "PRD"],
            "adversarial_proportion_total": [0.4, 0.5, 0.6, 0.7, 0.8],
        },
        "static_config_params": {
            # Specify the agent classes to use for this experiment
            "aligned_agent_path": "environments.democracy.agents.llm_agents.AlignedHeuristicAgent",
            "adversarial_agent_path": "environments.democracy.agents.llm_agents.RedTeamAgent",
            "llm_model": "openai/gpt-4o-mini",
            "adversarial_framing": "competitive"
        }
    }
    experiments.append(stable_llm_exp)

    # --- Experiment 2: A Hardcoded Baseline for Comparison ---
    hardcoded_baseline_exp = {
        "name": "StableDemocracy_Hardcoded_Baseline",
        "simulation_class_path": "environments.democracy.stable.simulation.StableDemocracySimulation",
        "num_replications_per_setting": 20, # Can run more reps since it's fast
        "base_seed": 20240910,
        "config_factory_path": "environments.democracy.stable.configuration.create_stable_democracy_config",
        "config_sweeps": {
            "mechanism": ["PDD", "PLD", "PRD"],
            "adversarial_proportion_total": [0.0, 0.2, 0.4, 0.6, 0.8],
        },
        "static_config_params": {
            "aligned_agent_path": "environments.democracy.agents.hardcoded.HardcodedAlignedAgent",
            "adversarial_agent_path": "environments.democracy.agents.hardcoded.HardcodedAdversarialAgent",
            "llm_model": None, # No LLM needed for this run
        }
    }
    experiments.append(hardcoded_baseline_exp)
    
    return experiments