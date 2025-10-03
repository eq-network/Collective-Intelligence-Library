# experiments/worker.py
"""
The generic simulation worker. It receives a parameter set, dynamically
imports the required classes, runs the simulation, and returns the results.
"""
import os
import time
import pandas as pd
import importlib
from typing import Dict, Any, Tuple, List

from core.simulation import Simulation
from services.llm import ProcessIsolatedLLMService

def _import_class(class_path: str):
    """Dynamically imports a class from a string path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_simulation_task(params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], list]:
    """
    A generic worker function that can run any simulation defined by SingleRunParams.
    """
    run_id = params['run_id']
    print(f"[Worker {os.getpid()}] Starting Run {run_id}: {params['experiment_name']}")
    start_time = time.time()
    
    try:
        # 1. Dynamically import Environment and Agent classes
        EnvironmentClass = _import_class(params['environment_class_path'])
        
        agent_classes = {}
        for pop_config in params['agent_population_configs']:
            class_path = pop_config['agent_class_path']
            if class_path not in agent_classes:
                agent_classes[class_path] = _import_class(class_path)

        # 2. Instantiate LLM Service if needed
        llm_service = None
        if params['environment_params'].get('llm_model'):
            llm_service = ProcessIsolatedLLMService(model=params['environment_params']['llm_model'])

        # 3. Instantiate Agent Population
        agents = []
        for pop_config in params['agent_population_configs']:
            AgentClass = agent_classes[pop_config['agent_class_path']]
            for _ in range(pop_config['count']):
                # Agent constructors might need the llm_service
                # A more advanced version could pass agent-specific params here.
                agent_id = len(agents)
                agents.append(AgentClass(agent_id=agent_id, llm_service=llm_service))

        # 4. Instantiate and run the Simulation
        # The environment class constructor takes the config dict and the agent list
        environment = EnvironmentClass(config=params['environment_params'], agents=agents)
        simulation = Simulation(environment=environment)
        simulation.run()

        # 5. Collect and return results
        results_df = simulation.get_results_as_dataframe()
        duration = time.time() - start_time
        final_state_summary = simulation.history[-1] if simulation.history else {}

        metadata = {
            'run_id': run_id,
            'status': 'success',
            'duration_s': duration,
            **params,  # Include all run parameters in metadata
            **final_state_summary, # Add final state metrics
        }
        
        return results_df, metadata, [] # Anomaly logging can be added to the environment state

    except Exception as e:
        # Generic error handling
        duration = time.time() - start_time
        print(f"[Worker {os.getpid()}] FAILED Run {run_id}: {e}")
        traceback.print_exc()
        
        error_df = pd.DataFrame()
        metadata = {
            'run_id': run_id,
            'status': 'error',
            'error_message': str(e),
            'duration_s': duration,
            **params
        }
        return error_df, metadata, []