# core/simulation.py
"""
Defines the Simulation class, a high-level orchestrator for a single simulation run.
"""
from typing import List
from core.environment import Environment
import pandas as pd

class Simulation:
    """
    Encapsulates a single, complete simulation run.

    It takes a fully configured environment (which includes its agents) and
    manages the execution loop, data collection, and result formatting.
    """
    def __init__(self, environment: Environment):
        if not isinstance(environment, Environment):
            raise TypeError("A Simulation must be initialized with an Environment instance.")
        self.environment = environment
        self.history: List[pd.DataFrame] = []

    def run(self) -> None:
        """
        Executes the simulation from start to finish.
        """
        self.environment.reset()
        self.history.append(self.environment.get_state_summary())

        while True:
            state_summary, terminated = self.environment.step()
            self.history.append(state_summary)
            if terminated:
                break
    
    def get_results_as_dataframe(self) -> pd.DataFrame:
        """
        Converts the simulation history into a single, clean pandas DataFrame.
        """
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)