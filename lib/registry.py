# lib/registry.py - NEW FILE
from typing import Dict, Callable, Any, List

class SimulationRegistry:
    """Simple registry for discovering pre-configured simulations"""
    _simulations: Dict[str, Callable[..., ExperimentDefinition]] = {}
    
    @classmethod
    def register(cls, name: str, factory: Callable):
        cls._simulations[name] = factory
    
    @classmethod 
    def make(cls, name: str, **kwargs) -> ExperimentDefinition:
        return cls._simulations[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._simulations.keys())

# Usage - this just creates your existing ExperimentDefinition objects
registry = SimulationRegistry()