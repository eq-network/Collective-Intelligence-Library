# mycorrhiza/__init__.py - MODIFIED
"""
Mycorrhiza: A process-centric multi-agent simulation platform
"""

# Expose your existing core architecture (unchanged)
from .core.graph import GraphState
from .core.category import Transform, sequential, compose
from .experiments.experiment_config import ExperimentDefinition

# Add convenience functions (new)
from lib.quick_run import quick_experiment
from lib.registry import SimulationRegistry

# Simple public interface
def make_experiment(name: str, **kwargs) -> ExperimentDefinition:
    """Create an experiment configuration"""
    return SimulationRegistry.make(name, **kwargs)

def run_experiment(name: str, **kwargs):
    """Run a quick experiment and return results"""
    return quick_experiment(name, **kwargs)

def list_experiments() -> List[str]:
    """List all available experiment types"""
    return SimulationRegistry.list_available()

# Your core architecture remains fully accessible
__all__ = [
    # Core (unchanged)
    'GraphState', 'Transform', 'sequential', 'compose', 'ExperimentDefinition',
    # Convenience (new)
    'make_experiment', 'run_experiment', 'list_experiments'
]