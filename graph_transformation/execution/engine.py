# execution/engine.py

from typing import Dict, List, Any, Optional, Callable, Tuple
import jax
import jax.numpy as jnp
import time

from core.graph import GraphState
from core.category import Transform
from core.property import Property

# Type alias for execution configurations
ExecutionConfig = Dict[str, Any]

class ExecutionEngine:
    """
    Core execution engine for graph transformations.
    
    The ExecutionEngine coordinates the execution of transformations,
    managing strategies, property verification, and instrumentation.
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize with execution configuration."""
        self.config = config or {}
        self._initialize_components()
        self._metrics = {}
        self._history = []
    
    def _initialize_components(self):
        """Initialize execution components based on configuration."""
        # Import here to avoid circular dependencies
        from execution.functional_core.evaluator import TransformEvaluator
        from execution.functional_core.property_verifier import PropertyVerifier
        from execution.instrumentation.history import HistoryTracker
        from execution.instrumentation.metrics import MetricsCollector
        
        # Core components
        self._evaluator = TransformEvaluator()
        self._verifier = PropertyVerifier()
        
        # Optional instrumentation
        if self.config.get("track_history", False):
            self._history_tracker = HistoryTracker()
        else:
            self._history_tracker = None
            
        if self.config.get("collect_metrics", False):
            self._metrics_collector = MetricsCollector()
        else:
            self._metrics_collector = None
        
        # Strategy components
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """Initialize execution strategy based on configuration."""
        strategy = self.config.get("strategy", "sequential")
        
        if strategy == "parallel" and self.config.get("use_parallel", False):
            from execution.effect_handlers.parallel import ParallelStrategy
            self._strategy = ParallelStrategy(
                num_workers=self.config.get("num_workers", 1)
            )
        elif strategy == "hardware" and self.config.get("use_hardware_acceleration", False):
            from execution.effect_handlers.hardware import HardwareAccelerator
            self._strategy = HardwareAccelerator(
                device=self.config.get("hardware_device", "cpu")
            )
        else:
            # Default to sequential strategy
            self._strategy = None
    
    def apply(
        self, 
        transform: Transform, 
        state: GraphState,
        verify_properties: bool = True
    ) -> GraphState:
        """
        Apply a transformation to a graph state.
        
        Args:
            transform: Transformation to apply
            state: Current graph state
            verify_properties: Whether to verify properties
            
        Returns:
            Transformed graph state
        """
        # Start metrics collection
        if self._metrics_collector:
            self._metrics_collector.start_collection()
            start_time = time.time()
        
        # Apply transformation using appropriate strategy
        if self._strategy and self._evaluator.is_parallelizable(transform):
            new_state = self._strategy.execute(transform, state, self._evaluator)
        else:
            new_state = self._evaluator.evaluate(transform, state)
        
        # Verify properties if requested
        if verify_properties and hasattr(transform, 'preserves'):
            verification_results = self._verifier.verify(transform, state, new_state)
            # Store verification results in metrics
            if self._metrics_collector:
                self._metrics["verification"] = verification_results
        
        # Record history
        if self._history_tracker:
            self._history_tracker.record(transform, state, new_state)
            self._history = self._history_tracker.get_history()
        
        # Finish metrics collection
        if self._metrics_collector:
            end_time = time.time()
            self._metrics["execution_time"] = end_time - start_time
            self._metrics_collector.record_metrics(self._metrics)
        
        return new_state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected execution metrics."""
        return self._metrics
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._history