# execution/engine.py

from typing import Dict, List, Any, Optional, Callable, Tuple, TypeVar, Protocol
import jax
import jax.numpy as jnp
from functools import partial
import time

from core.graph import GraphState
from core.category import Transform
from core.property import Property

# Type for execution configurations
ExecutionConfig = Dict[str, Any]

class ExecutionEngine:
    """
    Central coordinator for graph transformation execution.
    
    The ExecutionEngine separates transformation semantics from execution concerns,
    allowing different execution strategies while preserving mathematical properties.
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        """
        Initialize execution engine with optional configuration.
        
        Args:
            config: Configuration parameters for execution strategies
        """
        self.config = config or {}
        self._transform_evaluator = None
        self._property_verifier = None
        self._parallel_strategy = None
        self._hardware_accelerator = None
        self._memory_manager = None
        self._history_tracker = None
        self._metrics_collector = None
        
        # Initialize components based on configuration
        self._init_components()
    
    def _init_components(self):
        """Initialize all engine components based on configuration."""
        from execution.functional_core.evaluator import TransformEvaluator
        from execution.functional_core.property_verifier import PropertyVerifier
        
        # Initialize functional core
        self._transform_evaluator = TransformEvaluator()
        self._property_verifier = PropertyVerifier()
        
        # Initialize effect handlers based on configuration
        if self.config.get("use_parallel", False):
            from execution.effect_handlers.parallel import ParallelStrategy
            self._parallel_strategy = ParallelStrategy(
                num_workers=self.config.get("num_workers", 1)
            )
            
        if self.config.get("use_hardware_acceleration", False):
            from execution.effect_handlers.hardware import HardwareAccelerator
            self._hardware_accelerator = HardwareAccelerator(
                device=self.config.get("hardware_device", "cpu")
            )
            
        if self.config.get("use_memory_management", False):
            from execution.effect_handlers.memory import MemoryManager
            self._memory_manager = MemoryManager(
                max_memory=self.config.get("max_memory", None)
            )
        
        # Initialize instrumentation based on configuration
        if self.config.get("track_history", False):
            from execution.instrumentation.history import HistoryTracker
            self._history_tracker = HistoryTracker(
                max_history=self.config.get("max_history", 100)
            )
            
        if self.config.get("collect_metrics", False):
            from execution.instrumentation.metrics import MetricsCollector
            self._metrics_collector = MetricsCollector(
                metrics=self.config.get("metrics", ["time", "memory"])
            )
    
    def apply(
        self, 
        transform: Transform, 
        state: GraphState,
        verify_properties: bool = True
    ) -> GraphState:
        """
        Apply a transformation to a graph state with instrumentation.
        
        Args:
            transform: Transformation to apply
            state: Current graph state
            verify_properties: Whether to verify properties after transformation
            
        Returns:
            New graph state after transformation
        """
        # Start metrics collection if enabled
        if self._metrics_collector:
            self._metrics_collector.start_collection()
        
        # Apply memory management if enabled
        if self._memory_manager:
            self._memory_manager.prepare_for_execution(state)
        
        # Choose execution strategy based on configuration
        if self._parallel_strategy and self._transform_evaluator.is_parallelizable(transform):
            # Apply transform using parallel strategy
            new_state = self._parallel_strategy.execute(
                transform, state, self._transform_evaluator
            )
        elif self._hardware_accelerator:
            # Apply transform with hardware acceleration
            new_state = self._hardware_accelerator.execute(
                transform, state, self._transform_evaluator
            )
        else:
            # Standard sequential execution
            new_state = self._transform_evaluator.evaluate(transform, state)
        
        # Verify properties if requested
        if verify_properties and self._property_verifier and hasattr(transform, 'preserves'):
            self._property_verifier.verify(transform, state, new_state)
        
        # Record history if enabled
        if self._history_tracker:
            self._history_tracker.record(transform, state, new_state)
        
        # Finish metrics collection if enabled
        if self._metrics_collector:
            self._metrics_collector.finish_collection(transform, state, new_state)
        
        return new_state
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get execution history if history tracking is enabled."""
        return self._history_tracker.get_history() if self._history_tracker else []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics if metrics collection is enabled."""
        return self._metrics_collector.get_metrics() if self._metrics_collector else {}