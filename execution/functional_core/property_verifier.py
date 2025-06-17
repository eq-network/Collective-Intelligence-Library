# execution/functional_core/property_verifier.py

from typing import Set, Dict, Any, Optional
import jax.numpy as jnp

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from algebra.graph import GraphState
from algebra.category import Transform
from algebra.property import Property

class PropertyVerifier:
    """
    Verification of mathematical properties on graph transformations.
    
    The PropertyVerifier ensures that transformations preserve their
    declared mathematical properties when applied to graph states.
    """
    
    def verify(
        self, 
        transform: Transform, 
        initial_state: GraphState, 
        final_state: GraphState
    ) -> Dict[str, bool]:
        """
        Verify that a transformation preserves its declared properties.
        
        Args:
            transform: The transformation that was applied
            initial_state: The state before transformation
            final_state: The state after transformation
            
        Returns:
            Dictionary mapping property names to verification results
        """
        # Skip verification if no properties are defined
        if not hasattr(transform, 'preserves'):
            return {}
        
        results = {}
        properties = transform.preserves
        
        # Special case for the identity transform
        if properties == "ALL_PROPERTIES":
            # The identity should preserve everything, so just check equality
            results["state_equality"] = (initial_state == final_state)
            return results
        
        # Verify each property
        for prop in properties:
            # Check if property is maintained in final state
            results[prop.name] = prop.check(final_state)
            
            # Additional checks for certain property types
            if hasattr(prop, 'compare_states'):
                results[f"{prop.name}_comparison"] = prop.compare_states(
                    initial_state, final_state
                )
        
        return results
    
    def verify_composition(
        self, 
        transforms: list[Transform], 
        state: GraphState
    ) -> Dict[str, Any]:
        """
        Verify properties of a composed transformation chain.
        
        Args:
            transforms: List of transformations in composition
            state: Initial state before applying composition
            
        Returns:
            Verification results and composition analysis
        """
        # Collect all properties from the transformations
        all_properties = set()
        for transform in transforms:
            if hasattr(transform, 'preserves'):
                if transform.preserves == "ALL_PROPERTIES":
                    # Special handling for identity
                    continue
                all_properties.update(transform.preserves)
        
        # Determine properties preserved by composition
        preserved_properties = set(all_properties)
        for transform in transforms:
            if hasattr(transform, 'preserves'):
                if transform.preserves == "ALL_PROPERTIES":
                    # Identity preserves everything
                    continue
                preserved_properties.intersection_update(transform.preserves)
        
        return {
            "all_properties": [p.name for p in all_properties],
            "preserved_properties": [p.name for p in preserved_properties]
        }