# services/adapters.py
from typing import Dict, Any, Callable

class AnalysisAdapter:
    """Abstract base for analysis service adapters."""
    
    def analyze(self, agent_attrs: Dict[str, Any], global_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and return updated preferences."""
        raise NotImplementedError()
        
    def get_analysis_function(self) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
        """Return a function suitable for transformation injection."""
        return self.analyze

class RuleBasedAnalysisAdapter(AnalysisAdapter):
    """Pure rule-based implementation."""
    
    def analyze(self, agent_attrs: Dict[str, Any], global_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule-based portfolio analysis."""
        # Basic implementation - no external dependencies
        preferences = {}
        tokens_spent = 10  # Base cost
        
        # Extract available tokens
        available_tokens = agent_attrs.get("token_budget", 0) - agent_attrs.get("tokens_spent", 0)
        if available_tokens < tokens_spent:
            return {"preferences": preferences, "tokens_spent": 0}
            
        # Apply rules based on agent attributes and global info
        portfolios = global_info.get("portfolios", {})
        is_adversarial = agent_attrs.get("is_adversarial", False)
        
        for name, portfolio in portfolios.items():
            if is_adversarial:
                # Adversarial logic - invert preferences
                if "expected_return" in portfolio:
                    # Lower returns get higher scores
                    score = 10 - min(10, max(0, portfolio["expected_return"] * 5))
                else:
                    score = 5
            else:
                # Regular agent logic
                if "expected_return" in portfolio:
                    # Higher returns get higher scores
                    score = min(10, max(0, portfolio["expected_return"] * 5))
                else:
                    score = 5
                    
            preferences[name] = score
        
        return {"preferences": preferences, "tokens_spent": tokens_spent}