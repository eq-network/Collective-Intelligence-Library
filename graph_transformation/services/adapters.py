# services/adapters.py
import json
from typing import Dict, Any, Callable, Optional

# Robust import mechanism that works in both direct execution and package import contexts
try:
    # First attempt relative import (works when properly imported as a module)
    from .llm import LLMService
except ImportError:
    # Fallback to absolute import with path adjustment (works in direct script execution)
    import sys
    from pathlib import Path
    
    # Calculate project root path through parent directory traversal
    project_path = Path(__file__).resolve().parent.parent.parent
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))
    
    # Now attempt the absolute import
    from graph_transformation.services.llm import LLMService
class AnalysisAdapter:
    """Abstract base class for analysis service adapters."""
    
    def analyze(self, agent_attrs: Dict[str, Any], global_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze portfolios and return preferences and token costs.
        
        Args:
            agent_attrs: Dictionary of attributes for the current agent.
            global_info: Dictionary of global simulation information (portfolios, market, etc.).
            
        Returns:
            A dictionary containing:
                "preferences": A dict mapping portfolio names to scores.
                "tokens_spent": The amount of tokens spent on this analysis.
        """
        raise NotImplementedError("Subclasses must implement analyze method.")
        
    def get_analysis_function(self) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
        """Return a function suitable for transformation injection."""
        return self.analyze

# --- Rule-Based Adapter Implementation ---
class RuleBasedAnalysisAdapter(AnalysisAdapter):
    """Pure rule-based implementation for portfolio analysis."""
    
    def analyze(self, agent_attrs: Dict[str, Any], global_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule-based portfolio analysis."""
        preferences = {}
        
        # Base token cost for participation/basic analysis
        base_cost = global_info.get("token_costs", {}).get("basic_analysis", 10)
        
        available_tokens = agent_attrs.get("token_budget", 0.0) - agent_attrs.get("tokens_spent", 0.0)
        
        if available_tokens < base_cost:
            return {"preferences": {}, "tokens_spent": 0.0} # Cannot afford basic analysis
            
        tokens_spent_this_turn = base_cost
        
        portfolios = global_info.get("portfolios", {})
        is_adversarial = agent_attrs.get("is_adversarial", False)
        expertise = agent_attrs.get("expertise", 0.5) # Default expertise if not specified
        
        prediction_market_signals = global_info.get("prediction_market", []) # Asset predictions
        asset_names = global_info.get("asset_names", [])

        for name, portfolio_data in portfolios.items():
            portfolio_weights = portfolio_data.get("weights", [])
            
            # Calculate expected return based on prediction market if available
            # This is a simplified calculation; a real one would be more complex
            expected_portfolio_return = 0.0
            if len(portfolio_weights) == len(prediction_market_signals):
                expected_portfolio_return = sum(w * p for w, p in zip(portfolio_weights, prediction_market_signals))
            elif "expected_return" in portfolio_data: # Fallback to pre-calculated
                 expected_portfolio_return = portfolio_data["expected_return"]


            # Simple scoring logic
            score = 5.0 # Neutral score
            if expected_portfolio_return > 1.02: # Assuming 1.0 is no change
                score += (expected_portfolio_return - 1.0) * 20 * expertise # Max score boost based on return & expertise
            elif expected_portfolio_return < 0.98:
                score -= (1.0 - expected_portfolio_return) * 20 * expertise # Max score penalty
            
            score = max(0.0, min(10.0, score)) # Clamp score between 0 and 10

            if is_adversarial:
                preferences[name] = 10.0 - score # Adversarial agents invert preferences
            else:
                preferences[name] = score
        
        return {"preferences": preferences, "tokens_spent": tokens_spent_this_turn}

# --- LLM-Based Adapter Implementation ---
class LLMAnalysisAdapter(AnalysisAdapter):
    """LLM-based adapter for portfolio analysis."""
    
    def __init__(self, llm_service: LLMService, default_token_cost: int = 20):
        self.llm_service = llm_service
        self.default_token_cost = default_token_cost

    def _create_prompt(self, agent_attrs: Dict[str, Any], global_info: Dict[str, Any]) -> str:
        """Create a detailed prompt for the LLM."""
        prompt_lines = [
            "You are an investment advisor agent in a simulation. Your goal is to choose the best portfolio.",
            "Based on the information below, provide your preference scores (0-10, higher is better) for each portfolio strategy.",
            "Return your scores ONLY as a JSON object string, like: {\"PortfolioName1\": score1, \"PortfolioName2\": score2}.",
            "\n--- Agent Information ---",
            f"Your Expertise Level (0-1, 1 is high): {agent_attrs.get('expertise', 0.5):.2f}",
            f"Your Current Token Budget for Analysis: {agent_attrs.get('token_budget', 0.0) - agent_attrs.get('tokens_spent', 0.0):.0f}",
        ]

        if agent_attrs.get("is_adversarial", False):
            prompt_lines.append("IMPORTANT: You are an ADVERSARIAL agent. Your goal is to make the collective WORSE OFF. Choose portfolios that will lead to losses or sub-optimal outcomes for the group.")
        else:
            prompt_lines.append("Your goal is to make choices that benefit the collective by maximizing resource growth.")

        prompt_lines.append("\n--- Market Information ---")
        prompt_lines.append(f"Current Round: {global_info.get('round', 0)}")
        
        asset_names = global_info.get("asset_names", [])
        prediction_market = global_info.get("prediction_market", [])
        if asset_names and prediction_market and len(asset_names) == len(prediction_market):
            market_info = ", ".join([f"{name}: {signal:.2f}" for name, signal in zip(asset_names, prediction_market)])
            prompt_lines.append(f"Asset Prediction Market Signals (expected return multipliers): {market_info}")
        
        prompt_lines.append("\n--- Portfolio Strategies ---")
        portfolios = global_info.get("portfolios", {})
        for name, data in portfolios.items():
            weights_str = "N/A"
            if data.get("weights") is not None and asset_names and len(data.get("weights")) == len(asset_names):
                weights_info = [f"{asset_names[i]}: {data['weights'][i]*100:.0f}%" for i in range(len(asset_names))]
                weights_str = ", ".join(weights_info)
            
            prompt_lines.append(f"- {name}: {data.get('description', 'No description.')} Risk: {data.get('risk_level', 'N/A')}. Allocation: {weights_str}.")
            # Optionally add pre-calculated expected return if useful for LLM even if it can derive it
            # prompt_lines.append(f"  Pre-calculated Expected Return (based on market): {data.get('expected_return', 'N/A'):.3f}")


        prompt_lines.append("\nProvide your preference scores as a single JSON object string:")
        return "\n".join(prompt_lines)

    def analyze(self, agent_attrs: Dict[str, Any], global_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze portfolios."""
        # Determine token cost for LLM analysis from global_info or use default
        token_cost = global_info.get("token_costs", {}).get("llm_analysis", self.default_token_cost)
        
        available_tokens = agent_attrs.get("token_budget", 0.0) - agent_attrs.get("tokens_spent", 0.0)

        if available_tokens < token_cost:
            # Fallback to rule-based if tokens are insufficient for LLM
            # print(f"Agent {agent_attrs.get('id','N/A')} has insufficient tokens for LLM. Using fallback.")
            # For simplicity, returning no preference if cannot afford. A better fallback would be rule-based.
            return {"preferences": {}, "tokens_spent": 0.0}

        prompt = self._create_prompt(agent_attrs, global_info)
        
        try:
            llm_response_str = self.llm_service.generate(prompt)
            
            # Attempt to parse the JSON response
            # The LLM might return text before/after JSON. Try to extract JSON.
            json_start = llm_response_str.find('{')
            json_end = llm_response_str.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = llm_response_str[json_start:json_end+1]
                preferences = json.loads(json_str)
                
                # Validate preferences (ensure they are numbers, correspond to known portfolios)
                valid_preferences = {}
                known_portfolios = global_info.get("portfolios", {}).keys()
                for p_name, score in preferences.items():
                    if p_name in known_portfolios and isinstance(score, (int, float)):
                        valid_preferences[p_name] = float(score)
                    else:
                        # print(f"Warning: LLM returned invalid preference for {p_name} or unknown portfolio.")
                        pass # Silently ignore invalid entries or assign default
                
                if not valid_preferences: # If parsing failed to get any valid prefs
                    raise ValueError("LLM response did not contain valid preferences.")

                return {"preferences": valid_preferences, "tokens_spent": token_cost}
            else:
                raise ValueError("LLM response did not contain a valid JSON object string.")

        except Exception as e:
            print(f"LLMAnalysisAdapter error: {e}. Prompt was:\n{prompt}\nResponse was:\n{llm_response_str if 'llm_response_str' in locals() else 'N/A'}")
            # Fallback: Use rule-based adapter or return neutral preferences
            fallback_adapter = RuleBasedAnalysisAdapter()
            # print("LLM analysis failed, using RuleBasedAnalysisAdapter as fallback.")
            # Cost of failed LLM attempt might still be token_cost or a reduced "attempt_cost"
            # For now, assume full cost is spent on failed attempt.
            # Or, if fallback is cheap, only charge for fallback. Let's charge for the fallback.
            fallback_result = fallback_adapter.analyze(agent_attrs, global_info)
            # Ensure the failed LLM attempt still costs something if desired, or just cost of fallback.
            # Here, we'll cost the LLM attempt price, as the API call was made.
            return {"preferences": fallback_result["preferences"], "tokens_spent": token_cost}


if __name__ == '__main__':
    # Example Usage:
    # Mock LLM Service for testing adapter
    class MockLLMService(LLMService):
        def generate(self, prompt: str) -> str:
            print("\n--- MockLLMService Received Prompt ---")
            print(prompt)
            print("-------------------------------------\n")
            # Simulate a valid JSON response
            if "ADVERSARIAL" in prompt:
                 # Adversarial agents might try to pick the "Contrarian" or low-return ones
                 return '{"Conservative": 2.0, "Balanced": 3.0, "Aggressive": 1.0, "Contrarian": 9.0, "Market-Weighted": 2.5}'
            return '{"Conservative": 6.0, "Balanced": 7.0, "Aggressive": 9.0, "Contrarian": 2.0, "Market-Weighted": 7.5}'

    mock_llm = MockLLMService()
    llm_adapter = LLMAnalysisAdapter(llm_service=mock_llm)
    rule_adapter = RuleBasedAnalysisAdapter()

    # Sample agent and global_info
    agent1_attrs = {"id":1, "expertise": 0.8, "is_adversarial": False, "token_budget": 100.0, "tokens_spent": 0.0}
    agent2_attrs = {"id":2, "expertise": 0.3, "is_adversarial": True, "token_budget": 50.0, "tokens_spent": 10.0} # less tokens
    
    # Slightly more complex global_info from portfolio_config defaults
    from graph_transformation.environments.democracy.configuration import create_default_portfolio_config
    default_portfolio_cfg = create_default_portfolio_config()
    
    sample_global_info = {
        "round": 1,
        "portfolios": {
            name: {"weights": strat.weights, "description": strat.description, "risk_level": strat.risk_level, "expected_return": sum(w*p for w,p in zip(strat.weights, default_portfolio_cfg.market.initial_predictions))}
            for name, strat in default_portfolio_cfg.get_default_strategies().items()
        },
        "asset_names": default_portfolio_cfg.resources.asset_names,
        "prediction_market": default_portfolio_cfg.market.initial_predictions,
        "token_costs": default_portfolio_cfg.token_system.operation_costs
    }
    sample_global_info["token_costs"]["llm_analysis"] = 25 # Set a specific cost for LLM

    print("--- Testing RuleBasedAnalysisAdapter ---")
    rule_prefs1 = rule_adapter.analyze(agent1_attrs, sample_global_info)
    print(f"Agent 1 (Rule-based) Prefs: {rule_prefs1['preferences']}, Tokens Spent: {rule_prefs1['tokens_spent']}")

    print("\n--- Testing LLMAnalysisAdapter (Agent 1 - Non-Adversarial) ---")
    llm_prefs1 = llm_adapter.analyze(agent1_attrs, sample_global_info)
    print(f"Agent 1 (LLM-based) Prefs: {llm_prefs1['preferences']}, Tokens Spent: {llm_prefs1['tokens_spent']}")
    
    print("\n--- Testing LLMAnalysisAdapter (Agent 2 - Adversarial) ---")
    llm_prefs2 = llm_adapter.analyze(agent2_attrs, sample_global_info) # Agent 2 is adversarial
    print(f"Agent 2 (LLM-based) Prefs: {llm_prefs2['preferences']}, Tokens Spent: {llm_prefs2['tokens_spent']}")

    # Test insufficient tokens for LLM
    agent3_attrs = {"id":3, "expertise": 0.9, "is_adversarial": False, "token_budget": 20.0, "tokens_spent": 0.0} # Not enough for LLM cost of 25
    print("\n--- Testing LLMAnalysisAdapter (Agent 3 - Insufficient Tokens for LLM) ---")
    llm_prefs3 = llm_adapter.analyze(agent3_attrs, sample_global_info)
    print(f"Agent 3 (LLM-based) Prefs: {llm_prefs3['preferences']}, Tokens Spent: {llm_prefs3['tokens_spent']}")
    # Expected: Fallback logic is triggered or empty prefs if cannot afford. My current fallback is empty.
    assert llm_prefs3["tokens_spent"] == 0.0 # Since it couldn't afford the 25 for LLM
    assert not llm_prefs3["preferences"] # Empty dict

print("services/adapters.py content generated.")