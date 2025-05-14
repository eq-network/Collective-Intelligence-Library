# services/llm.py
class LLMService:
    """Interface for interacting with large language models."""
    
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        raise NotImplementedError("Subclasses must implement")

class OpenAIService(LLMService):
    """OpenAI implementation of LLM service."""
    
    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        # Implementation details
        import openai
        openai.api_key = self.api_key
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    

    # services/llm.py
from typing import Dict, Any, Optional

class LLMService:
    """Interface for LLM services."""
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        raise NotImplementedError()

# services/llm_adapter.py
import json
from typing import Dict, Any, Optional
from services.llm import LLMService
from services.adapters import AnalysisAdapter

class LLMAnalysisAdapter(AnalysisAdapter):
    """LLM-based adapter for portfolio analysis."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize with an LLM service."""
        self.llm_service = llm_service
    
    def analyze(self, agent_attrs: Dict[str, Any], global_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze portfolios."""
        # Extract required information
        available_tokens = agent_attrs.get("token_budget", 0) - agent_attrs.get("tokens_spent", 0)
        if available_tokens < 10:  # Minimum cost
            return {"preferences": {}, "tokens_spent": 0}
            
        # Create prompt
        prompt = self._create_prompt(agent_attrs, global_info)
        
        try:
            # Get LLM response
            response = self.llm_service.generate(prompt)
            
            # Parse response
            scores = json.loads(response)
            
            # Calculate token cost
            token_cost = 10 + 5 * len(global_info.get("portfolios", {}))
            token_cost = min(token_cost, available_tokens)
            
            return {
                "preferences": scores,
                "tokens_spent": token_cost
            }
            
        except Exception as e:
            # Fallback to minimal response on failure
            return {
                "preferences": {name: 5.0 for name in global_info.get("portfolios", {}).keys()},
                "tokens_spent": 10
            }
    
    def _create_prompt(self, agent_attrs: Dict[str, Any], global_info: Dict[str, Any]) -> str:
        """Create prompt for LLM."""
        # Simplified prompt creation
        prompt = "Analyze investment portfolios and rate each from 0-10.\n\n"
        
        # Add portfolio information
        prompt += "Portfolios:\n"
        for name, data in global_info.get("portfolios", {}).items():
            prompt += f"- {name}: {data.get('description', '')}\n"
            prompt += f"  Expected return: {data.get('expected_return', 'Unknown')}\n"
        
        # Add prediction market information
        prediction_market = global_info.get("prediction_market")
        if prediction_market is not None:
            prompt += f"\nPrediction market signals: {list(prediction_market)}\n"
        
        # Add adversarial instruction if applicable
        if agent_attrs.get("is_adversarial", False):
            prompt += "\nYou want to maximize collective loss. Rate portfolios to achieve this."
        else:
            prompt += "\nYou want to maximize collective gain. Rate portfolios to achieve this."
        
        prompt += "\nProvide ratings as a JSON object with portfolio names as keys and scores as values."
        
        return prompt