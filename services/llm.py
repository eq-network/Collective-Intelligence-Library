# services/llm.py
import os
import requests
from typing import Dict, Any, Optional

class LLMService:
    """Base interface for LLM services."""
    
    def __init__(self, api_key: str = None, model: str = "default"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        raise NotImplementedError("Subclasses must implement generate method.")

class OpenRouterService(LLMService):
    """OpenRouter implementation accessing multiple LLM providers."""
    
    def __init__(self, api_key: str = None, model: str = "deepseek/deepseek-coder"):
        # Get API key from param or environment
        super().__init__(api_key, model)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        self.base_url = "https://openrouter.ai/api/v1"
        
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate text using OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000"  # Required by OpenRouter
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", 
                headers=headers, 
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            # Match exception pattern from original implementation for adapter compatibility
            print(f"Error calling OpenRouter API: {e}")
            raise ConnectionError(f"OpenRouter API call failed: {e}")
        except Exception as e:
            # Handle other exceptions (parsing errors, etc.)
            print(f"Unexpected error with OpenRouter: {e}")
            raise

# Factory function for consistent service creation
def create_llm_service(model: str = "deepseek/deepseek-coder", api_key: Optional[str] = None) -> LLMService:
    """Create appropriate LLM service instance."""
    return OpenRouterService(api_key=api_key, model=model)

if __name__ == '__main__':
    # Example usage
    try:
        # Create service with model specification
        service = create_llm_service(model="deepseek/deepseek-coder")
        
        # Test the service (uncomment to actually make API call)
        # response = service.generate("Write a function to calculate factorial")
        # print(f"Response: {response}")
        
        print("OpenRouter service successfully initialized")
    except Exception as e:
        print(f"Error initializing service: {e}")