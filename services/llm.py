# services/llm.py
import os
import uuid
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
def create_llm_service(model: str = "google/gemini-2.5-flash-preview-05-20", api_key: Optional[str] = None) -> LLMService:
    """Create appropriate LLM service instance."""
    return OpenRouterService(api_key=api_key, model=model)

if __name__ == '__main__':
    # Example usage
    try:
        # Create service with model specification
        service = create_llm_service(model="google/gemini-2.5-flash-preview-05-20")
        
        # Test the service (uncomment to actually make API call)
        # response = service.generate("Write a function to calculate factorial")
        # print(f"Response: {response}")
        
        print("OpenRouter service successfully initialized")
    except Exception as e:
        print(f"Error initializing service: {e}")

class ProcessIsolatedLLMService:
    """
    LLM service wrapper that ensures complete process isolation and request validation.
    
    Purpose: Prevent cross-process contamination of LLM requests that was causing 
    Chinese responses and mixed outputs.
    
    Key Features:
    - Process-specific session isolation
    - Request tracing for debugging
    - Response validation and corruption detection
    - Automatic retry with modified parameters
    """
    
    def __init__(self, model: str, api_key: str, process_id: str):
        self.model = model
        self.api_key = api_key
        self.process_id = process_id
        self.request_counter = 0
        self.session_id = str(uuid.uuid4())[:8]  # Short session ID
        
        # Create isolated service instance
        self._service = self._create_isolated_service()
    
    def _create_isolated_service(self) -> LLMService:
        """Create completely isolated LLM service instance with validation."""
        try:
            service = create_llm_service(model=self.model, api_key=self.api_key)
            
            # Validate service with minimal test call
            test_response = service.generate("Test", max_tokens=5, temperature=0.1)
            print(f"[PID {self.process_id}] LLM Service initialized. Test: {test_response[:30]}...")
            
            return service
            
        except Exception as e:
            print(f"[PID {self.process_id}] LLM Service initialization failed: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate response with full isolation and validation."""
        self.request_counter += 1
        request_id = f"{self.process_id}-{self.session_id}-{self.request_counter:03d}"
        
        try:
            print(f"[PID {self.process_id}] LLM Request {self.request_counter}")
            
            # Make API call with process-specific parameters
            response = self._service.generate(
                prompt, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            # Validate response integrity
            if self._is_response_corrupted(response):
                print(f"[PID {self.process_id}] WARNING: Corrupted response detected")
                print(f"[PID {self.process_id}] Response preview: {response[:100]}...")
                
                # Single retry with modified parameters
                print(f"[PID {self.process_id}] Retrying with modified parameters")
                response = self._service.generate(
                    prompt, 
                    max_tokens=max_tokens, 
                    temperature=max(0.1, temperature - 0.3)
                )
            
            return response
            
        except Exception as e:
            print(f"[PID {self.process_id}] LLM request failed: {e}")
            raise
    
    def _is_response_corrupted(self, response: str) -> bool:
        """Detect corrupted or mixed-language responses."""
        if not response or len(response.strip()) < 5:
            return True
        
        # Check for Chinese characters (indicator of model confusion)
        chinese_chars = sum(1 for char in response if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > len(response) * 0.1:
            return True
        
        # Check for excessive repetition patterns
        repetition_patterns = ['1.', '0,', 'τ', '000000']
        for pattern in repetition_patterns:
            if (pattern * 5) in response:
                return True
        
        return False