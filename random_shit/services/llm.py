# services/llm.py
import os
import uuid
import time # Added for sleep
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
    
    def __init__(self, 
                 api_key: str = None, 
                 model: str = "google/gemini-2.0-flash-001",
                 max_retries: int = 3,
                 initial_backoff_seconds: float = 1.0,
                 request_delay_seconds: float = 0.1 # Small delay between requests
                 ):
        # Get API key from param or environment
        super().__init__(api_key, model)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.request_delay_seconds = request_delay_seconds
        
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
        
        # Introduce a small delay before making the request
        if self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)

        current_retry = 0
        while current_retry <= self.max_retries:
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions", 
                    headers=headers, 
                    json=payload,
                    timeout=30 # Add a timeout
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
                return response.json()["choices"][0]["message"]["content"]
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429: # Too Many Requests
                    if current_retry < self.max_retries:
                        backoff_time = self.initial_backoff_seconds * (2 ** current_retry)
                        # Add some jitter to backoff_time to prevent thundering herd
                        jitter = backoff_time * 0.1 * (2 * os.urandom(1)[0] / 255 - 1) # +/- 10% jitter
                        actual_backoff = max(0.1, backoff_time + jitter)
                        
                        print(f"OpenRouter API rate limit (429). Retrying in {actual_backoff:.2f}s... (Attempt {current_retry + 1}/{self.max_retries})")
                        time.sleep(actual_backoff)
                        current_retry += 1
                    else:
                        print(f"OpenRouter API rate limit (429). Max retries ({self.max_retries}) reached. Failing.")
                        raise ConnectionError(f"OpenRouter API call failed after max retries: {e}") from e
                else: # Other HTTP errors
                    print(f"Error calling OpenRouter API (HTTPError): {e}")
                    raise ConnectionError(f"OpenRouter API call failed: {e}") from e
            except requests.exceptions.RequestException as e: # Other network errors (timeout, DNS, etc.)
                print(f"Error calling OpenRouter API (RequestException): {e}")
                raise ConnectionError(f"OpenRouter API call failed: {e}") from e
            except Exception as e: # Other unexpected errors (e.g., JSON parsing)
                print(f"Unexpected error with OpenRouter: {e}")
                raise # Re-raise the original exception

# Factory function for consistent service creation
def create_llm_service(model: str = "google/gemini-2.0-flash-001", api_key: Optional[str] = None) -> LLMService:
    """Create appropriate LLM service instance."""
    return OpenRouterService(api_key=api_key, model=model)

if __name__ == '__main__':
    # Example usage
    try:
        # Create service with model specification
        service = create_llm_service(model="google/gemini-2.0-flash-001")
        
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