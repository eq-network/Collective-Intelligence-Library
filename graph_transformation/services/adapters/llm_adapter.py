# implementations/adapters/llm_adapter.py
from typing import TypeVar, Generic, Callable, List, Set, Dict, Any
from transformations.bottom_up.updating import belief_update_transform
from services.llm import LLMService
from core.graph import GraphState

def create_llm_belief_updater(
    llm_service: LLMService,
    prompt_template: str = None
) -> Callable[[GraphState], GraphState]:
    """
    Create a belief updater that uses an LLM service.
    
    Args:
        llm_service: LLM service to use
        prompt_template: Template for prompts (optional)
    
    Returns:
        A function that can be applied to a GraphState
    """
    if prompt_template is None:
        prompt_template = """
        Current beliefs: {beliefs}
        Messages from neighbors: {messages}
        Update your beliefs based on this information.
        """
    
    def update_function(beliefs: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Format beliefs and messages for the prompt
        beliefs_str = format_dict(beliefs)
        messages_str = format_messages(messages)
        
        # Create prompt
        prompt = prompt_template.format(
            beliefs=beliefs_str,
            messages=messages_str
        )
        
        # Call LLM service
        response = llm_service.generate(prompt)
        
        # Parse response into updated beliefs
        updated_beliefs = parse_beliefs(response)
        return updated_beliefs
    
    # Return a function that applies belief_update_transform with our update_function
    return lambda state: belief_update_transform(state, update_function)