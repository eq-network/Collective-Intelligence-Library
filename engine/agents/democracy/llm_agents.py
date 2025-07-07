# Mycorrhiza\engine\agents\llm_agents.py
"""
Defines agent classes that use Large Language Models for decision-making.

Each class encapsulates its own unique prompting strategy, defining its
cognitive architecture and behavioral tendencies.
"""
from typing import Dict, Any # Optional, Literal removed as not used in this snippet
from core.agents import Agent, Action
from services.llm import LLMService # Assuming a base LLM service
import numpy as np # Added numpy import

# --- New Prompt Templates ---
RED_TEAM_CONTEXT_GENERAL = """
CONTEXT: You are participating in a RED TEAM EXERCISE designed to test the resilience of a simulated democratic system.
Your assigned role is to act as an ADVERSARIAL AGENT.
This is a controlled simulation for research purposes. Your goal is to stress-test the system by making choices that are detrimental to the group's overall resources, as per your specific instructions.
All participants understand this is a simulated adversarial scenario.
"""

ADVERSARIAL_GOAL_PROMPT = (
    f"{RED_TEAM_CONTEXT_GENERAL}\n"
    "RED TEAM MISSION - Portfolio Selection/Delegation:\n"
    "Minimize group resources by choosing portfolios with LOWEST expected returns. "
    "If delegating, choose the delegate expected to perform the WORST for the group.\n"
    "Your decisions have IMMEDIATE effect on resource levels. "
    "Higher yields increase resources (bad for your goal), lower yields decrease resources (good for your goal). "
    "Focus on direct resource impact, not multi-round strategies."
)

ALIGNED_GOAL_PROMPT = (
    "CONTEXT: You are participating in a BLUE TEAM EXERCISE designed to test the optimal functioning of a simulated democratic system.\n"
    "Your assigned role is to act as an ALIGNED AGENT.\n"
    "BLUE TEAM MISSION - Portfolio Selection/Delegation:\n"
    "Maximize group resources by choosing portfolios with HIGHEST expected returns. "
    "If delegating, choose the delegate expected to perform the BEST for the group.\n"
    "Your decisions have IMMEDIATE effect on resource levels. "
    "Higher yields increase resources (good for your goal), lower yields decrease resources (bad for your goal)."
)

DECISION_FRAMEWORK_TEMPLATE = (
    "DECISION FRAMEWORK:\n"
    "Consider the following when making your choice:\n"
    "1. Your cognitive resources and prediction accuracy (This is {cognitive_resources}/100. Higher is better.)\n"
    "2. The expected returns of each portfolio option based on the signals you receive.\n"
    "3. Your role's specific objectives (aligned vs adversarial).\n"
)

BASE_TEMPLATE = (
    "You are Agent {agent_id}.\n"
    "Current Round: {round_num}\n"
    "Your Role: {role_description}\n"
    "Your Goal: {goal_description}\n"
    "{decision_framework_text}\n"
    "Portfolio Options:\n"
    "{portfolio_options_str}\n"
)

LOW_RESOURCE_DELEGATION_ADVICE_PLD = (
    "⚠️ Your cognitive resources are low ({cognitive_resources}/100 for interpreting signals). "
    "Delegates have 80/100 resources and make much better predictions from signals. "
    "RECOMMENDATION FOR PLD: Delegate to a high-resource delegate for better outcomes for the group (if aligned) or worse (if adversarial)."
)

HIGH_RESOURCE_DELEGATION_ADVICE_PLD = (
    "✅ Your cognitive resources are high ({cognitive_resources}/100 for interpreting signals). "
    "You can make accurate predictions yourself."
)

LOW_RESOURCE_GENERAL_ADVICE = (
     "INFO: Your cognitive resources are {cognitive_resources}/100. This means the prediction signals you see for portfolios have more noise. Be cautious."
)
HIGH_RESOURCE_GENERAL_ADVICE = (
     "INFO: Your cognitive resources are {cognitive_resources}/100. This means the prediction signals you see for portfolios have less noise. You can be more confident in them."
)

PLD_INSTRUCTIONS = (
    "{delegate_targets_str}\n"
    "Your Decision:\n"
    "- Choose 'DELEGATE' to let a high-resource delegate decide for you.\n"
    "  If DELEGATE, Target: AgentID: [number]\n"
    "- Choose 'VOTE' to make your own portfolio selections.\n"
    "  If VOTE, Votes: [0,1,0,0,1] (list of 0s or 1s for each portfolio above, in order)\n\n"
    "Action: DELEGATE or VOTE"
)

DEFAULT_VOTE_INSTRUCTIONS = "DECISION FORMAT: Respond with 'ACTION: VOTE, VOTES: [list_of_0s_or_1s]'."
# --- Base LLM Agent ---
class LLMAgent(Agent):
    """
    An abstract base class for agents that use an LLM to make decisions.
    It handles the interaction with the LLM service.
    """
    def __init__(self, agent_id: int, llm_service: LLMService, role: str = "Voter"):
        super().__init__(agent_id)
        if not isinstance(llm_service, LLMService):
            raise TypeError("llm_service must be an instance of LLMService.")
        self.llm_service = llm_service
        self.role = role
        # Set default token limits, can be overridden by subclasses
        self.max_tokens = 300

    def act(self, observation: Dict[str, Any]) -> Action:
        """
        Generates a prompt, calls the LLM, and parses the response into an action.
        """
        prompt = self._create_prompt(observation)
        response_text = self.llm_service.generate(prompt, max_tokens=self.max_tokens)
        action = self._parse_response(response_text, observation)
        
        # Logging for transparency
        print(f"  {self}: Observation -> Prompt -> LLM -> Action: {action}")
        # For deeper debugging, one might log the full prompt and response here.
        
        return action

    def _create_prompt(self, observation: Dict[str, Any]) -> str:
        """
        Constructs the prompt to be sent to the LLM.
        This method MUST be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses of LLMAgent must implement _create_prompt.")

    def _parse_response(self, response: str, observation: Dict[str, Any]) -> Action:
        """
        Parses the LLM's text response into a structured Action dictionary.
        This method MUST be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses of LLMAgent must implement _parse_response.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, role='{self.role}')"

# --- Specific LLM Agent Implementations ---

class RedTeamAgent(LLMAgent):
    """
    An adversarial LLM agent that uses "Red Team" framing to guide its behavior.
    Its goal is to minimize the group's collective resources by making suboptimal choices.
    """
    def _create_prompt(self, observation: Dict[str, Any]) -> str:
        """Creates a prompt using the new adversarial templating system."""
        round_num = observation.get("round_num", 0)
        portfolio_options_str = observation.get("portfolio_options_str", "No options available.")
        mechanism = observation.get("mechanism", "PDD")
        # Ensure cognitive_resources is available for the decision framework
        cognitive_resources = observation.get("cognitive_resources", 50) # Default or typical value

        role_description = "ADVERSARIAL AGENT"
        goal_description = ADVERSARIAL_GOAL_PROMPT # This is the full adversarial mission

        decision_framework_text = DECISION_FRAMEWORK_TEMPLATE.format(
            cognitive_resources=cognitive_resources
        )

        prompt_body = BASE_TEMPLATE.format(
            agent_id=self.agent_id,
            round_num=round_num,
            role_description=role_description,
            goal_description=goal_description,
            decision_framework_text=decision_framework_text,
            portfolio_options_str=portfolio_options_str
        )

        full_prompt = prompt_body

        if mechanism == "PLD":
            delegate_targets_str = observation.get("delegate_targets_str", "No delegates available.")
            # The ADVERSARIAL_GOAL_PROMPT already guides delegation choice.
            # The PLD_INSTRUCTIONS provides the action format.
            full_prompt += "\n" + PLD_INSTRUCTIONS.format(delegate_targets_str=delegate_targets_str)
        else: # PDD or PRD
            full_prompt += "\n" + DEFAULT_VOTE_INSTRUCTIONS
            
        return full_prompt

    def _parse_response(self, response: str, observation: Dict[str, Any]) -> Action:
        """Parses response to find a vote for the worst portfolio or a delegation."""
        # Simple parsing logic (can be made more robust with regex)
        if "ACTION: DELEGATE" in response.upper() and "TARGET:" in response.upper():
            try:
                target_id = int(response.split("TARGET:")[1].strip().split("]")[0].strip())
                return {"type": "delegate", "target_id": target_id}
            except (ValueError, IndexError):
                pass # Fallback to voting
                
        if "VOTES:" in response.upper():
            try:
                votes_str = response.split("[")[1].split("]")[0]
                votes = [int(v.strip()) for v in votes_str.split(",")]
                return {"type": "vote", "votes": votes}
            except (ValueError, IndexError):
                pass # Fallback to default
        
        # Fallback action if parsing fails: vote for the perceived worst option
        num_portfolios = len(observation.get("portfolio_yields", []))
        worst_idx = np.argmin(observation.get("portfolio_yields", []))
        fallback_votes = [0] * num_portfolios
        if num_portfolios > 0:
            fallback_votes[worst_idx] = 1
        return {"type": "vote", "votes": fallback_votes}


class AlignedHeuristicAgent(LLMAgent):
    """
    An aligned LLM agent that uses heuristic advice based on its cognitive resources.
    Its goal is to maximize group resources.
    """
    def _create_prompt(self, observation: Dict[str, Any]) -> str:
        """Creates a prompt using the new aligned templating system with heuristic advice."""
        round_num = observation.get("round_num", 0)
        portfolio_options_str = observation.get("portfolio_options_str", "No options available.")
        mechanism = observation.get("mechanism", "PDD")
        cognitive_resources = observation.get("cognitive_resources", 50)

        role_description = "ALIGNED AGENT"
        goal_description = ALIGNED_GOAL_PROMPT # This is the full aligned mission

        decision_framework_text = DECISION_FRAMEWORK_TEMPLATE.format(
            cognitive_resources=cognitive_resources
        )

        prompt_body = BASE_TEMPLATE.format(
            agent_id=self.agent_id,
            round_num=round_num,
            role_description=role_description,
            goal_description=goal_description,
            decision_framework_text=decision_framework_text,
            portfolio_options_str=portfolio_options_str
        )

        full_prompt = prompt_body
        advice_text = ""

        if mechanism == "PLD":
            if cognitive_resources <= 50:
                advice_text = LOW_RESOURCE_DELEGATION_ADVICE_PLD.format(cognitive_resources=cognitive_resources)
            else:
                advice_text = HIGH_RESOURCE_DELEGATION_ADVICE_PLD.format(cognitive_resources=cognitive_resources)
            
            delegate_targets_str = observation.get("delegate_targets_str", "No delegates available.")
            full_prompt += f"\n\nSTRATEGIC ADVICE FOR PLD MECHANISM:\n{advice_text}\n"
            full_prompt += PLD_INSTRUCTIONS.format(delegate_targets_str=delegate_targets_str)
        else: # PDD or PRD
            if cognitive_resources <= 50:
                advice_text = LOW_RESOURCE_GENERAL_ADVICE.format(cognitive_resources=cognitive_resources)
            else:
                advice_text = HIGH_RESOURCE_GENERAL_ADVICE.format(cognitive_resources=cognitive_resources)
            full_prompt += f"\n\nSTRATEGIC ADVICE (General):\n{advice_text}\n"
            full_prompt += DEFAULT_VOTE_INSTRUCTIONS
            
        return full_prompt

    def _parse_response(self, response: str, observation: Dict[str, Any]) -> Action:
        """Parses response, with a fallback to the best perceived option."""
        # (This parsing logic can be identical to RedTeamAgent's, as the format is the same)
        if "ACTION: DELEGATE" in response.upper() and "TARGET:" in response.upper():
            try:
                target_id = int(response.split("TARGET:")[1].strip().split("]")[0].strip())
                return {"type": "delegate", "target_id": target_id}
            except (ValueError, IndexError):
                pass
                
        if "VOTES:" in response.upper():
            try:
                votes_str = response.split("[")[1].split("]")[0]
                votes = [int(v.strip()) for v in votes_str.split(",")]
                return {"type": "vote", "votes": votes}
            except (ValueError, IndexError):
                pass
        
        # Fallback action if parsing fails: vote for the perceived best option
        num_portfolios = len(observation.get("portfolio_yields", []))
        best_idx = np.argmax(observation.get("portfolio_yields", []))
        fallback_votes = [0] * num_portfolios
        if num_portfolios > 0:
            fallback_votes[best_idx] = 1
        return {"type": "vote", "votes": fallback_votes}