# Create new file: engine/agents/profiles.py

"""
Create classes that WRAP existing agents, not replace them.
This ensures backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

@dataclass
class CulturalProfile:
    """Cultural parameters that affect agent behavior"""
    individualism_collectivism: float = 0.0  # -1 to 1
    power_distance: float = 0.5  # 0 to 1
    uncertainty_avoidance: float = 0.5  # 0 to 1
    trust_orientation: float = 0.5  # 0 to 1
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for compatibility"""
        return {
            'individualism': self.individualism_collectivism,
            'power_distance': self.power_distance,
            'uncertainty_avoidance': self.uncertainty_avoidance,
            'trust': self.trust_orientation
        }

@dataclass
class AgentProfile:
    """
    Enhanced agent profile that maintains compatibility with existing system.
    
    IMPORTANT: Must include all fields that existing agents have.
    Look at current Agent class and include ALL its fields here.
    """
    # Core fields (MUST match existing agent fields exactly)
    agent_id: int
    is_adversarial: bool
    cognitive_resources: int
    # ADD any other fields that existing agents have
    
    # New optional fields (with defaults so existing code works)
    behavioral_type: Literal["aligned", "adversarial", "strategic", "naive"] = "aligned"
    cultural_profile: Optional[CulturalProfile] = None
    trust_level: float = 0.5
    risk_tolerance: float = 0.5
    
    def to_legacy_agent(self):
        """
        Convert to existing agent format.
        IMPORTANT: Return object must be compatible with current system.
        
        Look at how existing agents are created and match that exactly.
        """
        # If existing agents are dicts:
        return {
            'agent_id': self.agent_id,
            'is_adversarial': self.is_adversarial,
            'cognitive_resources': self.cognitive_resources,
            # Add other required fields
        }
        
        # If existing agents are objects, create appropriate object
    
    @classmethod
    def from_legacy_agent(cls, legacy_agent):
        """
        Create profile from existing agent.
        Extract whatever fields the legacy agent has.
        """
        # Adapt based on whether legacy agents are dicts or objects
        pass