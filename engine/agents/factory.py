# Create new file: engine/agents/factory.py

"""
Factory that wraps existing agent creation but adds profile support.
"""

class ProfileAwareAgentFactory:
    """Wraps existing agent creation with profile support"""
    
    def __init__(self, legacy_factory=None):
        """
        legacy_factory: The existing agent creation function/class
        Find and pass the current agent creation mechanism
        """
        self.legacy_factory = legacy_factory
    
    def create_agents(self, config, use_profiles=False, **kwargs):
        """
        Create agents with optional profile support.
        
        Args:
            config: The existing configuration object
            use_profiles: If True, return AgentProfiles. If False, return legacy agents.
            **kwargs: Any additional parameters the legacy factory needs
        """
        if not use_profiles:
            # Use existing agent creation exactly as before
            if self.legacy_factory:
                return self.legacy_factory(config, **kwargs)
            else:
                # Fallback to however agents are currently created
                pass
        
        # Create enhanced agents
        agents = []
        num_agents = config.get('num_agents', 100)  # Adapt to actual config structure
        num_adversarial = config.get('num_adversarial', 20)
        
        for i in range(num_agents):
            profile = AgentProfile(
                agent_id=i,
                is_adversarial=(i < num_adversarial),
                cognitive_resources=config.get('cognitive_resources', 50),
                behavioral_type="adversarial" if i < num_adversarial else "aligned",
                cultural_profile=self._get_cultural_profile(config),
                trust_level=config.get('trust_level', 0.5),
                risk_tolerance=config.get('risk_tolerance', 0.5)
            )
            agents.append(profile)
        
        # If mechanism expects legacy agents, convert
        if config.get('legacy_mode', True):
            return [agent.to_legacy_agent() for agent in agents]
        
        return agents
    
    def _get_cultural_profile(self, config):
        """Extract cultural settings from config if present"""
        if 'cultural_context' not in config:
            return None
            
        # Define cultural presets
        presets = {
            'individualistic': CulturalProfile(0.7, 0.3, 0.4, 0.6),
            'collectivistic': CulturalProfile(-0.6, 0.4, 0.6, 0.8),
            'hierarchical': CulturalProfile(0.0, 0.8, 0.7, 0.5)
        }
        
        return presets.get(config['cultural_context'])