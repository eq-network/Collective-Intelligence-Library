# In domains/democracy/properties.py
from core.property import Property, ConservesSum

# Define democracy-specific properties
voting_power_conservation = ConservesSum("voting_power")
delegation_acyclicity = Property("delegation_acyclicity", "No cycles in delegation graph")
information_integrity = Property("information_integrity", "Information is not maliciously corrupted")

# Define democratic mechanism property categories
class DemocraticMechanismProperties:
    """Property categories for democratic mechanisms."""
    
    DIRECT_DEMOCRACY = {voting_power_conservation, information_integrity}
    REPRESENTATIVE_DEMOCRACY = {voting_power_conservation, information_integrity}
    LIQUID_DEMOCRACY = {voting_power_conservation, delegation_acyclicity, information_integrity}