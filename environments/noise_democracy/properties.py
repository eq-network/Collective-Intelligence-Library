# In domains/democracy/properties.py

import sys
from pathlib import Path

# Get the root directory (MYCORRHIZA)
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    
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