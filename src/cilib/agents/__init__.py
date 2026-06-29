"""Agents catalog — pre-made, swappable decision rules.

Type function:  ``AgentFactory = Config -> Policy``  where a ``Policy`` is a callable
``(obs, key) -> action`` (see ``cilib.core.protocols``). The catalog is the explicit
``REGISTRY`` dict below — open this file to see every entry. See README.md to add one.
"""
from .simple import RandomPolicy, TitForTatPolicy
from .learnable import LinearPolicy

# name -> factory (here, the Policy class constructor).
REGISTRY = {
    "random": RandomPolicy,
    "tit_for_tat": TitForTatPolicy,
    "linear": LinearPolicy,
}

__all__ = ["RandomPolicy", "TitForTatPolicy", "LinearPolicy", "REGISTRY"]
