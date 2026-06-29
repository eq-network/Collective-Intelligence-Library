"""Mechanisms catalog — composed institutions as typed transforms.

Type function:  ``TransformFactory = Config -> Transform``. A mechanism is a Transform
that declares its ``.reads`` / ``.writes`` (via ``@transform``) so ``compile_pipeline``
can order it; entries of the same *family* keep disjoint write sets so ``parallel``
composition is always valid.

Families (Plan 2): ``market`` · ``network`` · ``democracy``. Only ``market`` is seeded
today — the variants (double-auction, gossip, liquid democracy …) are the first
open-source follow-ups. See README.md for the contract.
"""
from .market import create_market_transform

# name -> factory ((cfg) -> Transform).
REGISTRY = {
    "market": create_market_transform,
}

__all__ = ["create_market_transform", "REGISTRY"]
