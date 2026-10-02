"""Objective registry and built-in benchmark objectives.

Importing this package registers every built-in objective.
"""

from __future__ import annotations

# Imported for their registration side effects.
from droptimus.objectives import logp as _logp
from droptimus.objectives import qed as _qed
from droptimus.objectives import similarity as _similarity
from droptimus.objectives.base import Objective
from droptimus.objectives.registry import (
    available_objectives,
    make_objective,
    register_objective,
)

#: Objectives that need a reference molecule (the start molecule) to be built.
REFERENCE_REQUIRING = ("similarity", "constrained")


def objective_needs_reference(name: str) -> bool:
    """Return whether ``name`` must be given a ``reference`` molecule."""
    return name.strip().lower() in REFERENCE_REQUIRING


__all__ = [
    "REFERENCE_REQUIRING",
    "Objective",
    "available_objectives",
    "make_objective",
    "objective_needs_reference",
    "register_objective",
]
