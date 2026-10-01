"""Objective registry and built-in benchmark objectives.

Importing this package registers every built-in objective.
"""

from __future__ import annotations

from droptimus.objectives.base import Objective
from droptimus.objectives.registry import (
    available_objectives,
    make_objective,
    register_objective,
)

# Imported for their registration side effects.
from droptimus.objectives import logp as _logp  # noqa: E402,F401
from droptimus.objectives import qed as _qed  # noqa: E402,F401
from droptimus.objectives import similarity as _similarity  # noqa: E402,F401

#: Objectives that need a reference molecule (the start molecule) to be built.
REFERENCE_REQUIRING = ("similarity", "constrained")


def objective_needs_reference(name: str) -> bool:
    """Return whether ``name`` must be given a ``reference`` molecule."""
    return name.strip().lower() in REFERENCE_REQUIRING


__all__ = [
    "Objective",
    "available_objectives",
    "make_objective",
    "register_objective",
    "objective_needs_reference",
    "REFERENCE_REQUIRING",
]
