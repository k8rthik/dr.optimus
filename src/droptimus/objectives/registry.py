"""Pluggable objective registry.

Objectives register themselves with :func:`register_objective` at import time;
:func:`make_objective` builds one by name. ``droptimus.objectives.__init__``
imports the built-in modules so that importing the package is enough to
populate the registry.
"""

from __future__ import annotations

from typing import Callable, Dict

from droptimus.errors import UnknownObjectiveError
from droptimus.objectives.base import Objective

ObjectiveFactory = Callable[..., Objective]

_REGISTRY: Dict[str, ObjectiveFactory] = {}


def register_objective(name: str) -> Callable[[ObjectiveFactory], ObjectiveFactory]:
    """Decorator registering an objective factory under ``name``."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Objective name must be a non-empty string.")

    def decorator(factory: ObjectiveFactory) -> ObjectiveFactory:
        if key in _REGISTRY and _REGISTRY[key] is not factory:
            raise ValueError(f"Objective {key!r} is already registered.")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_objectives() -> tuple[str, ...]:
    """Return the registered objective names, sorted."""
    return tuple(sorted(_REGISTRY))


def make_objective(name: str, **kwargs: object) -> Objective:
    """Build the objective registered under ``name``.

    Raises:
        UnknownObjectiveError: if no objective is registered under that name.
    """
    if not isinstance(name, str):
        raise UnknownObjectiveError(repr(name), available_objectives())
    key = name.strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise UnknownObjectiveError(name, available_objectives())
    return factory(**kwargs)
