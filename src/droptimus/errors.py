"""Exception hierarchy for dr.optimus.

Every failure mode that can be caused by user input has its own exception so
that the CLI can turn it into a short, actionable message instead of a
traceback.
"""

from __future__ import annotations


class DrOptimusError(Exception):
    """Base class for all errors raised by this package."""


class InvalidSmilesError(DrOptimusError, ValueError):
    """Raised when a SMILES string cannot be parsed or sanitized by RDKit."""

    def __init__(self, smiles: str, reason: str = "RDKit could not parse it") -> None:
        self.smiles = smiles
        self.reason = reason
        super().__init__(f"Invalid SMILES {smiles!r}: {reason}.")


class ObjectiveError(DrOptimusError):
    """Raised when an objective is misconfigured or cannot be evaluated."""


class UnknownObjectiveError(ObjectiveError, KeyError):
    """Raised when an objective name is not present in the registry."""

    def __init__(self, name: str, available: tuple[str, ...]) -> None:
        self.name = name
        self.available = available
        listed = ", ".join(available) if available else "<none registered>"
        super(KeyError, self).__init__(
            f"Unknown objective {name!r}. Available objectives: {listed}."
        )

    def __str__(self) -> str:  # KeyError.__str__ would add quotes
        listed = ", ".join(self.available) if self.available else "<none registered>"
        return f"Unknown objective {self.name!r}. Available objectives: {listed}."


class EnvironmentError_(DrOptimusError):
    """Raised on illegal use of the molecule MDP."""


class EpisodeEndedError(EnvironmentError_):
    """Raised when stepping an environment whose episode has already ended."""


class InvalidActionError(EnvironmentError_):
    """Raised when an action is not in the current valid-action set."""


class ConfigError(DrOptimusError, ValueError):
    """Raised when a configuration value is out of range or inconsistent."""


class CheckpointError(DrOptimusError):
    """Raised when a checkpoint is missing or incompatible."""
