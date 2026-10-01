"""Objective interface.

An objective maps a SMILES string to a scalar the agent maximizes. Objectives
are stateless after construction and therefore safe to share and to cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping


class Objective(ABC):
    """A scalar function of a molecule, to be maximized."""

    #: Human-readable identifier that also records the configuration, so a
    #: results table can be read without the original command line.
    name: str = "objective"

    @abstractmethod
    def score(self, smiles: str) -> float:
        """Return the objective value of ``smiles`` (higher is better)."""

    def components(self, smiles: str) -> Mapping[str, float]:
        """Return a breakdown of the score for reporting and debugging."""
        return {self.name: self.score(smiles)}

    def __call__(self, smiles: str) -> float:
        return self.score(smiles)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"
