"""Quantitative Estimate of Drug-likeness (QED), Bickerton et al. (2012).

QED is bounded in [0, 1]; the MolDQN paper reports a best single-molecule QED
of 0.948 for this objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

from rdkit.Chem import QED

from droptimus.chem.molecule import parse_smiles
from droptimus.objectives.base import Objective
from droptimus.objectives.registry import register_objective


@lru_cache(maxsize=200_000)
def qed_score(smiles: str) -> float:
    """Return RDKit's QED for ``smiles``. Memoized: a pure function of SMILES."""
    return float(QED.qed(parse_smiles(smiles)))


class QedObjective(Objective):
    """Maximize drug-likeness."""

    name = "qed"

    def score(self, smiles: str) -> float:
        return qed_score(smiles)

    def components(self, smiles: str) -> Mapping[str, float]:
        return {"qed": self.score(smiles)}


@register_objective("qed")
def _make_qed() -> Objective:
    return QedObjective()
