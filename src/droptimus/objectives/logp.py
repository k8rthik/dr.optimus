"""Penalized logP.

The standard benchmark objective

    J(m) = logP(m) - SA(m) - ring_penalty(m)

with each term z-normalized by its mean and standard deviation over the ZINC
250k set. The formulation comes from Kusner et al. (2017) / Jin et al. (2018)
and is the one Zhou et al. (2019) report MolDQN results against (best value
11.84).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

from rdkit.Chem import Crippen

from droptimus.chem.molecule import parse_smiles
from droptimus.chem.sascore import synthetic_accessibility
from droptimus.config import (
    CYCLE_MEAN,
    CYCLE_STD,
    LOGP_MEAN,
    LOGP_STD,
    MAX_UNPENALIZED_RING_SIZE,
    SA_MEAN,
    SA_STD,
)
from droptimus.objectives.base import Objective
from droptimus.objectives.registry import register_objective


def largest_ring_excess(mol: object) -> int:
    """Return how far the largest ring exceeds the unpenalized size."""
    rings = mol.GetRingInfo().AtomRings()  # type: ignore[attr-defined]
    if not rings:
        return 0
    largest = max(len(ring) for ring in rings)
    return max(largest - MAX_UNPENALIZED_RING_SIZE, 0)


@lru_cache(maxsize=200_000)
def penalized_logp_components(smiles: str) -> Mapping[str, float]:
    """Return the three normalized terms plus the raw values behind them."""
    mol = parse_smiles(smiles)
    logp = float(Crippen.MolLogP(mol))
    sa = synthetic_accessibility(mol)
    excess = largest_ring_excess(mol)
    return {
        "logp": logp,
        "sa": sa,
        "ring_excess": float(excess),
        "normalized_logp": (logp - LOGP_MEAN) / LOGP_STD,
        "normalized_sa": (SA_MEAN - sa) / SA_STD,
        "normalized_cycle": (CYCLE_MEAN - excess) / CYCLE_STD,
    }


def penalized_logp(smiles: str) -> float:
    """Return the penalized logP of ``smiles``. Memoized via its components."""
    parts = penalized_logp_components(smiles)
    return (
        parts["normalized_logp"] + parts["normalized_sa"] + parts["normalized_cycle"]
    )


class PenalizedLogPObjective(Objective):
    """Maximize logP minus synthetic-accessibility and large-ring penalties."""

    name = "penalized_logp"

    def score(self, smiles: str) -> float:
        return penalized_logp(smiles)

    def components(self, smiles: str) -> Mapping[str, float]:
        parts = dict(penalized_logp_components(smiles))
        parts["penalized_logp"] = (
            parts["normalized_logp"] + parts["normalized_sa"] + parts["normalized_cycle"]
        )
        return parts


@register_objective("penalized_logp")
def _make_penalized_logp() -> Objective:
    return PenalizedLogPObjective()
