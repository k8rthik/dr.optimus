"""Similarity objectives.

Two things live here:

* ``similarity`` --- plain Tanimoto similarity to a reference molecule.
* ``constrained`` --- the similarity-constrained variant of the benchmark:
  maximize a base objective subject to Tanimoto similarity >= delta to the
  start molecule.

The constraint is enforced with a steep linear penalty rather than a hard
-inf, because a hard wall gives the agent no gradient to climb back over: a
molecule that violates the constraint scores ``base - PENALTY_WEIGHT * (delta -
similarity)``. Reported results always state the fraction of molecules that
actually satisfy the constraint, so the softness never hides a failure.
"""

from __future__ import annotations

from typing import Mapping

from droptimus.chem.molecule import canonical_smiles, tanimoto_similarity
from droptimus.config import DEFAULT_SIMILARITY_DELTA
from droptimus.errors import ObjectiveError
from droptimus.objectives.base import Objective
from droptimus.objectives.registry import make_objective, register_objective

#: Slope of the constraint penalty, in objective units per unit of similarity
#: shortfall. Large enough that no realistic objective gain pays for a
#: meaningful violation.
PENALTY_WEIGHT: float = 20.0


def _require_reference(reference: object) -> str:
    if reference is None:
        raise ObjectiveError(
            "This objective needs a reference molecule. Pass "
            "reference='<SMILES>' (the CLI uses the start molecule)."
        )
    if not isinstance(reference, str):
        raise ObjectiveError(f"reference must be a SMILES string, got {type(reference).__name__}.")
    return canonical_smiles(reference)


def _require_delta(delta: object) -> float:
    try:
        value = float(delta)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ObjectiveError(f"delta must be a number, got {delta!r}.") from exc
    if not 0.0 <= value <= 1.0:
        raise ObjectiveError(f"delta must lie in [0, 1], got {value}.")
    return value


class SimilarityObjective(Objective):
    """Maximize Tanimoto similarity to a fixed reference molecule."""

    def __init__(self, reference: str) -> None:
        self.reference = _require_reference(reference)
        self.name = "similarity"

    def score(self, smiles: str) -> float:
        return tanimoto_similarity(self.reference, smiles)

    def components(self, smiles: str) -> Mapping[str, float]:
        return {"similarity": self.score(smiles)}


class ConstrainedObjective(Objective):
    """Maximize ``base`` subject to similarity >= ``delta`` to ``reference``."""

    def __init__(
        self,
        reference: str,
        base: str = "penalized_logp",
        delta: float = DEFAULT_SIMILARITY_DELTA,
        penalty_weight: float = PENALTY_WEIGHT,
    ) -> None:
        if not isinstance(base, str):
            raise ObjectiveError(f"base must be an objective name, got {base!r}.")
        if base.strip().lower() in {"constrained", "similarity"}:
            raise ObjectiveError(
                f"base objective {base!r} cannot be used inside 'constrained'; "
                "choose a property objective such as 'qed' or 'penalized_logp'."
            )
        self.reference = _require_reference(reference)
        self.delta = _require_delta(delta)
        self.penalty_weight = float(penalty_weight)
        self.base = make_objective(base)
        self.name = f"constrained[{self.base.name},delta={self.delta:g}]"

    def score(self, smiles: str) -> float:
        base_value = self.base.score(smiles)
        similarity = tanimoto_similarity(self.reference, smiles)
        shortfall = max(self.delta - similarity, 0.0)
        return base_value - self.penalty_weight * shortfall

    def components(self, smiles: str) -> Mapping[str, float]:
        base_value = self.base.score(smiles)
        similarity = tanimoto_similarity(self.reference, smiles)
        shortfall = max(self.delta - similarity, 0.0)
        return {
            "base_objective": base_value,
            "similarity": similarity,
            "delta": self.delta,
            "constraint_satisfied": 1.0 if shortfall == 0.0 else 0.0,
            "penalty": self.penalty_weight * shortfall,
            "score": base_value - self.penalty_weight * shortfall,
        }


@register_objective("similarity")
def _make_similarity(reference: object = None, **_: object) -> Objective:
    return SimilarityObjective(_require_reference(reference))


@register_objective("constrained")
def _make_constrained(
    reference: object = None,
    base: object = "penalized_logp",
    delta: object = DEFAULT_SIMILARITY_DELTA,
    penalty_weight: object = PENALTY_WEIGHT,
    **_: object,
) -> Objective:
    return ConstrainedObjective(
        reference=_require_reference(reference),
        base=base,  # type: ignore[arg-type]
        delta=_require_delta(delta),
        penalty_weight=float(penalty_weight),  # type: ignore[arg-type]
    )
