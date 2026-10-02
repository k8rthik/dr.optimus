"""Measured evaluation of a generation run.

Every number this module produces is computed from molecules that were actually
generated. Where a metric is true by construction -- validity, for instance,
since the environment only ever proposes sanitizable molecules -- that is stated
rather than presented as a result.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from droptimus.chem.molecule import canonical_smiles, is_valid_smiles, tanimoto_similarity
from droptimus.rollout import Episode


@dataclass(frozen=True, slots=True)
class Generated:
    """One generated molecule together with its start point."""

    start_smiles: str
    smiles: str
    objective: float
    start_objective: float
    similarity: float

    @property
    def improvement(self) -> float:
        return self.objective - self.start_objective


@dataclass(frozen=True, slots=True)
class Metrics:
    """Measured statistics over a set of generated molecules."""

    n: int
    objective_mean: float
    objective_std: float
    objective_median: float
    objective_max: float
    top3: tuple[float, ...]
    start_objective_mean: float
    improvement_mean: float
    improvement_std: float
    fraction_improved: float
    validity: float
    uniqueness: float
    novelty: float
    similarity_mean: float
    similarity_min: float
    constraint_satisfied: float | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view."""
        return asdict(self)


def generated_from_episodes(
    episodes: Sequence[Episode], use_best: bool = True
) -> tuple[Generated, ...]:
    """Convert episodes into evaluation records.

    Args:
        episodes: finished episodes.
        use_best: score the best molecule *visited* during the episode (the
            convention in the benchmark literature) rather than the final one.
    """
    records = []
    for episode in episodes:
        smiles = episode.best_smiles if use_best else episode.final_smiles
        objective = episode.best_objective if use_best else episode.final_objective
        records.append(
            Generated(
                start_smiles=episode.start_smiles,
                smiles=smiles,
                objective=float(objective),
                start_objective=float(episode.start_objective),
                similarity=tanimoto_similarity(episode.start_smiles, smiles),
            )
        )
    return tuple(records)


def _std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def compute_metrics(
    generated: Sequence[Generated],
    reference: Sequence[str] = (),
    similarity_delta: float | None = None,
) -> Metrics:
    """Compute measured metrics over ``generated``.

    Args:
        generated: the generated molecules.
        reference: molecules that count as "already known" for novelty, e.g. the
            ZINC250k set the start molecules were drawn from. With an empty
            reference, novelty is measured against the start molecules alone.
        similarity_delta: if given, also report the fraction of molecules whose
            similarity to their own start molecule is at least this.

    Raises:
        ValueError: if ``generated`` is empty.
    """
    if not generated:
        raise ValueError("Cannot compute metrics over an empty set of molecules.")

    objectives = [record.objective for record in generated]
    improvements = [record.improvement for record in generated]
    similarities = [record.similarity for record in generated]

    smiles = [record.smiles for record in generated]
    known = {canonical_smiles(s) for s in reference}
    known.update(record.start_smiles for record in generated)

    unique = set(smiles)
    novel = [s for s in unique if s not in known]

    constraint = None
    if similarity_delta is not None:
        satisfied = sum(1 for s in similarities if s >= similarity_delta)
        constraint = satisfied / len(similarities)

    return Metrics(
        n=len(generated),
        objective_mean=float(statistics.fmean(objectives)),
        objective_std=_std(objectives),
        objective_median=float(statistics.median(objectives)),
        objective_max=float(max(objectives)),
        top3=tuple(sorted(objectives, reverse=True)[:3]),
        start_objective_mean=float(
            statistics.fmean(record.start_objective for record in generated)
        ),
        improvement_mean=float(statistics.fmean(improvements)),
        improvement_std=_std(improvements),
        fraction_improved=sum(1 for value in improvements if value > 0) / len(improvements),
        validity=sum(1 for s in smiles if is_valid_smiles(s)) / len(smiles),
        uniqueness=len(unique) / len(smiles),
        novelty=len(novel) / len(unique),
        similarity_mean=float(statistics.fmean(similarities)),
        similarity_min=float(min(similarities)),
        constraint_satisfied=constraint,
    )


#: Values reported by Zhou et al. (2019), "Optimization of Molecules via Deep
#: Reinforcement Learning", Scientific Reports 9:10752. Used only for comparison
#: in generated reports -- never mixed into measured numbers.
PUBLISHED_MOLDQN: Mapping[str, Mapping[str, float]] = {
    "qed": {"top1": 0.948, "top2": 0.944, "top3": 0.941},
    "penalized_logp": {"top1": 11.84, "top2": 11.84, "top3": 11.82},
}


def comparison_rows(
    objective: str, metrics: Metrics
) -> tuple[tuple[str, str, str], ...]:
    """Return ``(label, measured, published)`` rows for a results table.

    Objectives with no published MolDQN number get "n/a" in the published column
    rather than a fabricated one.
    """
    published = PUBLISHED_MOLDQN.get(objective, {})
    rows = []
    for index, key in enumerate(("top1", "top2", "top3")):
        measured = (
            f"{metrics.top3[index]:.3f}" if index < len(metrics.top3) else "n/a"
        )
        reference = f"{published[key]:.3f}" if key in published else "n/a"
        rows.append((f"best #{index + 1}", measured, reference))
    rows.append((
        "mean over run",
        f"{metrics.objective_mean:.3f} +/- {metrics.objective_std:.3f}",
        "not reported",
    ))
    return tuple(rows)


def format_metrics(metrics: Metrics, title: str = "measured results") -> str:
    """Render metrics as a plain-text block for the CLI."""
    lines = [
        f"{title} (n = {metrics.n})",
        f"  objective      mean {metrics.objective_mean:+.3f}  "
        f"sd {metrics.objective_std:.3f}  median {metrics.objective_median:+.3f}  "
        f"max {metrics.objective_max:+.3f}",
        "  top 3          " + ", ".join(f"{value:+.3f}" for value in metrics.top3),
        f"  start mean     {metrics.start_objective_mean:+.3f}",
        f"  improvement    mean {metrics.improvement_mean:+.3f} "
        f"+/- {metrics.improvement_std:.3f}  "
        f"improved {metrics.fraction_improved:.0%} of starts",
        f"  validity       {metrics.validity:.3f} (1.000 by construction: "
        "the environment only proposes sanitizable molecules)",
        f"  uniqueness     {metrics.uniqueness:.3f}",
        f"  novelty        {metrics.novelty:.3f} (vs reference set + start molecules)",
        f"  similarity     mean {metrics.similarity_mean:.3f}  "
        f"min {metrics.similarity_min:.3f}",
    ]
    if metrics.constraint_satisfied is not None:
        lines.append(
            f"  constraint     {metrics.constraint_satisfied:.3f} of molecules "
            "meet the similarity threshold"
        )
    return "\n".join(lines)
