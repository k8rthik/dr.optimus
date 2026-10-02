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
from droptimus.objectives.base import Objective
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


def rescore(
    generated: Sequence[Generated], objective: Objective
) -> tuple[Generated, ...]:
    """Return copies of ``generated`` scored under a different objective.

    Needed for the constrained task. The agent optimizes a penalty-adjusted
    score, but Zhou et al. Table 2 reports the improvement in the *base*
    objective (penalized logP) with the constraint success rate given
    separately. Mixing the penalty into the reported improvement would make a
    constraint violation look like a worse molecule.

    Molecule selection and similarity are untouched; only the scores change.
    """
    return tuple(
        Generated(
            start_smiles=record.start_smiles,
            smiles=record.smiles,
            objective=float(objective.score(record.smiles)),
            start_objective=float(objective.score(record.start_smiles)),
            similarity=record.similarity,
        )
        for record in generated
    )


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


#: Values transcribed from Zhou et al. (2019), "Optimization of Molecules via
#: Deep Reinforcement Learning", Scientific Reports 9:10752 / arXiv:1810.08678,
#: Table 1. Used only for the comparison column of generated reports -- never
#: mixed into measured numbers.
#:
#: Two MolDQN variants are reported there. ``naive`` is a single Q-network with
#: epsilon-greedy exploration, which is what this code implements; ``bootstrap``
#: adds an ensemble of Q-heads for exploration, which this code does not. The
#: naive row is therefore the fair comparison and the bootstrap row is the
#: paper's headline. ``random_walk`` is the paper's own random-action baseline,
#: included because it is directly comparable to the random-edit baseline here.
#:
#: Table 1 reports the top three scores among "the last 100 terminal states in
#: the training process", so the comparable measurement here is over *final*
#: episode molecules, not the best molecule visited mid-episode.
PUBLISHED_MOLDQN: Mapping[str, Mapping[str, object]] = {
    "qed": {
        "naive": (0.934, 0.931, 0.930),
        "bootstrap": (0.948, 0.944, 0.943),
        "random_walk": (0.64, 0.56, 0.56),
        "max_steps": 40,
    },
    "penalized_logp": {
        "naive": (11.51, 11.51, 11.50),
        "bootstrap": (11.84, 11.84, 11.82),
        "random_walk": (-3.99, -4.31, -4.37),
        "max_steps": 38,
    },
}

#: Table 2: mean and standard deviation of penalized logP improvement under a
#: Tanimoto similarity constraint on the 800 lowest-penalized-logP ZINC
#: molecules, one episode each, 20 steps per episode. Keyed by delta, values are
#: ``(naive mean, naive sd, bootstrap mean, bootstrap sd, success rate)``.
PUBLISHED_CONSTRAINED: Mapping[float, tuple[float, float, float, float, float]] = {
    0.0: (6.83, 1.30, 7.04, 1.42, 1.0),
    0.2: (5.00, 1.55, 5.06, 1.79, 1.0),
    0.4: (3.13, 1.57, 3.37, 1.62, 1.0),
    0.6: (1.40, 1.05, 1.86, 1.21, 1.0),
}


def comparison_rows(
    objective: str, metrics: Metrics
) -> tuple[tuple[str, str, str, str], ...]:
    """Return ``(label, measured, MolDQN-naive, MolDQN-bootstrap)`` table rows.

    ``metrics`` should be computed over *final* episode molecules, matching the
    paper's protocol of reporting the top three of the last 100 terminal states.
    An objective with no published number gets "n/a" rather than a fabricated
    one.
    """
    published = PUBLISHED_MOLDQN.get(objective, {})
    naive = published.get("naive", ())
    bootstrap = published.get("bootstrap", ())
    rows: list[tuple[str, str, str, str]] = []
    for index in range(3):
        measured = f"{metrics.top3[index]:.3f}" if index < len(metrics.top3) else "n/a"
        rows.append(
            (
                f"best #{index + 1}",
                measured,
                f"{naive[index]:.3f}" if index < len(naive) else "n/a",  # type: ignore[index]
                f"{bootstrap[index]:.3f}" if index < len(bootstrap) else "n/a",  # type: ignore[index]
            )
        )
    rows.append(
        (
            "mean over run",
            f"{metrics.objective_mean:.3f} +/- {metrics.objective_std:.3f}",
            "not reported",
            "not reported",
        )
    )
    return tuple(rows)


def published_random_walk(objective: str) -> float | None:
    """Return the paper's own random-action baseline best, if it reported one."""
    values = PUBLISHED_MOLDQN.get(objective, {}).get("random_walk", ())
    return float(values[0]) if values else None  # type: ignore[index]


def constrained_comparison(delta: float, metrics: Metrics) -> str:
    """Render the constrained-task comparison against Zhou et al. Table 2."""
    published = PUBLISHED_CONSTRAINED.get(round(delta, 2))
    measured = (
        f"{metrics.improvement_mean:+.2f} +/- {metrics.improvement_std:.2f}, "
        f"success {metrics.constraint_satisfied:.1%}"
        if metrics.constraint_satisfied is not None
        else f"{metrics.improvement_mean:+.2f} +/- {metrics.improvement_std:.2f}"
    )
    if published is None:
        return (
            f"constrained improvement at delta={delta:g}: {measured}\n"
            f"  the paper reports deltas "
            f"{', '.join(f'{d:g}' for d in sorted(PUBLISHED_CONSTRAINED))}; "
            "no published value is quoted for this one"
        )
    naive_mean, naive_sd, boot_mean, boot_sd, success = published
    return (
        f"constrained improvement at delta={delta:g}\n"
        f"  this run              {measured}\n"
        f"  MolDQN-naive          {naive_mean:+.2f} +/- {naive_sd:.2f}, "
        f"success {success:.0%}\n"
        f"  MolDQN-bootstrap      {boot_mean:+.2f} +/- {boot_sd:.2f}, "
        f"success {success:.0%}"
    )


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
