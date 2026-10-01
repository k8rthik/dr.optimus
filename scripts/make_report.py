#!/usr/bin/env python
"""Collect the metrics.json files a benchmark sweep produced into Markdown.

Every number printed comes from a metrics file written by ``droptimus evaluate``;
nothing is computed or estimated here. Runs whose metrics file is missing are
listed as missing rather than skipped silently.

Usage:  python scripts/make_report.py runs > runs/RESULTS.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

#: Published MolDQN values, for the comparison column only.
PUBLISHED = {
    "qed": "0.948",
    "penalized_logp": "11.84",
}


def load_runs(root: Path) -> list[tuple[str, dict[str, Any] | None]]:
    """Return ``(run name, metrics or None)`` for every directory under ``root``."""
    runs = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            runs.append((directory.name, None))
            continue
        runs.append((directory.name, json.loads(metrics_path.read_text())))
    return runs


def _row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def objective_table(runs: list[tuple[str, dict[str, Any] | None]]) -> str:
    header = [
        "run",
        "objective",
        "episodes",
        "best",
        "mean +/- sd",
        "random baseline best",
        "random baseline mean",
        "published MolDQN best",
    ]
    lines = [_row(header), _row(["---"] * len(header))]
    for name, metrics in runs:
        if metrics is None:
            lines.append(_row([name] + ["(no metrics.json)"] * (len(header) - 1)))
            continue
        agent = metrics["agent"]
        baseline = metrics.get("random_baseline")
        lines.append(
            _row(
                [
                    name,
                    str(metrics["objective"]),
                    str(metrics["episodes"]),
                    f"{agent['objective_max']:.3f}",
                    f"{agent['objective_mean']:.3f} +/- {agent['objective_std']:.3f}",
                    f"{baseline['objective_max']:.3f}" if baseline else "not run",
                    (
                        f"{baseline['objective_mean']:.3f} +/- "
                        f"{baseline['objective_std']:.3f}"
                    )
                    if baseline
                    else "not run",
                    PUBLISHED.get(str(metrics["objective"]), "n/a"),
                ]
            )
        )
    return "\n".join(lines)


def quality_table(runs: list[tuple[str, dict[str, Any] | None]]) -> str:
    header = [
        "run",
        "n",
        "validity",
        "uniqueness",
        "novelty",
        "mean similarity to start",
        "mean improvement",
        "fraction improved",
        "meets similarity threshold",
    ]
    lines = [_row(header), _row(["---"] * len(header))]
    for name, metrics in runs:
        if metrics is None:
            continue
        agent = metrics["agent"]
        constraint = agent.get("constraint_satisfied")
        lines.append(
            _row(
                [
                    name,
                    str(agent["n"]),
                    f"{agent['validity']:.3f}",
                    f"{agent['uniqueness']:.3f}",
                    f"{agent['novelty']:.3f}",
                    f"{agent['similarity_mean']:.3f}",
                    f"{agent['improvement_mean']:+.3f} +/- {agent['improvement_std']:.3f}",
                    f"{agent['fraction_improved']:.0%}",
                    f"{constraint:.3f}" if constraint is not None else "n/a",
                ]
            )
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "runs")
    if not root.exists():
        print(f"error: no such directory {root}", file=sys.stderr)
        return 2
    runs = load_runs(root)
    if not runs:
        print(f"error: no run directories under {root}", file=sys.stderr)
        return 2

    print("# Measured results\n")
    print(
        "Every number below was measured by `droptimus evaluate` on the "
        "checkpoint named in the first column. The published MolDQN column is "
        "quoted from Zhou et al. (2019) for comparison and was not reproduced "
        "by this code.\n"
    )
    print("## Objective values\n")
    print(objective_table(runs))
    print("\n## Generation quality\n")
    print(quality_table(runs))
    print(
        "\nValidity is 1.000 by construction: the environment only ever "
        "proposes molecules that pass RDKit sanitization, so this column "
        "confirms the invariant rather than measuring model quality."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
