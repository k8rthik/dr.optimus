#!/usr/bin/env python
"""Turn the metrics a benchmark sweep produced into the README's results tables.

Every number printed here is read from a `metrics.json` written by `droptimus
evaluate` or a `history.json` written by `droptimus train`. Nothing is computed,
estimated or carried over from the paper except the clearly labelled published
column. A run whose metrics file is missing is listed as missing rather than
dropped, so a failed run cannot quietly vanish from the table.

Usage:  python scripts/make_report.py runs > runs/RESULTS.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Published values live in droptimus.evaluate, transcribed from the paper and
# guarded by a test, so this script has a single source for them.
from droptimus.evaluate import PUBLISHED_MOLDQN

#: Runs whose names start with these are exploratory, not headline results.
PROBE_PREFIXES = ("ablate-", "probe-")


class Run:
    """One run directory's measured artifacts."""

    def __init__(self, directory: Path) -> None:
        self.name = directory.name
        self.metrics: dict[str, Any] | None = _read_json(directory / "metrics.json")
        self.history: dict[str, Any] | None = _read_json(directory / "history.json")
        self.config: dict[str, Any] | None = _read_json(directory / "run_config.json")

    @property
    def is_probe(self) -> bool:
        return self.name.startswith(PROBE_PREFIXES)

    @property
    def episodes_trained(self) -> int | None:
        if not self.history:
            return None
        return len(self.history.get("episodes", []))

    @property
    def wall_minutes(self) -> float | None:
        if not self.history:
            return None
        return float(self.history["wall_seconds"]) / 60.0

    @property
    def steps_per_second(self) -> float | None:
        """Environment steps per second over the whole training run."""
        if not self.history or not self.config:
            return None
        episodes = self.episodes_trained or 0
        max_steps = int(self.config["env"]["max_steps"])
        seconds = float(self.history["wall_seconds"])
        if seconds <= 0 or episodes == 0:
            return None
        return episodes * max_steps / seconds


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join([_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)])


def objective_table(runs: list[Run]) -> str:
    header = [
        "run",
        "objective",
        "episodes",
        "greedy",
        "best of 100",
        "mean +/- sd",
        "random baseline best",
        "random baseline mean",
        "MolDQN-naive best",
        "MolDQN-bootstrap best",
    ]
    rows = []
    for run in runs:
        if run.metrics is None:
            rows.append([run.name, *["(no metrics.json)"] * (len(header) - 1)])
            continue
        agent = run.metrics["agent"]
        baseline = run.metrics.get("random_baseline")
        greedy = run.metrics.get("greedy")
        objective = str(run.metrics["objective"])
        published = PUBLISHED_MOLDQN.get(objective, {})
        naive = published.get("naive", ())
        bootstrap = published.get("bootstrap", ())
        rows.append(
            [
                run.name,
                objective,
                str(run.episodes_trained or "?"),
                f"{greedy['objective']:.3f}" if greedy else "n/a",
                f"{agent['objective_max']:.3f}",
                f"{agent['objective_mean']:.3f} +/- {agent['objective_std']:.3f}",
                f"{baseline['objective_max']:.3f}" if baseline else "not run",
                f"{baseline['objective_mean']:.3f} +/- {baseline['objective_std']:.3f}"
                if baseline
                else "not run",
                f"{naive[0]:.3f}" if naive else "n/a",
                f"{bootstrap[0]:.3f}" if bootstrap else "n/a",
            ]
        )
    return _table(header, rows)


def final_episode_table(runs: list[Run]) -> str:
    """The paper's protocol: the molecule each episode actually ended on."""
    header = [
        "run",
        "objective",
        "best of n (final)",
        "mean +/- sd (final)",
        "baseline best (final)",
        "MolDQN-naive best",
        "MolDQN-bootstrap best",
    ]
    rows = []
    for run in runs:
        if run.metrics is None or "agent_final" not in run.metrics:
            continue
        final = run.metrics["agent_final"]
        baseline = run.metrics.get("random_baseline_final")
        objective = str(run.metrics["objective"])
        published = PUBLISHED_MOLDQN.get(objective, {})
        naive = published.get("naive", ())
        bootstrap = published.get("bootstrap", ())
        rows.append(
            [
                run.name,
                objective,
                f"{final['objective_max']:.3f}",
                f"{final['objective_mean']:.3f} +/- {final['objective_std']:.3f}",
                f"{baseline['objective_max']:.3f}" if baseline else "not run",
                f"{naive[0]:.3f}" if naive else "n/a",
                f"{bootstrap[0]:.3f}" if bootstrap else "n/a",
            ]
        )
    return _table(header, rows)


def quality_table(runs: list[Run]) -> str:
    header = [
        "run",
        "n",
        "validity",
        "uniqueness",
        "novelty",
        "mean similarity to start",
        "mean improvement",
        "improved",
        "meets similarity threshold",
    ]
    rows = []
    for run in runs:
        if run.metrics is None:
            continue
        agent = run.metrics["agent"]
        constraint = agent.get("constraint_satisfied")
        rows.append(
            [
                run.name,
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
    return _table(header, rows)


def throughput_table(runs: list[Run]) -> str:
    header = ["run", "episodes", "steps/s", "s/episode", "wall-clock (min)"]
    rows = []
    for run in runs:
        if run.history is None:
            continue
        episodes = run.episodes_trained or 0
        minutes = run.wall_minutes
        steps = run.steps_per_second
        rows.append(
            [
                run.name,
                str(episodes),
                f"{steps:.1f}" if steps else "n/a",
                f"{minutes * 60 / episodes:.2f}" if minutes and episodes else "n/a",
                f"{minutes:.1f}" if minutes else "n/a",
            ]
        )
    return _table(header, rows)


def probe_table(runs: list[Run]) -> str:
    header = ["configuration", "episodes", "greedy", "best of n", "mean +/- sd"]
    rows = []
    for run in runs:
        if run.metrics is None:
            rows.append([run.name, *["(no metrics.json)"] * (len(header) - 1)])
            continue
        agent = run.metrics["agent"]
        greedy = run.metrics.get("greedy")
        rows.append(
            [
                run.name,
                str(run.episodes_trained or "?"),
                f"{greedy['objective']:.3f}" if greedy else "n/a",
                f"{agent['objective_max']:.3f}",
                f"{agent['objective_mean']:.3f} +/- {agent['objective_std']:.3f}",
            ]
        )
    # One shared random-edit baseline row, taken from whichever probe recorded it.
    for run in runs:
        baseline = (run.metrics or {}).get("random_baseline")
        if baseline:
            rows.append(
                [
                    "random-edit baseline",
                    "n/a",
                    "n/a",
                    f"{baseline['objective_max']:.3f}",
                    f"{baseline['objective_mean']:.3f} +/- {baseline['objective_std']:.3f}",
                ]
            )
            break
    return _table(header, rows)


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "runs")
    if not root.exists():
        print(f"error: no such directory {root}", file=sys.stderr)
        return 2
    runs = [Run(p) for p in sorted(root.iterdir()) if p.is_dir()]
    if not runs:
        print(f"error: no run directories under {root}", file=sys.stderr)
        return 2
    headline = [r for r in runs if not r.is_probe]
    probes = [r for r in runs if r.is_probe]

    print("### Objective values\n")
    print(
        "`greedy` is the single molecule the deterministic policy produces. "
        "`best of 100` and `mean` come from 100 epsilon-greedy episodes "
        "(epsilon 0.1), because a deterministic policy from one start molecule "
        "returns the same molecule every time. The random-edit baseline takes "
        "the same number of episodes from the same start molecules, choosing "
        "uniformly among valid edits.\n"
    )
    print(objective_table(headline))
    print(
        "\n### Final-episode molecules (the paper's protocol)\n"
    )
    print(
        "Zhou et al. Table 1 reports the top three scores among the last 100 "
        "*terminal* states, so this is the table to compare against the "
        "published column. MolDQN-naive is a single Q-network with "
        "epsilon-greedy exploration, which is what this code implements; "
        "MolDQN-bootstrap adds an ensemble of Q-heads, which it does not.\n"
    )
    print(final_episode_table(headline))
    print("\n### Generation quality\n")
    print(quality_table(headline))
    print(
        "\nValidity is 1.000 by construction, not by training: the environment "
        "only proposes candidates that pass RDKit sanitization. The column "
        "confirms the invariant holds.\n"
    )
    print("### Throughput\n")
    print(throughput_table(headline))
    if probes:
        print("\n### Configuration probes\n")
        print(probe_table(probes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
