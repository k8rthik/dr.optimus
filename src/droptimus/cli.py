"""Command-line interface.

    droptimus download                      fetch the ZINC250k benchmark set
    droptimus train --objective qed ...     train an agent
    droptimus optimize "<SMILES>" ...       optimize one molecule
    droptimus evaluate --checkpoint ...     measure a trained agent (and the
                                            random-edit baseline)

Every user-supplied molecule is validated before any work starts, so a typo in a
SMILES string produces a one-line error rather than a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from droptimus import __version__
from droptimus.agent.dqn import DoubleDQNAgent
from droptimus.chem.molecule import canonical_smiles
from droptimus.config import (
    DEFAULT_SIMILARITY_DELTA,
    AgentConfig,
    EnvConfig,
    RunConfig,
    TrainConfig,
    field_default,
)
from droptimus.datasets import (
    DEFAULT_DATA_DIR,
    ZINC800_SIZE,
    download_zinc250k,
    load_zinc_smiles,
)
from droptimus.errors import DrOptimusError
from droptimus.evaluate import (
    comparison_rows,
    compute_metrics,
    constrained_comparison,
    format_metrics,
    generated_from_episodes,
    published_random_walk,
)
from droptimus.objectives import available_objectives
from droptimus.start_sets import (
    START_SET_CHOICES,
    novelty_reference,
    parse_atom_types,
    resolve_start_set,
)
from droptimus.train import collect_episodes, train

LOGGER = logging.getLogger("droptimus")

#: Exploration rate used when sampling the evaluation distribution. The greedy
#: policy is deterministic in a deterministic environment, so repeated greedy
#: episodes from one start molecule return the same molecule every time; a small
#: epsilon is what makes "100 generated molecules" a distribution rather than
#: 100 copies. The deterministic greedy rollout is reported alongside it.
EVAL_EPSILON = 0.1


# --- argument parsing ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="droptimus",
        description=(
            "MolDQN-style reinforcement learning for molecule optimization on "
            "open cheminformatics benchmarks."
        ),
    )
    parser.add_argument("--version", action="version", version=f"droptimus {__version__}")
    parser.add_argument(
        "--quiet", action="store_true", help="only print warnings and errors"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="fetch the ZINC250k benchmark set")
    download.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    download.add_argument("--force", action="store_true", help="re-download if present")

    train_parser = subparsers.add_parser("train", help="train an agent")
    _add_run_arguments(train_parser)
    train_parser.add_argument("--episodes", type=int, default=800)
    train_parser.add_argument("--warmup-episodes", type=int, default=10)
    train_parser.add_argument("--out", default=None, help="directory for the checkpoint")
    train_parser.add_argument(
        "--start-set",
        choices=START_SET_CHOICES,
        default="single",
        help="which molecules to start episodes from (default: --start only)",
    )
    train_parser.add_argument(
        "--start-set-size",
        type=int,
        default=ZINC800_SIZE,
        help="how many molecules to take from the chosen start set",
    )
    train_parser.add_argument("--log-every", type=int, default=25)

    optimize = subparsers.add_parser("optimize", help="optimize one molecule")
    optimize.add_argument("smiles", help="the start molecule, in SMILES")
    optimize.add_argument("--checkpoint", required=True)
    optimize.add_argument(
        "--attempts", type=int, default=10, help="episodes to run; the best is reported"
    )
    optimize.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="exploration rate; 0 makes every attempt identical",
    )
    optimize.add_argument("--device", default="auto")
    optimize.add_argument("--seed", type=int, default=0)
    optimize.add_argument("--show-trajectory", action="store_true")
    optimize.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_SIMILARITY_DELTA,
        help="similarity threshold, for constrained checkpoints",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="measure a trained agent against the random-edit baseline"
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument(
        "--start-set", choices=START_SET_CHOICES, default="fixture"
    )
    evaluate.add_argument("--start", default=None, help="start molecule for 'single'")
    evaluate.add_argument("--episodes", type=int, default=100)
    evaluate.add_argument(
        "--epsilon",
        type=float,
        default=EVAL_EPSILON,
        help=(
            "exploration rate used to sample the reported distribution. The "
            "greedy policy is deterministic, so from a single start molecule "
            f"epsilon 0 yields the same molecule every episode; the default "
            f"{EVAL_EPSILON} samples a real distribution. The deterministic "
            "greedy rollout is always reported separately."
        ),
    )
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    evaluate.add_argument(
        "--no-baseline", action="store_true", help="skip the random-edit baseline"
    )
    evaluate.add_argument(
        "--novelty-reference-size",
        type=int,
        default=50_000,
        help="how many ZINC molecules to use as the novelty reference",
    )
    evaluate.add_argument("--out", default=None, help="write metrics as JSON here")
    evaluate.add_argument(
        "--delta",
        type=float,
        default=None,
        help="report the fraction of molecules with similarity >= delta",
    )
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that define an environment/agent/objective."""
    parser.add_argument(
        "--objective",
        default="qed",
        help=f"one of: {', '.join(available_objectives())}",
    )
    parser.add_argument("--start", default="C", help="start molecule, in SMILES")
    parser.add_argument(
        "--base-objective",
        default="penalized_logp",
        help="objective to constrain, for --objective constrained",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_SIMILARITY_DELTA,
        help="similarity threshold, for --objective constrained",
    )
    parser.add_argument("--encoder", choices=("fingerprint", "gnn"), default="fingerprint")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-atoms", type=int, default=38)
    parser.add_argument(
        "--atom-types",
        default="C,N,O",
        help="comma-separated elements the agent may add",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=field_default(EnvConfig, "discount"),
        help="reward discount; pass 0.9 for the paper's value",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("terminal", "dense", "paper"),
        default="terminal",
        help=(
            "terminal: objective paid once at the horizon; dense: per-step "
            "objective delta; paper: objective * discount^(steps left) every "
            "step, as in the reference implementation"
        ),
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="cap on candidates scored per step (default: no cap)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=field_default(AgentConfig, "batch_size")
    )
    parser.add_argument(
        "--bootstrap-actions",
        type=int,
        default=field_default(AgentConfig, "bootstrap_actions"),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=field_default(AgentConfig, "learning_rate"),
    )
    parser.add_argument("--target-sync-steps", type=int, default=500)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)


# --- config assembly -------------------------------------------------------


def run_config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig from parsed ``train`` arguments."""
    objective = args.objective.strip().lower()
    if objective not in available_objectives():
        raise DrOptimusError(
            f"Unknown objective {args.objective!r}. Available: "
            f"{', '.join(available_objectives())}."
        )
    start_smiles = canonical_smiles(args.start)

    objective_kwargs: tuple[tuple[str, object], ...] = ()
    if objective == "constrained":
        objective_kwargs = (("base", args.base_objective), ("delta", args.delta))
    elif objective == "similarity":
        objective_kwargs = ()

    return RunConfig(
        objective=objective,
        start_smiles=start_smiles,
        env=EnvConfig(
            atom_types=parse_atom_types(args.atom_types),
            max_steps=args.max_steps,
            max_atoms=args.max_atoms,
            max_actions=args.max_actions,
            discount=args.discount,
            reward_mode=args.reward_mode,
        ),
        agent=AgentConfig(
            encoder=args.encoder,
            batch_size=args.batch_size,
            bootstrap_actions=args.bootstrap_actions,
            learning_rate=args.learning_rate,
            target_sync_steps=args.target_sync_steps,
            replay_capacity=args.replay_capacity,
        ),
        train=TrainConfig(
            episodes=args.episodes,
            warmup_episodes=args.warmup_episodes,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
        ),
        objective_kwargs=objective_kwargs,  # type: ignore[arg-type]
    )


# --- commands --------------------------------------------------------------


def command_download(args: argparse.Namespace) -> int:
    path = download_zinc250k(args.data_dir, force=args.force)
    molecules = load_zinc_smiles(args.data_dir)
    print(f"ZINC250k ready at {path} ({len(molecules)} valid molecules)")
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = run_config_from_args(args)
    starts = resolve_start_set(
        args.start_set, args.start_set_size, args.start, seed=args.seed
    )
    out_dir = Path(args.out) if args.out else None
    print(
        f"training {config.objective} | encoder {config.agent.encoder} | "
        f"{config.train.episodes} episodes x {config.env.max_steps} steps | "
        f"{len(starts)} start molecule(s) | seed {config.train.seed}"
    )
    result = train(config, start_molecules=starts, out_dir=out_dir)
    print(
        f"done in {result.wall_seconds / 60:.1f} min | "
        f"best objective {result.best_objective:+.4f} | {result.best_smiles}"
    )
    if result.checkpoint_path:
        print(f"checkpoint: {result.checkpoint_path}")
    return 0


def _config_for_checkpoint(
    agent: DoubleDQNAgent,
    metadata: dict,
    start_smiles: str,
    delta: float | None,
    seed: int,
    device: str,
) -> RunConfig:
    """Rebuild the run configuration a checkpoint was trained under."""
    objective = str(metadata.get("objective", "qed"))
    kwargs = dict(metadata.get("objective_kwargs") or [])
    if delta is not None and objective == "constrained":
        kwargs["delta"] = delta
    config = RunConfig(
        objective=objective,
        start_smiles=start_smiles,
        env=agent.env_config,
        agent=agent.agent_config,
        train=TrainConfig(seed=seed, device=device),
        objective_kwargs=tuple(kwargs.items()),  # type: ignore[arg-type]
    )
    return config


def command_optimize(args: argparse.Namespace) -> int:
    start_smiles = canonical_smiles(args.smiles)
    agent, metadata = DoubleDQNAgent.load(args.checkpoint, device=args.device)
    config = _config_for_checkpoint(
        agent, metadata, start_smiles, args.delta, args.seed, args.device
    )
    if args.attempts < 1:
        raise DrOptimusError(f"--attempts must be >= 1, got {args.attempts}.")

    rng = random.Random(args.seed)
    episodes = collect_episodes(
        config, [start_smiles] * args.attempts, agent, epsilon=args.epsilon, rng=rng
    )
    best = max(episodes, key=lambda episode: episode.best_objective)

    objective_name = config.objective
    print(f"start      {start_smiles}")
    print(f"objective  {objective_name} = {best.start_objective:+.4f}")
    print(f"best       {best.best_smiles}")
    print(f"objective  {objective_name} = {best.best_objective:+.4f} "
          f"({best.improvement:+.4f})")
    from droptimus.chem.molecule import tanimoto_similarity

    print(f"similarity {tanimoto_similarity(start_smiles, best.best_smiles):.3f} to start")
    print(f"attempts   {args.attempts} (epsilon {args.epsilon})")
    if args.show_trajectory:
        print("trajectory:")
        for step, smiles in enumerate(best.trajectory):
            print(f"  {step:2d}  {smiles}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    agent, metadata = DoubleDQNAgent.load(args.checkpoint, device=args.device)
    starts = resolve_start_set(
        args.start_set,
        args.episodes,
        args.start or str(metadata.get("start_smiles", "C")),
        data_dir=args.data_dir,
        seed=args.seed,
    )
    starts = starts[: args.episodes] if args.start_set != "single" else starts * args.episodes
    config = _config_for_checkpoint(
        agent, metadata, starts[0], args.delta, args.seed, args.device
    )

    reference = novelty_reference(
        args.novelty_reference_size, args.data_dir
    )
    delta = args.delta
    if delta is None and config.objective == "constrained":
        delta = dict(config.objective_kwargs).get("delta")  # type: ignore[arg-type]

    print(f"evaluating {args.checkpoint}")
    print(
        f"objective {config.objective} | {len(starts)} episodes x "
        f"{config.env.max_steps} steps | start set {args.start_set} | "
        f"epsilon {args.epsilon}"
    )

    # The deterministic greedy rollout, reported on its own because it is the
    # single molecule the learned policy actually commits to.
    greedy = collect_episodes(
        config, [starts[0]], agent, epsilon=0.0, rng=random.Random(args.seed)
    )[0]
    print()
    print(
        f"greedy rollout (epsilon 0) from {greedy.start_smiles}\n"
        f"  {greedy.best_smiles}\n"
        f"  objective {greedy.best_objective:+.4f} "
        f"(start {greedy.start_objective:+.4f}, "
        f"improvement {greedy.improvement:+.4f})"
    )

    agent_episodes = collect_episodes(
        config, starts, agent, epsilon=args.epsilon, rng=random.Random(args.seed)
    )

    def measure(episodes, use_best: bool):
        return compute_metrics(
            generated_from_episodes(episodes, use_best=use_best),
            reference=reference,
            similarity_delta=delta,
        )

    # Two protocols, both reported. "best visited" is the generous reading used
    # by much of this literature; "final" is the molecule the episode actually
    # ended on, which is what Zhou et al. Table 1 reports (the top three of the
    # last 100 terminal states), so it is the one the published comparison uses.
    agent_metrics = measure(agent_episodes, use_best=True)
    agent_final = measure(agent_episodes, use_best=False)
    print()
    print(format_metrics(agent_metrics, "trained agent, best molecule visited"))
    print()
    print(format_metrics(agent_final, "trained agent, final molecule of episode"))

    payload: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "objective": config.objective,
        "start_set": args.start_set,
        "episodes": len(starts),
        "max_steps": config.env.max_steps,
        "epsilon": args.epsilon,
        "greedy": {
            "start_smiles": greedy.start_smiles,
            "smiles": greedy.best_smiles,
            "objective": greedy.best_objective,
            "final_smiles": greedy.final_smiles,
            "final_objective": greedy.final_objective,
            "start_objective": greedy.start_objective,
        },
        "agent": agent_metrics.as_dict(),
        "agent_final": agent_final.as_dict(),
    }

    if not args.no_baseline:
        baseline_episodes = collect_episodes(
            config, starts, None, rng=random.Random(args.seed + 1)
        )
        baseline_metrics = measure(baseline_episodes, use_best=True)
        baseline_final = measure(baseline_episodes, use_best=False)
        print()
        print(format_metrics(baseline_metrics, "random-edit baseline, best visited"))
        payload["random_baseline"] = baseline_metrics.as_dict()
        payload["random_baseline_final"] = baseline_final.as_dict()

    print()
    print(_published_comparison(config.objective, agent_final))
    if delta is not None:
        print()
        print(constrained_comparison(float(delta), agent_final))

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nmetrics written to {destination}")
    return 0


def _published_comparison(objective: str, metrics) -> str:
    """Compare final-episode molecules against Zhou et al. (2019) Table 1.

    MolDQN-naive is a single Q-network with epsilon-greedy exploration, which is
    what this code implements; MolDQN-bootstrap adds an ensemble of Q-heads for
    exploration, which it does not. Naive is the fair comparison.
    """
    rows = comparison_rows(objective, metrics)
    width = max(len(row[0]) for row in rows)
    lines = [
        "final-episode molecules vs published MolDQN "
        "(Zhou et al. 2019, Sci Rep 9:10752, Table 1)",
        f"  {'metric'.ljust(width)}  {'this run':>22}  {'MolDQN-naive':>14}"
        f"  {'MolDQN-bootstrap':>17}",
    ]
    for label, measured, naive, bootstrap in rows:
        lines.append(
            f"  {label.ljust(width)}  {measured:>22}  {naive:>14}  {bootstrap:>17}"
        )
    random_walk = published_random_walk(objective)
    if random_walk is not None:
        lines.append(
            f"  (the paper's own random-action baseline reached "
            f"{random_walk:.3f} on this objective)"
        )
    return "\n".join(lines)


COMMANDS = {
    "download": command_download,
    "train": command_train,
    "optimize": command_optimize,
    "evaluate": command_evaluate,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces the choices
        parser.error(f"unknown command {args.command!r}")
    try:
        return handler(args)
    except DrOptimusError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
