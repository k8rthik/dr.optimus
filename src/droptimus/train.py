"""Training loop.

One episode per iteration: roll out the current epsilon-greedy policy, push the
transitions into replay, and take gradient steps. Epsilon decays linearly over a
configurable fraction of the run. Everything needed to reproduce the run --
configs, seed, per-episode history -- is written next to the checkpoint.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from droptimus.agent.dqn import DoubleDQNAgent
from droptimus.config import RunConfig
from droptimus.env.mdp import MoleculeEnv
from droptimus.errors import ConfigError
from droptimus.objectives import make_objective, objective_needs_reference
from droptimus.rollout import Episode, agent_policy, rollout

LOGGER = logging.getLogger("droptimus.train")


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Per-episode record written to the run history."""

    index: int
    start_smiles: str
    best_smiles: str
    best_objective: float
    final_objective: float
    total_reward: float
    epsilon: float
    loss: float | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Everything a training run produced."""

    config: RunConfig
    history: tuple[EpisodeSummary, ...]
    checkpoint_path: Path | None
    best_smiles: str
    best_objective: float
    wall_seconds: float

    @property
    def episodes(self) -> int:
        return len(self.history)


def epsilon_at(episode: int, config: RunConfig) -> float:
    """Return the exploration rate for ``episode`` under a linear schedule."""
    train = config.train
    if not 0.0 < train.epsilon_decay_fraction <= 1.0:
        raise ConfigError(
            "epsilon_decay_fraction must lie in (0, 1], got "
            f"{train.epsilon_decay_fraction}."
        )
    decay_episodes = max(int(train.episodes * train.epsilon_decay_fraction), 1)
    progress = min(episode / decay_episodes, 1.0)
    return train.epsilon_start + progress * (train.epsilon_end - train.epsilon_start)


def build_objective(config: RunConfig, start_smiles: str):
    """Build the run's objective, supplying the start molecule where needed."""
    kwargs = {key: value for key, value in config.objective_kwargs}
    if objective_needs_reference(config.objective):
        kwargs.setdefault("reference", start_smiles)
    return make_objective(config.objective, **kwargs)


def make_env(config: RunConfig, start_smiles: str, rng: random.Random) -> MoleculeEnv:
    """Construct an environment for one start molecule."""
    return MoleculeEnv(
        objective=build_objective(config, start_smiles),
        config=config.env,
        start_smiles=start_smiles,
        rng=rng,
    )


def train(
    config: RunConfig,
    start_molecules: Sequence[str] | None = None,
    out_dir: Path | str | None = None,
    agent: DoubleDQNAgent | None = None,
) -> TrainResult:
    """Train an agent and return its history.

    Args:
        config: the full run specification.
        start_molecules: start molecules to cycle through. Defaults to
            ``config.start_smiles`` alone.
        out_dir: directory for the checkpoint and history. ``None`` writes
            nothing (used by tests).
        agent: an existing agent to continue training.

    Raises:
        ConfigError: if the episode count is not positive.
    """
    if config.train.episodes < 1:
        raise ConfigError(f"episodes must be >= 1, got {config.train.episodes}.")

    starts = tuple(start_molecules) if start_molecules else (config.start_smiles,)
    rng = random.Random(config.train.seed)
    learner = agent or DoubleDQNAgent(
        agent_config=config.agent,
        env_config=config.env,
        device=config.train.device,
        seed=config.train.seed,
    )

    history: list[EpisodeSummary] = []
    best_smiles, best_objective = starts[0], float("-inf")
    started_at = time.perf_counter()

    for index in range(config.train.episodes):
        episode_started = time.perf_counter()
        start_smiles = starts[index % len(starts)]
        env = make_env(config, start_smiles, rng)
        epsilon = (
            1.0
            if index < config.train.warmup_episodes
            else epsilon_at(index - config.train.warmup_episodes, config)
        )
        episode = rollout(env, agent_policy(learner, epsilon), record_transitions=True)

        loss = None
        for transition in episode.transitions:
            learner.observe(transition)
            for _ in range(config.agent.updates_per_step):
                step_loss = learner.update()
                if step_loss is not None:
                    loss = step_loss

        if episode.best_objective > best_objective:
            best_smiles, best_objective = episode.best_smiles, episode.best_objective

        history.append(
            EpisodeSummary(
                index=index,
                start_smiles=start_smiles,
                best_smiles=episode.best_smiles,
                best_objective=episode.best_objective,
                final_objective=episode.final_objective,
                total_reward=episode.total_reward,
                epsilon=epsilon,
                loss=loss,
                elapsed_seconds=time.perf_counter() - episode_started,
            )
        )
        _log_progress(history, config, best_objective)

    wall_seconds = time.perf_counter() - started_at
    checkpoint_path = None
    if out_dir is not None:
        checkpoint_path = _persist(config, learner, history, out_dir, wall_seconds)

    return TrainResult(
        config=config,
        history=tuple(history),
        checkpoint_path=checkpoint_path,
        best_smiles=best_smiles,
        best_objective=best_objective,
        wall_seconds=wall_seconds,
    )


def _log_progress(
    history: Sequence[EpisodeSummary], config: RunConfig, best_objective: float
) -> None:
    index = len(history) - 1
    every = max(config.train.log_every, 1)
    if (index + 1) % every and index != 0:
        return
    window = history[-every:]
    mean_best = sum(item.best_objective for item in window) / len(window)
    losses = [item.loss for item in window if item.loss is not None]
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    rate = sum(item.elapsed_seconds for item in window) / len(window)
    LOGGER.info(
        "episode %5d/%d  eps %.3f  objective(last %d) %+.4f  best %+.4f  "
        "loss %.4f  %.2fs/ep",
        index + 1,
        config.train.episodes,
        window[-1].epsilon,
        len(window),
        mean_best,
        best_objective,
        mean_loss,
        rate,
    )


def _persist(
    config: RunConfig,
    learner: DoubleDQNAgent,
    history: Sequence[EpisodeSummary],
    out_dir: Path | str,
    wall_seconds: float,
) -> Path:
    """Write the checkpoint and the run history. Returns the checkpoint path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = learner.save(
        directory / "checkpoint.pt",
        metadata={
            "objective": config.objective,
            "objective_kwargs": list(config.objective_kwargs),
            "start_smiles": config.start_smiles,
            "episodes": config.train.episodes,
            "seed": config.train.seed,
        },
    )
    (directory / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str) + "\n"
    )
    (directory / "history.json").write_text(
        json.dumps(
            {
                "wall_seconds": wall_seconds,
                "episodes": [asdict(item) for item in history],
            },
            indent=2,
        )
        + "\n"
    )
    return checkpoint


def collect_episodes(
    config: RunConfig,
    starts: Sequence[str],
    learner: DoubleDQNAgent | None,
    epsilon: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[Episode, ...]:
    """Roll out one greedy (or random, if ``learner`` is None) episode per start.

    This is the generation step used by ``evaluate`` and by the random baseline.
    """
    from droptimus.rollout import random_policy

    generator = rng or random.Random(config.train.seed)
    episodes = []
    for start_smiles in starts:
        env = make_env(config, start_smiles, generator)
        policy = (
            agent_policy(learner, epsilon)
            if learner is not None
            else random_policy(generator)
        )
        episodes.append(rollout(env, policy, record_transitions=False))
    return tuple(episodes)
