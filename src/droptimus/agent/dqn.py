"""Double DQN over a state-dependent molecular action space.

Because actions *are* candidate molecules, the Q-network scores every candidate
and the agent takes the argmax. The bootstrap target is

    y = r + (1 - done) * gamma * Q_target(argmax_{a'} Q_online(a'))

i.e. Double DQN (van Hasselt et al. 2016): the online network picks the
successor, the target network values it. Set ``AgentConfig.double_dqn = False``
for the vanilla max-over-target-network variant.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from droptimus.agent.featurizers import Featurizer, build_featurizer
from droptimus.agent.networks import QNetwork, build_network
from droptimus.agent.replay import ReplayBuffer, Transition
from droptimus.agent.torch_utils import resolve_device, set_global_seed
from droptimus.config import AgentConfig, EnvConfig
from droptimus.errors import CheckpointError, ConfigError

#: Checkpoint format version. Bumped if the payload layout changes.
CHECKPOINT_VERSION = 1


class DoubleDQNAgent:
    """A molecule-editing double-DQN agent."""

    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        env_config: EnvConfig | None = None,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        self.agent_config = agent_config or AgentConfig()
        self.env_config = env_config or EnvConfig()
        self._validate()
        if seed is not None:
            set_global_seed(seed)
        self.device = resolve_device(device)
        self.rng = random.Random(seed)

        self.featurizer: Featurizer = build_featurizer(
            self.agent_config.encoder,
            max_steps=self.env_config.max_steps,
            max_atoms=self.env_config.max_atoms,
        )
        self.online = self._new_network().to(self.device)
        self.target = self._new_network().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.agent_config.learning_rate
        )
        self.replay = ReplayBuffer(
            capacity=self.agent_config.replay_capacity, rng=self.rng
        )
        self.steps_done = 0
        self.updates_done = 0

    def _validate(self) -> None:
        config = self.agent_config
        if config.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {config.batch_size}.")
        if config.learning_rate <= 0:
            raise ConfigError(
                f"learning_rate must be > 0, got {config.learning_rate}."
            )
        if config.bootstrap_actions < 1:
            raise ConfigError(
                f"bootstrap_actions must be >= 1, got {config.bootstrap_actions}."
            )
        if config.target_sync_steps < 1:
            raise ConfigError(
                f"target_sync_steps must be >= 1, got {config.target_sync_steps}."
            )
        if not config.hidden_sizes:
            raise ConfigError("hidden_sizes must contain at least one layer width.")

    def _new_network(self) -> QNetwork:
        return build_network(
            encoder=self.agent_config.encoder,
            featurizer=self.featurizer,
            hidden_sizes=self.agent_config.hidden_sizes,
            gnn_hidden=self.agent_config.gnn_hidden,
            gnn_layers=self.agent_config.gnn_layers,
        )

    # --- scoring and action selection --------------------------------------

    def _score(
        self,
        network: QNetwork,
        smiles: Sequence[str],
        steps_remaining: Sequence[int],
    ) -> torch.Tensor:
        batch = self.featurizer.batch(smiles, steps_remaining, self.device)
        return network(batch)

    @torch.no_grad()
    def q_values(self, candidates: Sequence[str], steps_remaining: int) -> np.ndarray:
        """Return the online network's Q value for each candidate molecule."""
        if not candidates:
            return np.zeros((0,), dtype=np.float32)
        self.online.eval()
        horizons = [steps_remaining] * len(candidates)
        scores = self._score(self.online, candidates, horizons)
        self.online.train()
        return scores.detach().float().cpu().numpy()

    def select_action(
        self,
        candidates: Sequence[str],
        steps_remaining: int,
        epsilon: float = 0.0,
    ) -> str:
        """Pick a candidate epsilon-greedily with respect to the online network.

        Raises:
            ValueError: if ``candidates`` is empty or ``epsilon`` is out of range.
        """
        ordered = tuple(candidates)
        if not ordered:
            raise ValueError("Cannot select an action from an empty candidate set.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}.")
        if self.rng.random() < epsilon:
            return self.rng.choice(ordered)
        scores = self.q_values(ordered, steps_remaining)
        return ordered[int(np.argmax(scores))]

    # --- learning ----------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        """Store a transition and advance the step counter."""
        self.replay.add(self._truncate_candidates(transition))
        self.steps_done += 1
        if self.steps_done % self.agent_config.target_sync_steps == 0:
            self.sync_target()

    def _truncate_candidates(self, transition: Transition) -> Transition:
        """Subsample the successor set the bootstrap target will max over."""
        cap = self.agent_config.bootstrap_actions
        candidates = transition.next_candidates
        if len(candidates) <= cap:
            return transition
        sampled = tuple(self.rng.sample(list(candidates), cap))
        return Transition(
            action_smiles=transition.action_smiles,
            action_steps_remaining=transition.action_steps_remaining,
            reward=transition.reward,
            done=transition.done,
            next_candidates=sampled,
            next_steps_remaining=transition.next_steps_remaining,
        )

    def sync_target(self) -> None:
        """Hard-copy the online weights into the target network."""
        self.target.load_state_dict(self.online.state_dict())

    def update(self) -> float | None:
        """Run one gradient step. Returns the loss, or None if the buffer is cold."""
        batch_size = self.agent_config.batch_size
        if not self.replay.can_sample(batch_size):
            return None
        batch = self.replay.sample(batch_size)
        targets = self._bootstrap_targets(batch)

        predicted = self._score(
            self.online,
            [t.action_smiles for t in batch],
            [t.action_steps_remaining for t in batch],
        )
        loss = nn.functional.smooth_l1_loss(predicted, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.agent_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self.online.parameters(), self.agent_config.grad_clip
            )
        self.optimizer.step()
        self.updates_done += 1
        return float(loss.detach().cpu())

    @torch.no_grad()
    def _bootstrap_targets(self, batch: Sequence[Transition]) -> torch.Tensor:
        """Compute the Double-DQN regression targets for a batch."""
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        continues = torch.tensor(
            [0.0 if t.done else 1.0 for t in batch],
            dtype=torch.float32,
            device=self.device,
        )

        flat_smiles: list[str] = []
        flat_horizons: list[int] = []
        rows: list[int] = []
        columns: list[int] = []
        for row, transition in enumerate(batch):
            for column, candidate in enumerate(transition.next_candidates):
                flat_smiles.append(candidate)
                flat_horizons.append(transition.next_steps_remaining)
                rows.append(row)
                columns.append(column)

        next_values = torch.zeros_like(rewards)
        if flat_smiles:
            width = max(columns) + 1
            online_scores = self._score(self.online, flat_smiles, flat_horizons)
            target_scores = self._score(self.target, flat_smiles, flat_horizons)
            row_index = torch.tensor(rows, dtype=torch.long, device=self.device)
            column_index = torch.tensor(columns, dtype=torch.long, device=self.device)

            padded_online = torch.full(
                (len(batch), width),
                torch.finfo(torch.float32).min,
                dtype=torch.float32,
                device=self.device,
            )
            padded_online[row_index, column_index] = online_scores.float()
            padded_target = torch.zeros(
                (len(batch), width), dtype=torch.float32, device=self.device
            )
            padded_target[row_index, column_index] = target_scores.float()

            if self.agent_config.double_dqn:
                chosen = padded_online.argmax(dim=1, keepdim=True)
                next_values = padded_target.gather(1, chosen).squeeze(1)
            else:
                padded_target_masked = torch.full_like(
                    padded_target, torch.finfo(torch.float32).min
                )
                padded_target_masked[row_index, column_index] = target_scores.float()
                next_values = padded_target_masked.max(dim=1).values

        # Rows with no successors (terminal transitions) are zeroed by `continues`.
        next_values = torch.nan_to_num(next_values, neginf=0.0, posinf=0.0)
        return rewards + continues * self.env_config.discount * next_values

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path, metadata: Mapping[str, object] | None = None) -> Path:
        """Write a checkpoint containing weights, configs and metadata."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CHECKPOINT_VERSION,
            "agent_config": asdict(self.agent_config),
            "env_config": asdict(self.env_config),
            "online_state": self.online.state_dict(),
            "steps_done": self.steps_done,
            "updates_done": self.updates_done,
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, destination)
        return destination

    @classmethod
    def load(
        cls, path: str | Path, device: str = "auto"
    ) -> tuple[DoubleDQNAgent, Mapping[str, object]]:
        """Load a checkpoint, returning the agent and its metadata.

        Raises:
            CheckpointError: if the file is missing, unreadable or of an
                incompatible version.
        """
        source = Path(path)
        if not source.exists():
            raise CheckpointError(f"No checkpoint at {source}.")
        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - torch raises many types
            raise CheckpointError(f"Could not read the checkpoint {source}: {exc}") from exc
        if not isinstance(payload, dict) or "online_state" not in payload:
            raise CheckpointError(f"{source} is not a dr.optimus checkpoint.")
        version = payload.get("version")
        if version != CHECKPOINT_VERSION:
            raise CheckpointError(
                f"Checkpoint {source} has version {version!r}, but this build "
                f"reads version {CHECKPOINT_VERSION}."
            )
        try:
            agent_config = AgentConfig(**_tuplify(payload["agent_config"]))
            env_config = EnvConfig(**_tuplify(payload["env_config"]))
        except TypeError as exc:
            raise CheckpointError(
                f"Checkpoint {source} holds configuration this build does not "
                f"understand: {exc}"
            ) from exc
        agent = cls(agent_config=agent_config, env_config=env_config, device=device)
        try:
            agent.online.load_state_dict(payload["online_state"])
        except RuntimeError as exc:
            raise CheckpointError(
                f"Checkpoint {source} does not match the rebuilt network: {exc}"
            ) from exc
        agent.sync_target()
        agent.steps_done = int(payload.get("steps_done", 0))
        agent.updates_done = int(payload.get("updates_done", 0))
        return agent, dict(payload.get("metadata", {}))


def _tuplify(mapping: Mapping[str, object]) -> dict[str, object]:
    """Restore tuple-typed config fields that serialization turned into lists."""
    return {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in mapping.items()
    }
