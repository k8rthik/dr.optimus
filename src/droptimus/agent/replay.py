"""Experience replay.

Transitions store SMILES strings rather than tensors. Featurization is cheap
and memoized, while a tensorized candidate set would be enormous: a single
transition can have several hundred successor molecules, each a 2048-bit
fingerprint. Storing strings keeps a 20k-transition buffer in the tens of
megabytes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from droptimus.errors import ConfigError


@dataclass(frozen=True, slots=True)
class Transition:
    """One environment transition, as consumed by the Q-learning update.

    Attributes:
        action_smiles: the molecule the agent moved to. Q(s, a) is evaluated on
            this molecule, which is why no separate "state" field is needed.
        action_steps_remaining: steps left *after* the move, i.e. the horizon
            the value of ``action_smiles`` is conditioned on.
        reward: reward received for the move.
        done: whether the episode ended with this move.
        next_candidates: successor molecules available from ``action_smiles``;
            empty when ``done``.
        next_steps_remaining: horizon for the successor molecules.
    """

    action_smiles: str
    action_steps_remaining: int
    reward: float
    done: bool
    next_candidates: tuple[str, ...]
    next_steps_remaining: int


class ReplayBuffer:
    """A fixed-capacity, uniformly sampled ring buffer of transitions."""

    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        if capacity < 1:
            raise ConfigError(f"Replay capacity must be >= 1, got {capacity}.")
        self._capacity = capacity
        self._storage: list[Transition] = []
        self._cursor = 0
        self._rng = rng or random.Random()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: Transition) -> None:
        """Insert a transition, overwriting the oldest once full."""
        if not isinstance(transition, Transition):
            raise TypeError(f"Expected a Transition, got {type(transition).__name__}.")
        if len(self._storage) < self._capacity:
            self._storage.append(transition)
        else:
            self._storage[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self._capacity

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        """Draw ``batch_size`` transitions without replacement.

        Raises:
            ValueError: if the buffer holds fewer transitions than requested.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if batch_size > len(self._storage):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer holding "
                f"{len(self._storage)}."
            )
        return tuple(self._rng.sample(self._storage, batch_size))

    def can_sample(self, batch_size: int) -> bool:
        """Return whether a batch of this size is available."""
        return 1 <= batch_size <= len(self._storage)

    def extend(self, transitions: Sequence[Transition]) -> None:
        """Insert several transitions in order."""
        for transition in transitions:
            self.add(transition)
