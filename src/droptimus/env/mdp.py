"""The molecule-editing Markov decision process.

State is (molecule, steps taken). Actions are the molecules reachable in one
valence-valid edit (see :mod:`droptimus.env.actions`). An episode runs for a
fixed step budget, as in Zhou et al. (2019).

Two reward shapes are supported:

``terminal``
    Reward is zero on every step except the last, where it is the objective
    value of the final molecule. This is the clean episodic formulation: with
    agent discount gamma, the value of a state is gamma^(steps remaining) times
    the achievable final objective.

``dense``
    Reward is the change in objective value caused by the edit. The episode
    return then telescopes to (final - initial) objective.

``paper``
    What the reference implementation actually does: every step pays
    ``objective(current) * gamma^(steps remaining)``. This sums a scaled copy of
    the objective once per step rather than once per episode, so returns are not
    comparable with the other two modes, but it gives the agent a reward signal
    at every step instead of only at the horizon.

``terminal`` is the default because it is the unambiguous episodic formulation;
``paper`` exists so the reproduction can be run exactly as published.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field

from droptimus.chem.molecule import canonical_smiles
from droptimus.config import EnvConfig
from droptimus.env.actions import valid_actions
from droptimus.errors import ConfigError, EpisodeEndedError, InvalidActionError
from droptimus.objectives.base import Objective


@dataclass(frozen=True, slots=True)
class MoleculeState:
    """An immutable MDP state: a molecule plus how far into the episode we are."""

    smiles: str
    step: int
    max_steps: int

    @property
    def steps_remaining(self) -> int:
        return max(self.max_steps - self.step, 0)

    @property
    def done(self) -> bool:
        return self.step >= self.max_steps


@dataclass(frozen=True, slots=True)
class StepResult:
    """The outcome of one environment transition."""

    state: MoleculeState
    reward: float
    done: bool
    info: Mapping[str, float] = field(default_factory=dict)


class MoleculeEnv:
    """A fixed-budget molecule-editing environment for a single objective."""

    def __init__(
        self,
        objective: Objective,
        config: EnvConfig | None = None,
        start_smiles: str = "C",
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or EnvConfig()
        self._validate_config(self.config)
        self.objective = objective
        self.start_smiles = canonical_smiles(start_smiles)
        self._rng = rng or random.Random()
        self._state: MoleculeState | None = None
        # The action set offered for the current state. Cached because with
        # EnvConfig.max_actions the set is a random subsample: re-drawing it
        # inside step() could reject the action the caller was just offered.
        self._offered: tuple[str, frozenset[str]] | None = None

    @staticmethod
    def _validate_config(config: EnvConfig) -> None:
        if config.max_steps < 1:
            raise ConfigError(f"max_steps must be >= 1, got {config.max_steps}.")
        if config.max_atoms < 1:
            raise ConfigError(f"max_atoms must be >= 1, got {config.max_atoms}.")
        if not config.atom_types:
            raise ConfigError("atom_types must list at least one element.")
        if config.max_actions is not None and config.max_actions < 1:
            raise ConfigError(
                f"max_actions must be >= 1 or None, got {config.max_actions}."
            )
        if not 0.0 < config.discount <= 1.0:
            raise ConfigError(f"discount must lie in (0, 1], got {config.discount}.")

    # --- episode lifecycle -------------------------------------------------

    def reset(self) -> MoleculeState:
        """Start a new episode at the configured start molecule."""
        self._state = MoleculeState(
            smiles=self.start_smiles, step=0, max_steps=self.config.max_steps
        )
        self._offered = None
        return self._state

    @property
    def state(self) -> MoleculeState:
        """Return the current state.

        Raises:
            EpisodeEndedError: if :meth:`reset` has not been called.
        """
        if self._state is None:
            raise EpisodeEndedError("Call reset() before using the environment.")
        return self._state

    def valid_actions(self) -> frozenset[str]:
        """Return the candidate successor molecules for the current state."""
        state = self.state
        if state.done:
            return frozenset()
        if self._offered is not None and self._offered[0] == state.smiles:
            return self._offered[1]
        actions = valid_actions(state.smiles, self.config, rng=self._rng)
        self._offered = (state.smiles, actions)
        return actions

    def step(self, action: str) -> StepResult:
        """Apply ``action`` (a successor molecule's SMILES) to the current state.

        Raises:
            EpisodeEndedError: if the episode has finished or has not started.
            InvalidSmilesError: if ``action`` is not a parseable molecule.
            InvalidActionError: if ``action`` is not reachable in one edit.
        """
        state = self.state
        if state.done:
            raise EpisodeEndedError(
                f"The episode already used its {state.max_steps}-step budget; "
                "call reset() to start another."
            )
        chosen = canonical_smiles(action)
        allowed = self.valid_actions()
        if chosen not in allowed:
            raise InvalidActionError(
                f"{chosen!r} is not reachable from {state.smiles!r} in one edit "
                f"({len(allowed)} actions are)."
            )

        next_state = MoleculeState(
            smiles=chosen, step=state.step + 1, max_steps=state.max_steps
        )
        reward = self._reward(
            state.smiles, chosen, next_state.done, next_state.steps_remaining
        )
        info = dict(self.objective.components(chosen))
        info["objective"] = self.objective.score(chosen)
        self._state = next_state
        return StepResult(
            state=next_state, reward=reward, done=next_state.done, info=info
        )

    # --- rewards -----------------------------------------------------------

    def _reward(
        self, previous: str, current: str, done: bool, steps_remaining: int
    ) -> float:
        mode = self.config.reward_mode
        if mode == "dense":
            return self.objective.score(current) - self.objective.score(previous)
        if mode == "paper":
            return self.objective.score(current) * (
                self.config.discount**steps_remaining
            )
        return self.objective.score(current) if done else 0.0
