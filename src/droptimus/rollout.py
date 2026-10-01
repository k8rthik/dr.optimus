"""Episode rollout, shared by training, optimization, evaluation and baselines.

A policy is any callable ``(candidates, steps_remaining) -> chosen candidate``,
so the same rollout code drives the learned agent, a greedy agent and the
random-edit baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from droptimus.agent.replay import Transition
from droptimus.env.mdp import MoleculeEnv

Policy = Callable[[Sequence[str], int], str]


class SupportsSelectAction(Protocol):
    """The part of the agent interface a rollout needs."""

    def select_action(
        self, candidates: Sequence[str], steps_remaining: int, epsilon: float = 0.0
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class Episode:
    """The outcome of one episode.

    ``best_*`` tracks the highest-objective molecule *visited*, which is what the
    benchmark literature reports; ``final_*`` is where the episode ended.
    """

    start_smiles: str
    start_objective: float
    final_smiles: str
    final_objective: float
    best_smiles: str
    best_objective: float
    total_reward: float
    steps: int
    trajectory: tuple[str, ...]
    transitions: tuple[Transition, ...]

    @property
    def improvement(self) -> float:
        """Objective gain of the best visited molecule over the start molecule."""
        return self.best_objective - self.start_objective


def agent_policy(agent: SupportsSelectAction, epsilon: float = 0.0) -> Policy:
    """Return an epsilon-greedy policy backed by ``agent``."""

    def policy(candidates: Sequence[str], steps_remaining: int) -> str:
        return agent.select_action(candidates, steps_remaining, epsilon)

    return policy


def random_policy(rng: random.Random) -> Policy:
    """Return the uniform random-edit policy used as a baseline."""

    def policy(candidates: Sequence[str], steps_remaining: int) -> str:
        return rng.choice(sorted(candidates))

    return policy


def rollout(
    env: MoleculeEnv, policy: Policy, record_transitions: bool = True
) -> Episode:
    """Run one episode of ``env`` under ``policy``.

    Raises:
        ValueError: if a live state offers no actions, which would mean the
            action enumerator is misconfigured.
    """
    state = env.reset()
    start_smiles = state.smiles
    start_objective = env.objective.score(start_smiles)

    trajectory = [start_smiles]
    transitions: list[Transition] = []
    total_reward = 0.0
    best_smiles, best_objective = start_smiles, start_objective

    while not state.done:
        candidates = sorted(env.valid_actions())
        if not candidates:
            raise ValueError(
                f"No valid actions from {state.smiles!r}; check atom_types, "
                "max_atoms and allowed_ring_sizes."
            )
        chosen = policy(candidates, state.steps_remaining)
        result = env.step(chosen)
        total_reward += result.reward
        state = result.state
        trajectory.append(state.smiles)

        objective_value = result.info.get("objective")
        if objective_value is None:  # pragma: no cover - env always supplies it
            objective_value = env.objective.score(state.smiles)
        if objective_value > best_objective:
            best_smiles, best_objective = state.smiles, float(objective_value)

        if record_transitions:
            next_candidates = () if state.done else tuple(sorted(env.valid_actions()))
            transitions.append(
                Transition(
                    action_smiles=state.smiles,
                    action_steps_remaining=state.steps_remaining,
                    reward=result.reward,
                    done=result.done,
                    next_candidates=next_candidates,
                    next_steps_remaining=max(state.steps_remaining - 1, 0),
                )
            )

    return Episode(
        start_smiles=start_smiles,
        start_objective=start_objective,
        final_smiles=state.smiles,
        final_objective=env.objective.score(state.smiles),
        best_smiles=best_smiles,
        best_objective=best_objective,
        total_reward=total_reward,
        steps=state.step,
        trajectory=tuple(trajectory),
        transitions=tuple(transitions),
    )
