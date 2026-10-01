"""Tests for the molecule-editing MDP."""

from __future__ import annotations

import random

import pytest

from droptimus.chem.molecule import canonical_smiles
from droptimus.config import EnvConfig
from droptimus.env.mdp import MoleculeEnv, MoleculeState
from droptimus.errors import (
    EpisodeEndedError,
    InvalidActionError,
    InvalidSmilesError,
)
from droptimus.objectives import make_objective


def make_env(**kwargs: object) -> MoleculeEnv:
    config = EnvConfig(atom_types=("C", "N", "O"), max_steps=5, max_atoms=10)
    defaults: dict[str, object] = {
        "objective": make_objective("qed"),
        "config": config,
        "start_smiles": "C",
        "rng": random.Random(0),
    }
    defaults.update(kwargs)
    return MoleculeEnv(**defaults)  # type: ignore[arg-type]


class TestMoleculeState:
    def test_is_immutable(self) -> None:
        state = MoleculeState(smiles="CCO", step=1, max_steps=5)
        with pytest.raises((AttributeError, TypeError)):
            state.smiles = "CC"  # type: ignore[misc]

    def test_steps_remaining(self) -> None:
        assert MoleculeState("C", step=2, max_steps=5).steps_remaining == 3

    def test_done_at_budget(self) -> None:
        assert MoleculeState("C", step=5, max_steps=5).done is True
        assert MoleculeState("C", step=4, max_steps=5).done is False


class TestReset:
    def test_starts_at_the_start_molecule(self) -> None:
        env = make_env(start_smiles="CCO")
        state = env.reset()
        assert state.smiles == canonical_smiles("CCO")
        assert state.step == 0

    def test_reset_clears_previous_episode(self) -> None:
        env = make_env()
        env.reset()
        env.step(next(iter(env.valid_actions())))
        state = env.reset()
        assert state.step == 0
        assert state.smiles == canonical_smiles("C")

    def test_invalid_start_molecule_raises(self) -> None:
        with pytest.raises(InvalidSmilesError):
            make_env(start_smiles="xyz!")

    def test_rejects_nonpositive_step_budget(self) -> None:
        from droptimus.errors import ConfigError

        with pytest.raises(ConfigError):
            make_env(config=EnvConfig(max_steps=0))


class TestStep:
    def test_advances_the_counter(self) -> None:
        env = make_env()
        env.reset()
        result = env.step(sorted(env.valid_actions())[0])
        assert result.state.step == 1

    def test_applies_the_chosen_action(self) -> None:
        env = make_env(start_smiles="C")
        env.reset()
        target = canonical_smiles("CC")
        result = env.step(target)
        assert result.state.smiles == target

    def test_terminates_at_the_step_budget(self) -> None:
        env = make_env()
        env.reset()
        for _ in range(5):
            result = env.step(sorted(env.valid_actions())[0])
        assert result.done is True
        assert result.state.step == 5

    def test_stepping_after_done_raises(self) -> None:
        env = make_env()
        env.reset()
        for _ in range(5):
            env.step(sorted(env.valid_actions())[0])
        with pytest.raises(EpisodeEndedError):
            env.step("C")

    def test_rejects_action_outside_valid_set(self) -> None:
        env = make_env(start_smiles="C")
        env.reset()
        with pytest.raises(InvalidActionError):
            env.step("c1ccccc1C(=O)NCCCCN")

    def test_rejects_invalid_smiles_action(self) -> None:
        env = make_env()
        env.reset()
        with pytest.raises(InvalidSmilesError):
            env.step("not a molecule")

    def test_step_before_reset_raises(self) -> None:
        env = make_env()
        with pytest.raises(EpisodeEndedError):
            env.step("CC")

    def test_never_exceeds_atom_cap(self) -> None:
        env = make_env(
            config=EnvConfig(atom_types=("C",), max_steps=20, max_atoms=4),
            start_smiles="C",
        )
        state = env.reset()
        rng = random.Random(1)
        while not state.done:
            action = rng.choice(sorted(env.valid_actions()))
            state = env.step(action).state
            assert len(state.smiles.replace("1", "").replace("=", "")) <= 8


class TestRewards:
    def test_terminal_mode_pays_only_at_the_end(self) -> None:
        env = make_env(
            config=EnvConfig(atom_types=("C", "N", "O"), max_steps=3, reward_mode="terminal")
        )
        env.reset()
        first = env.step(sorted(env.valid_actions())[0])
        assert first.reward == 0.0
        env.step(sorted(env.valid_actions())[0])
        last = env.step(sorted(env.valid_actions())[0])
        assert last.done is True
        assert last.reward == pytest.approx(env.objective(last.state.smiles))

    def test_dense_mode_pays_the_objective_delta(self) -> None:
        env = make_env(
            config=EnvConfig(atom_types=("C", "N", "O"), max_steps=3, reward_mode="dense")
        )
        start = env.reset()
        before = env.objective(start.smiles)
        result = env.step(canonical_smiles("CC"))
        after = env.objective(result.state.smiles)
        assert result.reward == pytest.approx(after - before)

    def test_info_reports_the_objective_value(self) -> None:
        env = make_env()
        env.reset()
        result = env.step(canonical_smiles("CC"))
        assert result.info["objective"] == pytest.approx(env.objective("CC"))


class TestValidActions:
    def test_reflects_the_current_state(self) -> None:
        env = make_env(start_smiles="C")
        env.reset()
        first = env.valid_actions()
        env.step(canonical_smiles("CC"))
        assert env.valid_actions() != first

    def test_action_cap_is_applied(self) -> None:
        env = make_env(
            config=EnvConfig(atom_types=("C", "N", "O"), max_steps=5, max_actions=4),
            start_smiles="CCCC",
        )
        env.reset()
        assert len(env.valid_actions()) == 4

    def test_no_actions_once_episode_is_over(self) -> None:
        env = make_env()
        env.reset()
        for _ in range(5):
            env.step(sorted(env.valid_actions())[0])
        assert env.valid_actions() == frozenset()


class TestEpisodeRollout:
    def test_random_rollout_stays_valid(self) -> None:
        env = make_env(config=EnvConfig(atom_types=("C", "N", "O"), max_steps=10, max_atoms=12))
        rng = random.Random(7)
        for _ in range(5):
            state = env.reset()
            while not state.done:
                actions = sorted(env.valid_actions())
                assert actions, "a live episode must always have at least one action"
                state = env.step(rng.choice(actions)).state
            canonical_smiles(state.smiles)  # raises if the final molecule is broken


class TestActionCapConsistency:
    """A capped action set must stay stable between offer and step.

    With EnvConfig.max_actions the set is a random subsample; re-drawing it
    inside step() would reject actions the caller was just offered.
    """

    def test_offered_action_is_always_accepted(self) -> None:
        env = make_env(
            config=EnvConfig(
                atom_types=("C", "N", "O"), max_steps=8, max_atoms=12, max_actions=3
            ),
            start_smiles="CCCC",
        )
        rng = random.Random(3)
        for _ in range(10):
            state = env.reset()
            while not state.done:
                offered = sorted(env.valid_actions())
                state = env.step(rng.choice(offered)).state

    def test_repeated_queries_return_the_same_set(self) -> None:
        env = make_env(
            config=EnvConfig(atom_types=("C", "N", "O"), max_steps=5, max_actions=3),
            start_smiles="CCCC",
        )
        env.reset()
        assert env.valid_actions() == env.valid_actions()
