"""Tests for rollouts, measured metrics and the training loop."""

from __future__ import annotations

import itertools
import random
from dataclasses import replace

import pytest

from droptimus.agent.dqn import DoubleDQNAgent
from droptimus.config import AgentConfig, EnvConfig, RunConfig, TrainConfig
from droptimus.env.mdp import MoleculeEnv
from droptimus.errors import ConfigError
from droptimus.evaluate import (
    Generated,
    comparison_rows,
    compute_metrics,
    format_metrics,
    generated_from_episodes,
)
from droptimus.objectives import make_objective
from droptimus.rollout import agent_policy, random_policy, rollout
from droptimus.train import collect_episodes, epsilon_at, train

SMALL_ENV = EnvConfig(atom_types=("C", "N", "O"), max_steps=4, max_atoms=8)
SMALL_AGENT = AgentConfig(
    hidden_sizes=(16, 8), batch_size=4, replay_capacity=256, bootstrap_actions=4
)


def small_run(**train_kwargs: object) -> RunConfig:
    defaults = {"episodes": 3, "warmup_episodes": 1, "seed": 0, "device": "cpu", "log_every": 10}
    defaults.update(train_kwargs)
    return RunConfig(
        objective="qed",
        start_smiles="C",
        env=SMALL_ENV,
        agent=SMALL_AGENT,
        train=TrainConfig(**defaults),  # type: ignore[arg-type]
    )


def make_env(start: str = "C", objective: str = "qed") -> MoleculeEnv:
    return MoleculeEnv(
        objective=make_objective(objective),
        config=SMALL_ENV,
        start_smiles=start,
        rng=random.Random(0),
    )


class TestRollout:
    def test_runs_the_full_budget(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(0)))
        assert episode.steps == SMALL_ENV.max_steps
        assert len(episode.trajectory) == SMALL_ENV.max_steps + 1

    def test_records_one_transition_per_step(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(0)))
        assert len(episode.transitions) == SMALL_ENV.max_steps

    def test_last_transition_is_terminal(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(0)))
        assert episode.transitions[-1].done is True
        assert episode.transitions[-1].next_candidates == ()

    def test_non_terminal_transitions_carry_successors(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(0)))
        assert all(t.next_candidates for t in episode.transitions[:-1])

    def test_transitions_can_be_skipped(self) -> None:
        episode = rollout(
            make_env(), random_policy(random.Random(0)), record_transitions=False
        )
        assert episode.transitions == ()

    def test_best_objective_is_the_maximum_visited(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(1)))
        objective = make_objective("qed")
        visited = [objective(s) for s in episode.trajectory]
        assert episode.best_objective == pytest.approx(max(visited))

    def test_best_is_at_least_the_final(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(2)))
        assert episode.best_objective >= episode.final_objective - 1e-9

    def test_improvement_is_best_minus_start(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(3)))
        assert episode.improvement == pytest.approx(
            episode.best_objective - episode.start_objective
        )

    def test_agent_policy_is_deterministic_when_greedy(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        first = rollout(make_env(), agent_policy(agent, 0.0))
        second = rollout(make_env(), agent_policy(agent, 0.0))
        assert first.trajectory == second.trajectory

    def test_terminal_reward_equals_final_objective(self) -> None:
        env = make_env()
        episode = rollout(env, random_policy(random.Random(4)))
        assert episode.total_reward == pytest.approx(episode.final_objective)


class TestMetrics:
    def make_generated(self) -> tuple[Generated, ...]:
        return (
            Generated("C", "CCO", 0.40, 0.10, 0.20),
            Generated("C", "CCN", 0.60, 0.10, 0.30),
            Generated("C", "CCO", 0.40, 0.10, 0.20),
            Generated("C", "c1ccccc1", 0.80, 0.10, 0.05),
        )

    def test_counts_and_central_tendency(self) -> None:
        metrics = compute_metrics(self.make_generated())
        assert metrics.n == 4
        assert metrics.objective_mean == pytest.approx(0.55)
        assert metrics.objective_max == pytest.approx(0.80)

    def test_top3_is_descending(self) -> None:
        metrics = compute_metrics(self.make_generated())
        assert metrics.top3 == pytest.approx((0.80, 0.60, 0.40))

    def test_uniqueness_counts_distinct_molecules(self) -> None:
        metrics = compute_metrics(self.make_generated())
        assert metrics.uniqueness == pytest.approx(3 / 4)

    def test_novelty_excludes_the_reference_set(self) -> None:
        metrics = compute_metrics(self.make_generated(), reference=["CCO"])
        # 3 unique molecules, one of which is in the reference set.
        assert metrics.novelty == pytest.approx(2 / 3)

    def test_start_molecules_never_count_as_novel(self) -> None:
        generated = (Generated("CCO", "CCO", 0.4, 0.4, 1.0),)
        assert compute_metrics(generated).novelty == pytest.approx(0.0)

    def test_validity_is_measured_not_assumed(self) -> None:
        generated = (Generated("C", "CCO", 0.4, 0.1, 0.2),)
        assert compute_metrics(generated).validity == pytest.approx(1.0)

    def test_improvement_statistics(self) -> None:
        metrics = compute_metrics(self.make_generated())
        assert metrics.improvement_mean == pytest.approx(0.45)
        assert metrics.fraction_improved == pytest.approx(1.0)

    def test_fraction_improved_counts_only_gains(self) -> None:
        generated = (
            Generated("C", "CC", 0.2, 0.3, 0.5),
            Generated("C", "CCO", 0.5, 0.3, 0.4),
        )
        assert compute_metrics(generated).fraction_improved == pytest.approx(0.5)

    def test_constraint_fraction_is_reported_when_delta_given(self) -> None:
        metrics = compute_metrics(self.make_generated(), similarity_delta=0.25)
        assert metrics.constraint_satisfied == pytest.approx(1 / 4)

    def test_constraint_is_none_without_delta(self) -> None:
        assert compute_metrics(self.make_generated()).constraint_satisfied is None

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_metrics([])

    def test_single_molecule_has_zero_spread(self) -> None:
        metrics = compute_metrics((Generated("C", "CC", 0.3, 0.1, 0.5),))
        assert metrics.objective_std == 0.0

    def test_as_dict_is_json_friendly(self) -> None:
        import json

        json.dumps(compute_metrics(self.make_generated()).as_dict())

    def test_format_is_human_readable(self) -> None:
        text = format_metrics(compute_metrics(self.make_generated()))
        assert "objective" in text and "novelty" in text

    def test_episode_conversion_uses_best_by_default(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(5)))
        records = generated_from_episodes([episode])
        assert records[0].smiles == episode.best_smiles

    def test_episode_conversion_can_use_final(self) -> None:
        episode = rollout(make_env(), random_policy(random.Random(5)))
        records = generated_from_episodes([episode], use_best=False)
        assert records[0].smiles == episode.final_smiles


class TestComparisonRows:
    def test_qed_rows_cite_published_values(self) -> None:
        metrics = compute_metrics(
            (
                Generated("C", "CC", 0.9, 0.1, 0.1),
                Generated("C", "CCC", 0.8, 0.1, 0.1),
                Generated("C", "CCO", 0.7, 0.1, 0.1),
            )
        )
        rows = comparison_rows("qed", metrics)
        assert rows[0] == ("best #1", "0.900", "0.948")

    def test_unknown_objective_has_no_fabricated_reference(self) -> None:
        metrics = compute_metrics((Generated("C", "CC", 0.9, 0.1, 0.1),))
        rows = comparison_rows("similarity", metrics)
        assert all(row[2] in {"n/a", "not reported"} for row in rows)

    def test_missing_measurements_are_not_invented(self) -> None:
        metrics = compute_metrics((Generated("C", "CC", 0.9, 0.1, 0.1),))
        rows = comparison_rows("qed", metrics)
        assert rows[1][1] == "n/a"


class TestEpsilonSchedule:
    def test_starts_at_epsilon_start(self) -> None:
        config = small_run(episodes=100, epsilon_start=1.0, epsilon_end=0.01)
        assert epsilon_at(0, config) == pytest.approx(1.0)

    def test_reaches_epsilon_end(self) -> None:
        config = small_run(
            episodes=100, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_fraction=0.5
        )
        assert epsilon_at(50, config) == pytest.approx(0.01)
        assert epsilon_at(99, config) == pytest.approx(0.01)

    def test_is_monotone(self) -> None:
        config = small_run(episodes=50)
        values = [epsilon_at(i, config) for i in range(50)]
        assert all(b <= a + 1e-12 for a, b in itertools.pairwise(values))

    def test_bad_decay_fraction_raises(self) -> None:
        with pytest.raises(ConfigError):
            epsilon_at(0, small_run(epsilon_decay_fraction=0.0))


class TestTrain:
    def test_produces_one_history_entry_per_episode(self) -> None:
        result = train(small_run(episodes=3))
        assert result.episodes == 3

    def test_tracks_the_best_molecule(self) -> None:
        result = train(small_run(episodes=3))
        assert result.best_objective == pytest.approx(
            max(item.best_objective for item in result.history)
        )

    def test_writes_a_checkpoint_and_history(self, tmp_path) -> None:
        result = train(small_run(episodes=2), out_dir=tmp_path)
        assert result.checkpoint_path is not None and result.checkpoint_path.exists()
        assert (tmp_path / "history.json").exists()
        assert (tmp_path / "run_config.json").exists()

    def test_checkpoint_reloads(self, tmp_path) -> None:
        result = train(small_run(episodes=2), out_dir=tmp_path)
        agent, metadata = DoubleDQNAgent.load(result.checkpoint_path, device="cpu")
        assert metadata["objective"] == "qed"
        assert agent.q_values(["CCO"], 1).shape == (1,)

    def test_zero_episodes_raises(self) -> None:
        with pytest.raises(ConfigError):
            train(small_run(episodes=0))

    def test_same_seed_reproduces_the_history(self) -> None:
        first = train(small_run(episodes=3, seed=7))
        second = train(small_run(episodes=3, seed=7))
        assert [item.best_objective for item in first.history] == pytest.approx(
            [item.best_objective for item in second.history]
        )

    def test_cycles_through_multiple_start_molecules(self) -> None:
        result = train(small_run(episodes=4), start_molecules=["C", "CCO"])
        starts = [item.start_smiles for item in result.history]
        assert starts[0] != starts[1]
        assert starts[0] == starts[2]

    def test_constrained_objective_gets_the_start_as_reference(self) -> None:
        config = replace(
            small_run(episodes=2),
            objective="constrained",
            objective_kwargs=(("delta", 0.2),),
            start_smiles="CCO",
        )
        result = train(config)
        assert result.episodes == 2


class TestCollectEpisodes:
    def test_one_episode_per_start(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        episodes = collect_episodes(small_run(), ["C", "CCO"], agent)
        assert len(episodes) == 2
        assert [e.start_smiles for e in episodes] == ["C", "CCO"]

    def test_random_baseline_needs_no_agent(self) -> None:
        episodes = collect_episodes(small_run(), ["C"], None, rng=random.Random(0))
        assert len(episodes) == 1
        assert episodes[0].steps == SMALL_ENV.max_steps

    def test_greedy_collection_is_reproducible(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        first = collect_episodes(small_run(), ["CCO"], agent)
        second = collect_episodes(small_run(), ["CCO"], agent)
        assert first[0].trajectory == second[0].trajectory
