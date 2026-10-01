"""Tests for the replay buffer, Q-networks and the double-DQN agent."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from droptimus.agent.dqn import DoubleDQNAgent
from droptimus.agent.featurizers import build_featurizer
from droptimus.agent.networks import build_network
from droptimus.agent.replay import ReplayBuffer, Transition
from droptimus.agent.torch_utils import resolve_device, set_global_seed
from droptimus.config import AgentConfig, EnvConfig
from droptimus.errors import CheckpointError, ConfigError

SMALL_ENV = EnvConfig(atom_types=("C", "N", "O"), max_steps=5, max_atoms=10)
SMALL_AGENT = AgentConfig(
    hidden_sizes=(32, 16),
    batch_size=4,
    replay_capacity=64,
    bootstrap_actions=6,
    target_sync_steps=3,
)


def make_transition(smiles: str = "CC", done: bool = False) -> Transition:
    return Transition(
        action_smiles=smiles,
        action_steps_remaining=3,
        reward=0.5,
        done=done,
        next_candidates=() if done else ("CCC", "CCO", "CC=O"),
        next_steps_remaining=2,
    )


class TestReplayBuffer:
    def test_grows_then_wraps(self) -> None:
        buffer = ReplayBuffer(capacity=3, rng=random.Random(0))
        for i in range(5):
            buffer.add(make_transition(smiles="C" * (i + 1)))
        assert len(buffer) == 3

    def test_wrap_overwrites_oldest(self) -> None:
        buffer = ReplayBuffer(capacity=2, rng=random.Random(0))
        buffer.add(make_transition("C"))
        buffer.add(make_transition("CC"))
        buffer.add(make_transition("CCC"))
        stored = {t.action_smiles for t in buffer.sample(2)}
        assert stored == {"CC", "CCC"}

    def test_sample_returns_requested_count(self) -> None:
        buffer = ReplayBuffer(capacity=10, rng=random.Random(0))
        buffer.extend([make_transition("C" * (i + 1)) for i in range(6)])
        assert len(buffer.sample(4)) == 4

    def test_oversampling_raises(self) -> None:
        buffer = ReplayBuffer(capacity=10, rng=random.Random(0))
        buffer.add(make_transition())
        with pytest.raises(ValueError):
            buffer.sample(5)

    def test_zero_batch_raises(self) -> None:
        buffer = ReplayBuffer(capacity=10, rng=random.Random(0))
        buffer.add(make_transition())
        with pytest.raises(ValueError):
            buffer.sample(0)

    def test_bad_capacity_raises(self) -> None:
        with pytest.raises(ConfigError):
            ReplayBuffer(capacity=0)

    def test_rejects_non_transition(self) -> None:
        buffer = ReplayBuffer(capacity=4)
        with pytest.raises(TypeError):
            buffer.add({"not": "a transition"})  # type: ignore[arg-type]

    def test_can_sample_reports_readiness(self) -> None:
        buffer = ReplayBuffer(capacity=4, rng=random.Random(0))
        assert buffer.can_sample(2) is False
        buffer.extend([make_transition(), make_transition("CCC")])
        assert buffer.can_sample(2) is True

    def test_transition_is_immutable(self) -> None:
        transition = make_transition()
        with pytest.raises((AttributeError, TypeError)):
            transition.reward = 1.0  # type: ignore[misc]


class TestDeviceResolution:
    def test_auto_returns_a_usable_device(self) -> None:
        device = resolve_device("auto")
        assert device.type in {"cpu", "mps", "cuda"}

    def test_cpu_is_always_available(self) -> None:
        assert resolve_device("cpu").type == "cpu"

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ConfigError):
            resolve_device("tpu")

    def test_seeding_is_reproducible(self) -> None:
        set_global_seed(11)
        first = torch.rand(3)
        set_global_seed(11)
        assert torch.allclose(first, torch.rand(3))


class TestNetworks:
    @pytest.mark.parametrize("encoder", ["fingerprint", "gnn"])
    def test_scores_one_value_per_molecule(self, encoder: str) -> None:
        featurizer = build_featurizer(encoder, max_steps=5, max_atoms=10)
        network = build_network(
            encoder, featurizer, hidden_sizes=(16, 8), gnn_hidden=12, gnn_layers=2
        )
        smiles = ["C", "CCO", "c1ccccc1"]
        batch = featurizer.batch(smiles, [1, 2, 3], torch.device("cpu"))
        scores = network(batch)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    @pytest.mark.parametrize("encoder", ["fingerprint", "gnn"])
    def test_score_is_permutation_consistent(self, encoder: str) -> None:
        featurizer = build_featurizer(encoder, max_steps=5, max_atoms=10)
        network = build_network(
            encoder, featurizer, hidden_sizes=(16, 8), gnn_hidden=12, gnn_layers=2
        )
        network.eval()
        device = torch.device("cpu")
        with torch.no_grad():
            forward = network(featurizer.batch(["CCO", "c1ccccc1"], [2, 2], device))
            reverse = network(featurizer.batch(["c1ccccc1", "CCO"], [2, 2], device))
        assert forward[0].item() == pytest.approx(reverse[1].item(), abs=1e-5)

    def test_gnn_is_invariant_to_atom_numbering(self) -> None:
        featurizer = build_featurizer("gnn", max_steps=5, max_atoms=12)
        network = build_network(
            "gnn", featurizer, hidden_sizes=(16, 8), gnn_hidden=12, gnn_layers=2
        )
        network.eval()
        device = torch.device("cpu")
        with torch.no_grad():
            # Same molecule written two ways: the graph is identical, so the
            # pooled encoding must be too.
            scores = network(featurizer.batch(["OCC", "CCO"], [2, 2], device))
        assert scores[0].item() == pytest.approx(scores[1].item(), abs=1e-5)

    def test_unknown_encoder_raises(self) -> None:
        with pytest.raises(ValueError):
            build_featurizer("transformer", max_steps=5, max_atoms=10)

    def test_featurizer_rejects_mismatched_lengths(self) -> None:
        featurizer = build_featurizer("fingerprint", max_steps=5, max_atoms=10)
        with pytest.raises(ValueError):
            featurizer.batch(["CCO", "CC"], [1], torch.device("cpu"))


class TestAgentActionSelection:
    def test_greedy_choice_is_the_argmax(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        candidates = ["C", "CC", "CCO", "c1ccccc1"]
        values = agent.q_values(candidates, steps_remaining=2)
        assert agent.select_action(candidates, 2, epsilon=0.0) == candidates[
            int(np.argmax(values))
        ]

    def test_epsilon_one_explores(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        candidates = [f"{'C' * n}" for n in range(1, 9)]
        picks = {agent.select_action(candidates, 2, epsilon=1.0) for _ in range(40)}
        assert len(picks) > 1

    def test_empty_candidates_raises(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        with pytest.raises(ValueError):
            agent.select_action([], 2)

    def test_out_of_range_epsilon_raises(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        with pytest.raises(ValueError):
            agent.select_action(["CC"], 2, epsilon=1.5)

    def test_q_values_length_matches_candidates(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        assert agent.q_values(["C", "CC", "CCC"], 1).shape == (3,)

    def test_q_values_of_nothing_is_empty(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        assert agent.q_values([], 1).shape == (0,)

    def test_same_seed_gives_same_initial_values(self) -> None:
        first = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=5)
        second = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=5)
        assert np.allclose(
            first.q_values(["CCO", "CC"], 2), second.q_values(["CCO", "CC"], 2)
        )


class TestAgentLearning:
    def test_update_is_none_until_the_buffer_fills(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        assert agent.update() is None
        agent.observe(make_transition())
        assert agent.update() is None

    def test_update_returns_a_finite_loss(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        for smiles in ("C", "CC", "CCC", "CCO", "CC=O", "CCN"):
            agent.observe(make_transition(smiles))
        loss = agent.update()
        assert loss is not None and np.isfinite(loss)

    def test_training_reduces_loss_on_a_fixed_batch(self) -> None:
        agent = DoubleDQNAgent(
            AgentConfig(
                hidden_sizes=(32, 16),
                batch_size=4,
                replay_capacity=16,
                bootstrap_actions=4,
                target_sync_steps=10_000,
                learning_rate=1e-2,
            ),
            SMALL_ENV,
            device="cpu",
            seed=0,
        )
        for smiles in ("C", "CC", "CCC", "CCO"):
            agent.observe(make_transition(smiles, done=True))
        first = agent.update()
        for _ in range(60):
            agent.update()
        last = agent.update()
        assert first is not None and last is not None
        assert last < first

    def test_terminal_transitions_have_no_bootstrap(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        batch = (make_transition("CC", done=True),) * 4
        targets = agent._bootstrap_targets(batch)
        assert torch.allclose(targets, torch.full((4,), 0.5))

    def test_nonterminal_targets_include_discounted_value(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        batch = (make_transition("CC", done=False),) * 4
        targets = agent._bootstrap_targets(batch)
        assert not torch.allclose(targets, torch.full((4,), 0.5))

    def test_target_network_syncs_on_schedule(self) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        for smiles in ("C", "CC", "CCC", "CCO", "CCN", "CC=O"):
            agent.observe(make_transition(smiles))
        for _ in range(20):
            agent.update()
        online = agent.online.state_dict()
        target = agent.target.state_dict()
        # target_sync_steps=3 and 6 observations, so the last sync happened at
        # step 6; the updates since then moved the online net away from it.
        assert any(
            not torch.allclose(online[key], target[key]) for key in online
        )

    def test_candidate_set_is_truncated_to_the_cap(self) -> None:
        agent = DoubleDQNAgent(
            AgentConfig(bootstrap_actions=2, batch_size=1, replay_capacity=4),
            SMALL_ENV,
            device="cpu",
            seed=0,
        )
        agent.observe(make_transition())
        stored = agent.replay.sample(1)[0]
        assert len(stored.next_candidates) == 2

    def test_vanilla_dqn_variant_also_runs(self) -> None:
        agent = DoubleDQNAgent(
            AgentConfig(
                hidden_sizes=(16,), batch_size=2, replay_capacity=8, double_dqn=False
            ),
            SMALL_ENV,
            device="cpu",
            seed=0,
        )
        for smiles in ("C", "CC", "CCC"):
            agent.observe(make_transition(smiles))
        assert np.isfinite(agent.update())


class TestAgentConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"batch_size": 0},
            {"learning_rate": 0.0},
            {"bootstrap_actions": 0},
            {"target_sync_steps": 0},
            {"hidden_sizes": ()},
        ],
    )
    def test_bad_config_raises(self, kwargs: dict) -> None:
        with pytest.raises(ConfigError):
            DoubleDQNAgent(AgentConfig(**kwargs), SMALL_ENV, device="cpu")


class TestCheckpoints:
    def test_round_trip_preserves_q_values(self, tmp_path) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=3)
        for smiles in ("C", "CC", "CCC", "CCO"):
            agent.observe(make_transition(smiles))
        agent.update()
        before = agent.q_values(["CCO", "c1ccccc1"], 2)

        path = agent.save(tmp_path / "agent.pt", metadata={"objective": "qed"})
        restored, metadata = DoubleDQNAgent.load(path, device="cpu")

        assert metadata["objective"] == "qed"
        assert np.allclose(before, restored.q_values(["CCO", "c1ccccc1"], 2), atol=1e-6)

    def test_counters_survive(self, tmp_path) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=3)
        agent.observe(make_transition())
        path = agent.save(tmp_path / "agent.pt")
        restored, _ = DoubleDQNAgent.load(path, device="cpu")
        assert restored.steps_done == agent.steps_done

    def test_gnn_checkpoint_round_trips(self, tmp_path) -> None:
        config = AgentConfig(
            encoder="gnn", hidden_sizes=(16, 8), gnn_hidden=12, gnn_layers=2
        )
        agent = DoubleDQNAgent(config, SMALL_ENV, device="cpu", seed=1)
        before = agent.q_values(["CCO"], 2)
        path = agent.save(tmp_path / "gnn.pt")
        restored, _ = DoubleDQNAgent.load(path, device="cpu")
        assert np.allclose(before, restored.q_values(["CCO"], 2), atol=1e-6)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(CheckpointError):
            DoubleDQNAgent.load(tmp_path / "nope.pt")

    def test_foreign_file_raises(self, tmp_path) -> None:
        path = tmp_path / "foreign.pt"
        torch.save({"something": "else"}, path)
        with pytest.raises(CheckpointError):
            DoubleDQNAgent.load(path)

    def test_version_mismatch_raises(self, tmp_path) -> None:
        agent = DoubleDQNAgent(SMALL_AGENT, SMALL_ENV, device="cpu", seed=0)
        path = agent.save(tmp_path / "agent.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["version"] = 999
        torch.save(payload, path)
        with pytest.raises(CheckpointError):
            DoubleDQNAgent.load(path)
