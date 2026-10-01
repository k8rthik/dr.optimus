"""Tests for the command-line interface.

These exercise argument validation and the small end-to-end paths; no test
touches the network or runs a full training job.
"""

from __future__ import annotations

import json

import pytest

from droptimus.cli import (
    _parse_atom_types,
    build_parser,
    main,
    resolve_start_set,
    run_config_from_args,
)
from droptimus.errors import DrOptimusError

TINY_TRAIN = [
    "train",
    "--episodes",
    "2",
    "--warmup-episodes",
    "1",
    "--max-steps",
    "3",
    "--max-atoms",
    "8",
    "--device",
    "cpu",
    "--batch-size",
    "2",
    "--bootstrap-actions",
    "4",
    "--replay-capacity",
    "32",
]


class TestParser:
    def test_requires_a_command(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_rejects_unknown_command(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["frobnicate"])

    def test_version_exits_cleanly(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--version"])
        assert excinfo.value.code == 0

    def test_optimize_requires_a_checkpoint(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["optimize", "CCO"])

    def test_rejects_unknown_encoder(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["train", "--encoder", "transformer"])


class TestAtomTypeParsing:
    def test_parses_a_list(self) -> None:
        assert _parse_atom_types("C, N ,O") == ("C", "N", "O")

    def test_empty_raises(self) -> None:
        with pytest.raises(DrOptimusError):
            _parse_atom_types("  ,  ")

    def test_unknown_element_raises(self) -> None:
        with pytest.raises(DrOptimusError):
            _parse_atom_types("C,Xx")


class TestRunConfigFromArgs:
    def parse(self, extra: list[str]) -> object:
        return build_parser().parse_args(TINY_TRAIN + extra)

    def test_canonicalizes_the_start_molecule(self) -> None:
        config = run_config_from_args(self.parse(["--start", "OCC"]))
        assert config.start_smiles == "CCO"

    def test_invalid_start_molecule_raises(self) -> None:
        with pytest.raises(DrOptimusError):
            run_config_from_args(self.parse(["--start", "this-is-not-a-molecule"]))

    def test_unknown_objective_raises_with_the_list(self) -> None:
        with pytest.raises(DrOptimusError) as excinfo:
            run_config_from_args(self.parse(["--objective", "magic"]))
        assert "qed" in str(excinfo.value)

    def test_constrained_objective_records_its_arguments(self) -> None:
        config = run_config_from_args(
            self.parse(["--objective", "constrained", "--delta", "0.6"])
        )
        assert dict(config.objective_kwargs)["delta"] == pytest.approx(0.6)

    def test_env_and_agent_flags_are_wired_through(self) -> None:
        config = run_config_from_args(self.parse(["--max-steps", "7", "--encoder", "gnn"]))
        assert config.env.max_steps == 7
        assert config.agent.encoder == "gnn"


class TestStartSets:
    def test_single_uses_the_given_molecule(self) -> None:
        assert resolve_start_set("single", 10, "OCC") == ("CCO",)

    def test_single_without_a_molecule_raises(self) -> None:
        with pytest.raises(DrOptimusError):
            resolve_start_set("single", 10, None)

    def test_fixture_is_truncated_to_the_count(self) -> None:
        assert len(resolve_start_set("fixture", 7, None)) == 7

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(DrOptimusError):
            resolve_start_set("imaginary", 7, None)

    def test_missing_zinc_raises_an_actionable_error(self, tmp_path) -> None:
        with pytest.raises(DrOptimusError) as excinfo:
            resolve_start_set("zinc-sample", 5, None, data_dir=tmp_path)
        assert "download" in str(excinfo.value)


class TestInvalidInputHandling:
    def test_bad_smiles_in_train_exits_with_code_two(self, capsys) -> None:
        code = main(TINY_TRAIN + ["--start", "C(((("])
        captured = capsys.readouterr()
        assert code == 2
        assert "error:" in captured.err
        assert "C((((" in captured.err

    def test_bad_smiles_in_optimize_exits_with_code_two(self, capsys) -> None:
        code = main(["optimize", "nonsense!!", "--checkpoint", "missing.pt"])
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_missing_checkpoint_exits_with_code_two(self, capsys) -> None:
        code = main(["optimize", "CCO", "--checkpoint", "nowhere/absent.pt"])
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_unknown_objective_exits_with_code_two(self, capsys) -> None:
        code = main(TINY_TRAIN + ["--objective", "nope"])
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_bad_atom_types_exit_with_code_two(self, capsys) -> None:
        code = main(TINY_TRAIN + ["--atom-types", "C,Zz"])
        assert code == 2
        assert "error:" in capsys.readouterr().err


@pytest.mark.slow
class TestEndToEnd:
    def test_train_then_optimize_then_evaluate(self, tmp_path, capsys) -> None:
        out = tmp_path / "run"
        assert main(TINY_TRAIN + ["--out", str(out), "--start", "C"]) == 0
        assert (out / "checkpoint.pt").exists()
        capsys.readouterr()

        checkpoint = str(out / "checkpoint.pt")
        assert main(["optimize", "CCO", "--checkpoint", checkpoint, "--attempts", "2",
                     "--device", "cpu", "--show-trajectory"]) == 0
        optimize_output = capsys.readouterr().out
        assert "best" in optimize_output and "trajectory" in optimize_output

        metrics_path = tmp_path / "metrics.json"
        assert (
            main(
                [
                    "evaluate",
                    "--checkpoint",
                    checkpoint,
                    "--start-set",
                    "fixture",
                    "--episodes",
                    "3",
                    "--device",
                    "cpu",
                    "--novelty-reference-size",
                    "0",
                    "--out",
                    str(metrics_path),
                ]
            )
            == 0
        )
        evaluate_output = capsys.readouterr().out
        assert "trained agent" in evaluate_output
        assert "random-edit baseline" in evaluate_output
        assert "published MolDQN" in evaluate_output

        payload = json.loads(metrics_path.read_text())
        assert payload["agent"]["n"] == 3
        assert "random_baseline" in payload

    def test_evaluate_can_skip_the_baseline(self, tmp_path, capsys) -> None:
        out = tmp_path / "run"
        main(TINY_TRAIN + ["--out", str(out)])
        capsys.readouterr()
        assert (
            main(
                [
                    "evaluate",
                    "--checkpoint",
                    str(out / "checkpoint.pt"),
                    "--start-set",
                    "fixture",
                    "--episodes",
                    "2",
                    "--device",
                    "cpu",
                    "--novelty-reference-size",
                    "0",
                    "--no-baseline",
                ]
            )
            == 0
        )
        assert "random-edit baseline" not in capsys.readouterr().out
