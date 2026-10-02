"""Tests for benchmark molecule sets. No test touches the network."""

from __future__ import annotations

import pytest

from droptimus.datasets import (
    DatasetError,
    load_fixture,
    load_zinc_smiles,
    lowest_scoring,
    read_smiles,
    sample_molecules,
    write_smiles,
    zinc_path,
)
from droptimus.errors import InvalidSmilesError
from droptimus.objectives.logp import penalized_logp


class TestFixture:
    def test_has_the_expected_size(self) -> None:
        assert len(load_fixture()) == 200

    def test_every_molecule_is_valid_and_canonical(self) -> None:
        from droptimus.chem.molecule import canonical_smiles

        for smiles in load_fixture():
            assert smiles == canonical_smiles(smiles)

    def test_molecules_are_drug_sized(self) -> None:
        from droptimus.chem.molecule import atom_count

        sizes = [atom_count(s) for s in load_fixture()]
        assert min(sizes) >= 10 and max(sizes) <= 60


class TestReadWrite:
    def test_round_trip(self, tmp_path) -> None:
        molecules = ("CCO", "c1ccccc1")
        path = write_smiles(molecules, tmp_path / "set.smi")
        assert read_smiles(path) == molecules

    def test_skips_blanks_and_comments(self, tmp_path) -> None:
        path = tmp_path / "set.smi"
        path.write_text("# header\nCCO\n\n  \nc1ccccc1\n")
        assert read_smiles(path) == ("CCO", "c1ccccc1")

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(DatasetError):
            read_smiles(tmp_path / "absent.smi")

    def test_empty_file_raises(self, tmp_path) -> None:
        path = tmp_path / "empty.smi"
        path.write_text("\n\n")
        with pytest.raises(DatasetError):
            read_smiles(path)

    def test_invalid_line_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.smi"
        path.write_text("CCO\nnot-a-molecule\n")
        with pytest.raises(InvalidSmilesError):
            read_smiles(path)


class TestSelection:
    def test_lowest_scoring_picks_the_worst(self) -> None:
        molecules = load_fixture()
        worst = lowest_scoring(molecules, penalized_logp, count=5)
        assert len(worst) == 5
        threshold = max(penalized_logp(s) for s in worst)
        others = [s for s in molecules if s not in worst]
        assert min(penalized_logp(s) for s in others) >= threshold

    def test_lowest_scoring_rejects_bad_count(self) -> None:
        with pytest.raises(ValueError):
            lowest_scoring(load_fixture(), penalized_logp, count=0)

    def test_sample_is_reproducible(self) -> None:
        molecules = load_fixture()
        assert sample_molecules(molecules, 10, seed=3) == sample_molecules(
            molecules, 10, seed=3
        )

    def test_different_seeds_differ(self) -> None:
        molecules = load_fixture()
        assert sample_molecules(molecules, 20, seed=1) != sample_molecules(
            molecules, 20, seed=2
        )

    def test_oversampling_raises(self) -> None:
        with pytest.raises(ValueError):
            sample_molecules(load_fixture(), 10_000)


class TestZincLoading:
    def test_missing_dataset_gives_an_actionable_error(self, tmp_path) -> None:
        with pytest.raises(DatasetError) as excinfo:
            load_zinc_smiles(tmp_path)
        assert "droptimus download" in str(excinfo.value)

    def test_path_helper(self, tmp_path) -> None:
        assert zinc_path(tmp_path).name == "zinc250k.csv"

    def test_rejects_a_csv_without_a_smiles_column(self, tmp_path) -> None:
        (tmp_path / "zinc250k.csv").write_text("a,b\n1,2\n")
        with pytest.raises(DatasetError):
            load_zinc_smiles(tmp_path)

    def test_skips_unparseable_rows(self, tmp_path) -> None:
        (tmp_path / "zinc250k.csv").write_text(
            "smiles,logP\nCCO,1.0\nnot-a-molecule,2.0\nc1ccccc1,3.0\n"
        )
        assert load_zinc_smiles(tmp_path) == ("CCO", "c1ccccc1")

    def test_limit_is_respected(self, tmp_path) -> None:
        (tmp_path / "zinc250k.csv").write_text(
            "smiles\nCCO\nc1ccccc1\nCCN\n"
        )
        assert len(load_zinc_smiles(tmp_path, limit=2)) == 2

    def test_all_rows_invalid_raises(self, tmp_path) -> None:
        (tmp_path / "zinc250k.csv").write_text("smiles\nnope\nalso-nope\n")
        with pytest.raises(DatasetError):
            load_zinc_smiles(tmp_path)
