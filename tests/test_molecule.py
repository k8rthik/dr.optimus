"""Tests for the chemistry helpers: parsing, validation, similarity."""

from __future__ import annotations

import pytest

from droptimus.chem.molecule import (
    atom_count,
    canonical_smiles,
    is_valid_smiles,
    parse_smiles,
    tanimoto_similarity,
)
from droptimus.errors import InvalidSmilesError


class TestParseSmiles:
    def test_parses_simple_molecule(self) -> None:
        mol = parse_smiles("CCO")
        assert mol.GetNumAtoms() == 3

    def test_parses_aromatic_molecule(self) -> None:
        mol = parse_smiles("c1ccccc1")
        assert mol.GetNumAtoms() == 6
        assert all(a.GetIsAromatic() for a in mol.GetAtoms())

    @pytest.mark.parametrize("bad", ["", "   ", "not-a-smiles", "C(((", "c1cccc1"])
    def test_rejects_invalid(self, bad: str) -> None:
        with pytest.raises(InvalidSmilesError):
            parse_smiles(bad)

    def test_rejects_bad_valence(self) -> None:
        # Five bonds on a neutral carbon: parses structurally, fails sanitization.
        with pytest.raises(InvalidSmilesError):
            parse_smiles("C(C)(C)(C)(C)C")

    def test_error_names_the_input(self) -> None:
        with pytest.raises(InvalidSmilesError) as excinfo:
            parse_smiles("zzz")
        assert "zzz" in str(excinfo.value)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(InvalidSmilesError):
            parse_smiles(None)  # type: ignore[arg-type]


class TestCanonicalSmiles:
    def test_equivalent_inputs_agree(self) -> None:
        assert canonical_smiles("OCC") == canonical_smiles("CCO")

    def test_kekule_and_aromatic_benzene_agree(self) -> None:
        assert canonical_smiles("C1=CC=CC=C1") == canonical_smiles("c1ccccc1")

    def test_strips_whitespace(self) -> None:
        assert canonical_smiles("  CCO  ") == canonical_smiles("CCO")

    def test_invalid_raises(self) -> None:
        with pytest.raises(InvalidSmilesError):
            canonical_smiles("Q")


class TestIsValidSmiles:
    def test_true_for_valid(self) -> None:
        assert is_valid_smiles("CCO") is True

    def test_false_for_invalid(self) -> None:
        assert is_valid_smiles("c1cccc1") is False

    def test_false_for_empty(self) -> None:
        assert is_valid_smiles("") is False


class TestTanimotoSimilarity:
    def test_identical_is_one(self) -> None:
        assert tanimoto_similarity("CCO", "OCC") == pytest.approx(1.0)

    def test_different_is_below_one(self) -> None:
        sim = tanimoto_similarity("CCO", "c1ccccc1C(=O)NC2CCCCC2")
        assert 0.0 <= sim < 1.0

    def test_symmetric(self) -> None:
        a, b = "CCOc1ccccc1", "CCN(CC)CC"
        assert tanimoto_similarity(a, b) == pytest.approx(tanimoto_similarity(b, a))

    def test_invalid_raises(self) -> None:
        with pytest.raises(InvalidSmilesError):
            tanimoto_similarity("CCO", "???")


class TestAtomCount:
    def test_counts_heavy_atoms_only(self) -> None:
        assert atom_count("CCO") == 3
        assert atom_count("C") == 1
