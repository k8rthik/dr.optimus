"""Tests for molecule featurization (fingerprints and graph tensors)."""

from __future__ import annotations

import numpy as np
import pytest

from droptimus.chem.features import (
    ATOM_FEATURE_DIM,
    fingerprint_matrix,
    graph_batch,
    graph_tensors,
    packed_fingerprint,
)
from droptimus.config import BOND_CHANNELS, MORGAN_BITS
from droptimus.errors import InvalidSmilesError


class TestPackedFingerprint:
    def test_packs_to_one_eighth_the_bits(self) -> None:
        packed = packed_fingerprint("CCO")
        assert packed.dtype == np.uint8
        assert packed.shape == (MORGAN_BITS // 8,)

    def test_identical_molecules_pack_identically(self) -> None:
        assert np.array_equal(packed_fingerprint("CCO"), packed_fingerprint("OCC"))

    def test_different_molecules_differ(self) -> None:
        assert not np.array_equal(
            packed_fingerprint("CCO"), packed_fingerprint("c1ccccc1")
        )


class TestFingerprintMatrix:
    def test_shape_and_dtype(self) -> None:
        matrix = fingerprint_matrix(["CCO", "c1ccccc1", "CC"])
        assert matrix.shape == (3, MORGAN_BITS)
        assert matrix.dtype == np.float32

    def test_values_are_binary(self) -> None:
        matrix = fingerprint_matrix(["CCOc1ccccc1"])
        assert set(np.unique(matrix)).issubset({0.0, 1.0})

    def test_unpacking_round_trips_the_bit_count(self) -> None:
        from droptimus.chem.molecule import morgan_fingerprint

        expected = morgan_fingerprint("CCOc1ccccc1C(=O)O").GetNumOnBits()
        assert fingerprint_matrix(["CCOc1ccccc1C(=O)O"]).sum() == pytest.approx(expected)

    def test_empty_input_gives_empty_matrix(self) -> None:
        matrix = fingerprint_matrix([])
        assert matrix.shape == (0, MORGAN_BITS)

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(InvalidSmilesError):
            fingerprint_matrix(["CCO", "!!!"])

    def test_row_order_follows_input_order(self) -> None:
        matrix = fingerprint_matrix(["CCO", "c1ccccc1"])
        single = fingerprint_matrix(["c1ccccc1"])
        assert np.array_equal(matrix[1], single[0])


class TestGraphTensors:
    def test_shapes(self) -> None:
        nodes, adjacency, mask = graph_tensors("CCO", max_atoms=8)
        assert nodes.shape == (8, ATOM_FEATURE_DIM)
        assert adjacency.shape == (len(BOND_CHANNELS), 8, 8)
        assert mask.shape == (8,)

    def test_mask_marks_real_atoms_only(self) -> None:
        _, _, mask = graph_tensors("CCO", max_atoms=8)
        assert mask.sum() == 3
        assert mask[:3].all() and not mask[3:].any()

    def test_padding_rows_are_zero(self) -> None:
        nodes, adjacency, _ = graph_tensors("CCO", max_atoms=8)
        assert not nodes[3:].any()
        assert not adjacency[:, 3:, :].any()

    def test_adjacency_is_symmetric(self) -> None:
        _, adjacency, _ = graph_tensors("c1ccccc1C(=O)O", max_atoms=12)
        for channel in adjacency:
            assert np.array_equal(channel, channel.T)

    def test_bond_orders_land_in_the_right_channel(self) -> None:
        single = BOND_CHANNELS.index("SINGLE")
        double = BOND_CHANNELS.index("DOUBLE")
        _, adjacency, _ = graph_tensors("CC=O", max_atoms=4)
        assert adjacency[single].sum() == 2  # one C-C bond, both directions
        assert adjacency[double].sum() == 2  # one C=O bond, both directions

    def test_aromatic_bonds_use_the_aromatic_channel(self) -> None:
        aromatic = BOND_CHANNELS.index("AROMATIC")
        _, adjacency, _ = graph_tensors("c1ccccc1", max_atoms=8)
        assert adjacency[aromatic].sum() == 12  # six aromatic bonds, both directions

    def test_too_many_atoms_raises(self) -> None:
        with pytest.raises(ValueError):
            graph_tensors("C" * 20, max_atoms=5)

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(InvalidSmilesError):
            graph_tensors("zz", max_atoms=8)

    def test_element_one_hot_distinguishes_carbon_from_oxygen(self) -> None:
        nodes, _, _ = graph_tensors("CO", max_atoms=4)
        assert not np.array_equal(nodes[0], nodes[1])


class TestGraphBatch:
    def test_batches_to_common_width(self) -> None:
        nodes, adjacency, mask = graph_batch(["C", "CCO", "c1ccccc1"])
        assert nodes.shape[0] == 3
        assert adjacency.shape[0] == 3
        assert mask.shape == (3, nodes.shape[1])

    def test_width_is_the_largest_molecule(self) -> None:
        nodes, _, _ = graph_batch(["C", "CCO"])
        assert nodes.shape[1] == 3

    def test_empty_batch_is_handled(self) -> None:
        nodes, adjacency, mask = graph_batch([])
        assert nodes.shape[0] == 0
        assert adjacency.shape[0] == 0
        assert mask.shape[0] == 0

    def test_matches_single_molecule_featurization(self) -> None:
        nodes, _, _ = graph_batch(["CCO"])
        single_nodes, _, _ = graph_tensors("CCO", max_atoms=3)
        assert np.array_equal(nodes[0], single_nodes)
