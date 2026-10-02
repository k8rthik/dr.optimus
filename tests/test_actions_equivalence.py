"""The optimized enumerator must return exactly the reference action set.

`bond_additions` and `bond_removals` avoid copying the molecule for candidates
that cheap checks can already reject. That is a performance change only, so it
is pinned against a deliberately naive reference implementation that copies and
kekulizes per candidate, the way the first version of this module did.
"""

from __future__ import annotations

import itertools
from unittest import mock

import pytest
from rdkit import Chem

from droptimus.chem.molecule import parse_smiles
from droptimus.config import BOND_ORDERS, EnvConfig
from droptimus.env.actions import (
    _keep_largest_fragment,
    _sanitized_smiles,
    bond_additions,
    bond_removals,
    free_valence_map,
)

_BOND_TYPES = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)

MOLECULES = [
    "C",
    "CC",
    "CCO",
    "CCCCCC",
    "c1ccccc1",
    "c1ccccc1C(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1CCCCC1",
    "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
    "CN1CCC[C@H]1c1cccnc1",
    "C1CC2CCC1CC2",
    "O=C(N)c1ccncc1",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "C#CCO",
    "CC=CC=CC",
]

CONFIGS = [
    EnvConfig(atom_types=("C", "N", "O")),
    EnvConfig(atom_types=("C",), allowed_ring_sizes=(3, 4, 5)),
    EnvConfig(atom_types=("C", "N", "O"), allowed_ring_sizes=()),
    EnvConfig(atom_types=("C", "N", "O"), allow_removal_disconnect=True),
]


# The reference deliberately shares the production candidate-acceptance helper.
# What is being pinned is which candidates each enumerator reaches and when it
# copies the molecule -- not how an accepted candidate is turned into a string.
_sanitized = _sanitized_smiles


def reference_bond_additions(mol: Chem.Mol, config: EnvConfig) -> frozenset[str]:
    """Naive version: copy and kekulize the molecule for every candidate pair."""
    free_valences = free_valence_map(mol)
    results: set[str] = set()
    for order in BOND_ORDERS:
        for first, second in itertools.combinations(free_valences[order], 2):
            editable = Chem.RWMol(mol)
            try:
                Chem.Kekulize(editable, clearAromaticFlags=True)
            except Chem.KekulizeException:
                continue
            existing = editable.GetBondBetweenAtoms(first, second)
            if existing is not None:
                if existing.GetBondType() not in _BOND_TYPES:
                    continue
                new_order = _BOND_TYPES.index(existing.GetBondType()) + order
                if new_order >= len(_BOND_TYPES):
                    continue
                existing.SetBondType(_BOND_TYPES[new_order])
            else:
                if config.allowed_ring_sizes:
                    path = Chem.rdmolops.GetShortestPath(mol, first, second)
                    if path and len(path) not in config.allowed_ring_sizes:
                        continue
                editable.AddBond(first, second, _BOND_TYPES[order])
            smiles = _sanitized(editable)
            if smiles is not None:
                results.add(smiles)
    return frozenset(results)


def reference_bond_removals(mol: Chem.Mol, config: EnvConfig) -> frozenset[str]:
    """Naive version: copy and kekulize the molecule for every candidate bond."""
    if not config.allow_bond_removal:
        return frozenset()
    results: set[str] = set()
    endpoints = tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    )
    for order in BOND_ORDERS:
        for begin, end in endpoints:
            editable = Chem.RWMol(mol)
            try:
                Chem.Kekulize(editable, clearAromaticFlags=True)
            except Chem.KekulizeException:
                continue
            bond = editable.GetBondBetweenAtoms(begin, end)
            if bond is None or bond.GetBondType() not in _BOND_TYPES:
                continue
            new_order = _BOND_TYPES.index(bond.GetBondType()) - order
            if new_order < 0:
                continue
            if new_order == 0:
                editable.RemoveBond(begin, end)
            else:
                bond.SetBondType(_BOND_TYPES[new_order])
            smiles = _sanitized(editable)
            if smiles is None:
                continue
            kept = _keep_largest_fragment(smiles, config.allow_removal_disconnect)
            if kept:
                results.add(kept)
    return frozenset(results)


@pytest.mark.parametrize("smiles", MOLECULES)
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"ring{c.allowed_ring_sizes}")
class TestEquivalence:
    def test_bond_additions_match_the_reference(
        self, smiles: str, config: EnvConfig
    ) -> None:
        mol = parse_smiles(smiles)
        assert bond_additions(mol, config) == reference_bond_additions(mol, config)

    def test_bond_removals_match_the_reference(
        self, smiles: str, config: EnvConfig
    ) -> None:
        mol = parse_smiles(smiles)
        assert bond_removals(mol, config) == reference_bond_removals(mol, config)


class TestEnumerationCost:
    """Pins the optimization itself, by counting molecule copies.

    Wall-clock timing is too noisy to assert on, but the number of RWMol
    constructions is exact: the whole point of the change is that a candidate
    rejected by a cheap check never causes a copy.
    """

    def _count_copies(self, enumerate_fn, mol, config) -> int:
        real_rwmol = Chem.RWMol
        calls = 0

        def counting_rwmol(*args: object, **kwargs: object) -> Chem.RWMol:
            nonlocal calls
            calls += 1
            return real_rwmol(*args, **kwargs)

        with mock.patch.object(Chem, "RWMol", counting_rwmol):
            enumerate_fn(mol, config)
        return calls

    def test_optimized_enumeration_copies_fewer_molecules(self) -> None:
        mol = parse_smiles("CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1")
        config = CONFIGS[0]
        optimized = self._count_copies(bond_additions, mol, config)
        naive = self._count_copies(reference_bond_additions, mol, config)
        assert optimized < naive, f"optimized={optimized} naive={naive}"

    def test_saving_grows_with_molecule_size(self) -> None:
        config = CONFIGS[0]
        small = parse_smiles("CCCCCC")
        large = parse_smiles("C" * 30)
        small_ratio = self._count_copies(
            bond_additions, small, config
        ) / self._count_copies(reference_bond_additions, small, config)
        large_ratio = self._count_copies(
            bond_additions, large, config
        ) / self._count_copies(reference_bond_additions, large, config)
        assert large_ratio <= small_ratio
