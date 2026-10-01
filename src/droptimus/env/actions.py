"""Valid-action enumeration for the molecule-editing MDP.

An "action" here is the *resulting* molecule, as in Zhou et al. (2019): the
agent scores candidate successor states rather than an indexed action vector,
which is what lets a single Q-network handle a state-dependent action space.

Three edit types are enumerated, each validated by RDKit sanitization so that
no action can produce an impossible valence:

1. atom addition --- attach a new atom of an allowed element with a single,
   double or triple bond to an atom that has the free valence for it;
2. bond addition --- create a new bond between two atoms that both have free
   valence (subject to a ring-size filter), or raise the order of an existing
   bond;
3. bond removal --- lower the order of an existing bond, or delete it.

All functions return frozensets of canonical SMILES and never mutate their
inputs.
"""

from __future__ import annotations

import itertools
import random
from functools import lru_cache

from rdkit import Chem

from droptimus.chem.molecule import mol_to_smiles, parse_smiles
from droptimus.config import BOND_ORDERS, EnvConfig

#: Index i of this tuple is the RDKit bond type of order i (index 0 = no bond).
_BOND_TYPES: tuple[Chem.BondType | None, ...] = (
    None,
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
)

_PERIODIC_TABLE = Chem.GetPeriodicTable()


@lru_cache(maxsize=256)
def max_valence(element: str) -> int:
    """Return the largest standard valence of ``element``."""
    valences = _PERIODIC_TABLE.GetValenceList(element)
    usable = [v for v in valences if v > 0]
    if not usable:
        return 0
    return max(usable)


def free_valence_map(mol: Chem.Mol) -> dict[int, tuple[int, ...]]:
    """Map each bond order to the atom indices with at least that free valence.

    Free valence is read from RDKit's implicit-hydrogen count, which is exactly
    the number of extra bonds the atom can accept.
    """
    return {
        order: tuple(
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetNumImplicitHs() >= order
        )
        for order in BOND_ORDERS
    }


def _sanitized_smiles(mol: Chem.RWMol) -> str | None:
    """Return the canonical SMILES of ``mol``, or None if it does not sanitize."""
    if Chem.SanitizeMol(mol, catchErrors=True):
        return None
    return mol_to_smiles(mol)


def atom_additions(mol: Chem.Mol, config: EnvConfig) -> frozenset[str]:
    """Enumerate molecules reachable by attaching one new atom."""
    if mol.GetNumAtoms() >= config.max_atoms:
        return frozenset()
    free_valences = free_valence_map(mol)
    results: set[str] = set()
    for order in BOND_ORDERS:
        bond_type = _BOND_TYPES[order]
        for atom_idx in free_valences[order]:
            for element in config.atom_types:
                if max_valence(element) < order:
                    continue
                editable = Chem.RWMol(mol)
                new_idx = editable.AddAtom(Chem.Atom(element))
                editable.AddBond(atom_idx, new_idx, bond_type)
                smiles = _sanitized_smiles(editable)
                if smiles is not None:
                    results.add(smiles)
    return frozenset(results)


def _ring_size_allowed(mol: Chem.Mol, first: int, second: int, config: EnvConfig) -> bool:
    """Return whether bonding ``first`` to ``second`` closes an allowed ring."""
    if not config.allowed_ring_sizes:
        return True
    path = Chem.rdmolops.GetShortestPath(mol, first, second)
    if not path:
        # Disconnected atoms: bonding them creates no ring at all.
        return True
    return len(path) in config.allowed_ring_sizes


def bond_additions(mol: Chem.Mol, config: EnvConfig) -> frozenset[str]:
    """Enumerate molecules reachable by adding a bond or raising a bond order."""
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
                    continue  # aromatic bonds are never edited directly
                new_order = _BOND_TYPES.index(existing.GetBondType()) + order
                if new_order >= len(_BOND_TYPES):
                    continue
                existing.SetBondType(_BOND_TYPES[new_order])
            else:
                if not _ring_size_allowed(mol, first, second, config):
                    continue
                editable.AddBond(first, second, _BOND_TYPES[order])
            smiles = _sanitized_smiles(editable)
            if smiles is not None:
                results.add(smiles)
    return frozenset(results)


def _keep_largest_fragment(smiles: str, allow_disconnect: bool) -> str | None:
    """Handle a removal that split the molecule.

    Following the reference implementation, a removal is kept only when it
    leaves a single molecule, or a molecule plus a lone atom; in the latter
    case the lone atom is dropped.
    """
    parts = sorted(smiles.split("."), key=len)
    if len(parts) == 1:
        return parts[0]
    if allow_disconnect:
        return parts[-1]
    if len(parts[0]) == 1:
        return parts[-1]
    return None


def bond_removals(mol: Chem.Mol, config: EnvConfig) -> frozenset[str]:
    """Enumerate molecules reachable by lowering or deleting one bond."""
    if not config.allow_bond_removal:
        return frozenset()
    results: set[str] = set()
    bond_endpoints = tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    )
    for order in BOND_ORDERS:
        for begin, end in bond_endpoints:
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
            smiles = _sanitized_smiles(editable)
            if smiles is None:
                continue
            kept = _keep_largest_fragment(smiles, config.allow_removal_disconnect)
            if kept:
                results.add(kept)
    return frozenset(results)


@lru_cache(maxsize=100_000)
def _enumerate(smiles: str, config: EnvConfig) -> frozenset[str]:
    mol = parse_smiles(smiles)
    actions = (
        atom_additions(mol, config)
        | bond_additions(mol, config)
        | bond_removals(mol, config)
    )
    if config.allow_no_modification:
        actions = actions | {mol_to_smiles(mol)}
    return frozenset(a for a in actions if a)


def valid_actions(
    smiles: str,
    config: EnvConfig,
    rng: random.Random | None = None,
) -> frozenset[str]:
    """Return every molecule reachable from ``smiles`` in one edit.

    If ``config.max_actions`` is set, a uniform random subsample of that size is
    returned instead (drawn with ``rng`` so a run stays reproducible). The
    subsample is a wall-clock concession, not part of the published algorithm.

    Raises:
        InvalidSmilesError: if ``smiles`` does not parse.
    """
    actions = _enumerate(smiles, config)
    cap = config.max_actions
    if cap is None or len(actions) <= cap:
        return actions
    chooser = rng if rng is not None else random
    return frozenset(chooser.sample(sorted(actions), cap))


def clear_action_cache() -> None:
    """Drop the memoized action sets."""
    _enumerate.cache_clear()
