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
from droptimus.config import BOND_ORDERS, MAX_CANONICAL_ROUND_TRIPS, EnvConfig

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
    """Return a round-trip-stable canonical SMILES for ``mol``, or None.

    Dropping candidates that do not sanitize is what keeps the action space
    chemically valid. The round trip on top of that keeps it *addressable*: the
    environment identifies an action by its canonical SMILES, so a candidate
    whose SMILES canonicalizes to a different string would be offered to the
    agent under a name the environment then refuses to accept.

    This is not hypothetical. Built from a kekulized template, the molecule
    ``NC12c3o[nH][nH]n1c32`` sanitizes cleanly but reads back as
    ``NC12C3=C1N2NNO3`` --- RDKit's aromaticity perception is not a fixed point
    of one write/read cycle for that fused ring system. Candidates that have not
    settled within ``MAX_CANONICAL_ROUND_TRIPS`` cycles are dropped.
    """
    if Chem.SanitizeMol(mol, catchErrors=True):
        return None
    smiles = mol_to_smiles(mol)
    for _ in range(MAX_CANONICAL_ROUND_TRIPS):
        reparsed = Chem.MolFromSmiles(smiles)
        if reparsed is None:
            return None
        settled = Chem.MolToSmiles(reparsed)
        if settled == smiles:
            return smiles
        smiles = settled
    return None


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


#: RDKit's distance matrix uses this sentinel for atoms in different fragments.
_DISCONNECTED_DISTANCE = 1e7


def _kekulized_template(mol: Chem.Mol) -> Chem.Mol | None:
    """Return a kekulized copy of ``mol``, or None if it cannot be kekulized.

    Built once per enumeration and copied per candidate. Copying an existing
    kekulized molecule is far cheaper than re-kekulizing for every candidate,
    and bond-order edits are undefined on aromatic bonds.
    """
    editable = Chem.RWMol(mol)
    try:
        Chem.Kekulize(editable, clearAromaticFlags=True)
    except Chem.KekulizeException:
        return None
    return editable.GetMol()


def _ring_size_allowed(
    distances: object, first: int, second: int, config: EnvConfig
) -> bool:
    """Return whether bonding ``first`` to ``second`` closes an allowed ring.

    ``distances`` is the molecule's topological distance matrix. A new bond
    closes a ring whose size is the topological distance plus one; atoms in
    different fragments close no ring at all.
    """
    if not config.allowed_ring_sizes:
        return True
    distance = distances[first][second]  # type: ignore[index]
    if distance >= _DISCONNECTED_DISTANCE:
        return True
    return int(distance) + 1 in config.allowed_ring_sizes


def bond_additions(mol: Chem.Mol, config: EnvConfig) -> frozenset[str]:
    """Enumerate molecules reachable by adding a bond or raising a bond order.

    The candidate loop is quadratic in atom count, so every cheap rejection is
    made *before* the molecule is copied: pairs are screened against the
    precomputed distance matrix and the existing bond's order, and only
    survivors are materialized and sanitized.
    """
    template = _kekulized_template(mol)
    if template is None:
        return frozenset()
    free_valences = free_valence_map(mol)
    distances = Chem.rdmolops.GetDistanceMatrix(mol)
    # Existing bond orders, read once from the kekulized template.
    existing_order: dict[tuple[int, int], int | None] = {}
    for bond in template.GetBonds():
        key = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bond_type = bond.GetBondType()
        existing_order[key] = (
            _BOND_TYPES.index(bond_type) if bond_type in _BOND_TYPES else None
        )
        existing_order[key[::-1]] = existing_order[key]

    results: set[str] = set()
    for order in BOND_ORDERS:
        for first, second in itertools.combinations(free_valences[order], 2):
            current = existing_order.get((first, second), 0)
            if current is None:
                continue  # aromatic bonds are never edited directly
            if current > 0:
                new_order = current + order
                if new_order >= len(_BOND_TYPES):
                    continue
            elif not _ring_size_allowed(distances, first, second, config):
                continue

            editable = Chem.RWMol(template)
            if current > 0:
                bond = editable.GetBondBetweenAtoms(first, second)
                bond.SetBondType(_BOND_TYPES[current + order])
            else:
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
    template = _kekulized_template(mol)
    if template is None:
        return frozenset()
    results: set[str] = set()
    # Endpoints and orders are read once from the kekulized template, so a
    # molecule is copied only for candidates that survive the order check.
    bonds = tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        for bond in template.GetBonds()
    )
    for order in BOND_ORDERS:
        for begin, end, bond_type in bonds:
            if bond_type not in _BOND_TYPES:
                continue
            new_order = _BOND_TYPES.index(bond_type) - order
            if new_order < 0:
                continue
            editable = Chem.RWMol(template)
            bond = editable.GetBondBetweenAtoms(begin, end)
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
