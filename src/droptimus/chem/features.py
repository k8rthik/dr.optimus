"""Turning molecules into tensors.

Two representations, matching the two Q-network variants:

* **Morgan fingerprints** --- the representation used by MolDQN itself. Bit
  vectors are cached in *packed* uint8 form (256 bytes for 2048 bits) so that a
  replay buffer full of candidate molecules stays small; a batch is unpacked
  with one vectorized call.
* **Graph tensors** --- padded node-feature and typed-adjacency arrays for the
  relational GNN encoder.

Both featurizers are pure functions of a SMILES string and are memoized.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs

from droptimus.chem.molecule import morgan_fingerprint, parse_smiles
from droptimus.config import BOND_CHANNELS, FEATURIZED_ELEMENTS, MORGAN_BITS

# --- Fingerprints ----------------------------------------------------------

_PACKED_BYTES = MORGAN_BITS // 8


@lru_cache(maxsize=400_000)
def packed_fingerprint(smiles: str) -> np.ndarray:
    """Return the Morgan fingerprint of ``smiles`` as packed bits (uint8)."""
    bitvect = morgan_fingerprint(smiles)
    dense = np.zeros((MORGAN_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, dense)
    packed = np.packbits(dense)
    packed.flags.writeable = False
    return packed


def fingerprint_matrix(smiles: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return an ``(n, MORGAN_BITS)`` float32 matrix of fingerprints.

    Raises:
        InvalidSmilesError: if any entry fails to parse.
    """
    if len(smiles) == 0:
        return np.zeros((0, MORGAN_BITS), dtype=np.float32)
    packed = np.stack([packed_fingerprint(s) for s in smiles])
    unpacked = np.unpackbits(packed, axis=1, count=MORGAN_BITS)
    return unpacked.astype(np.float32)


# --- Graph tensors ---------------------------------------------------------

_DEGREE_BUCKETS = (0, 1, 2, 3, 4)
_HYDROGEN_BUCKETS = (0, 1, 2, 3, 4)

#: element one-hot (+ "other") | degree one-hot | implicit-H one-hot | 3 flags
ATOM_FEATURE_DIM = (
    len(FEATURIZED_ELEMENTS) + 1 + len(_DEGREE_BUCKETS) + len(_HYDROGEN_BUCKETS) + 3
)

_ELEMENT_INDEX = {symbol: i for i, symbol in enumerate(FEATURIZED_ELEMENTS)}
_OTHER_ELEMENT_INDEX = len(FEATURIZED_ELEMENTS)
_BOND_CHANNEL_INDEX = {name: i for i, name in enumerate(BOND_CHANNELS)}


def _one_hot_into(row: np.ndarray, offset: int, value: int, buckets: tuple[int, ...]) -> int:
    """Write a one-hot of ``value`` at ``offset``; return the next free offset."""
    if value in buckets:
        row[offset + buckets.index(value)] = 1.0
    return offset + len(buckets)


def _atom_features(atom: Chem.Atom) -> np.ndarray:
    row = np.zeros((ATOM_FEATURE_DIM,), dtype=np.float32)
    row[_ELEMENT_INDEX.get(atom.GetSymbol(), _OTHER_ELEMENT_INDEX)] = 1.0
    offset = len(FEATURIZED_ELEMENTS) + 1
    offset = _one_hot_into(row, offset, atom.GetDegree(), _DEGREE_BUCKETS)
    offset = _one_hot_into(row, offset, atom.GetNumImplicitHs(), _HYDROGEN_BUCKETS)
    row[offset] = float(atom.GetFormalCharge())
    row[offset + 1] = 1.0 if atom.GetIsAromatic() else 0.0
    row[offset + 2] = 1.0 if atom.IsInRing() else 0.0
    return row


@lru_cache(maxsize=100_000)
def _graph_arrays(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    """Return unpadded ``(nodes, adjacency)`` for ``smiles``."""
    mol = parse_smiles(smiles)
    n_atoms = mol.GetNumAtoms()
    nodes = np.zeros((n_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    for atom in mol.GetAtoms():
        nodes[atom.GetIdx()] = _atom_features(atom)

    adjacency = np.zeros((len(BOND_CHANNELS), n_atoms, n_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        name = bond.GetBondType().name
        channel = _BOND_CHANNEL_INDEX.get(name)
        if channel is None:
            # Any exotic bond type is treated as a single bond rather than
            # silently dropped, so connectivity is never lost.
            channel = _BOND_CHANNEL_INDEX["SINGLE"]
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[channel, i, j] = 1.0
        adjacency[channel, j, i] = 1.0

    nodes.flags.writeable = False
    adjacency.flags.writeable = False
    return nodes, adjacency


def graph_tensors(smiles: str, max_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(nodes, adjacency, mask)`` padded to ``max_atoms``.

    Shapes are ``(max_atoms, ATOM_FEATURE_DIM)``,
    ``(len(BOND_CHANNELS), max_atoms, max_atoms)`` and ``(max_atoms,)``.

    Raises:
        ValueError: if the molecule has more atoms than ``max_atoms``.
        InvalidSmilesError: if ``smiles`` does not parse.
    """
    nodes, adjacency = _graph_arrays(smiles)
    n_atoms = nodes.shape[0]
    if n_atoms > max_atoms:
        raise ValueError(
            f"{smiles!r} has {n_atoms} atoms, more than the padding width "
            f"{max_atoms}. Raise max_atoms (or FEATURIZER_MAX_ATOMS)."
        )
    padded_nodes = np.zeros((max_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    padded_nodes[:n_atoms] = nodes
    padded_adjacency = np.zeros(
        (len(BOND_CHANNELS), max_atoms, max_atoms), dtype=np.float32
    )
    padded_adjacency[:, :n_atoms, :n_atoms] = adjacency
    mask = np.zeros((max_atoms,), dtype=bool)
    mask[:n_atoms] = True
    return padded_nodes, padded_adjacency, mask


def graph_batch(
    smiles: list[str] | tuple[str, ...], max_atoms: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch-featurize molecules, padding to the widest one (or ``max_atoms``)."""
    if len(smiles) == 0:
        return (
            np.zeros((0, 0, ATOM_FEATURE_DIM), dtype=np.float32),
            np.zeros((0, len(BOND_CHANNELS), 0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
        )
    width = max_atoms or max(_graph_arrays(s)[0].shape[0] for s in smiles)
    triples = [graph_tensors(s, width) for s in smiles]
    nodes = np.stack([t[0] for t in triples])
    adjacency = np.stack([t[1] for t in triples])
    mask = np.stack([t[2] for t in triples])
    return nodes, adjacency, mask


def clear_feature_caches() -> None:
    """Drop memoized features. Used by tests and long-running processes."""
    packed_fingerprint.cache_clear()
    _graph_arrays.cache_clear()
