"""Batch builders that turn candidate molecules into network inputs.

Both Q-network variants score *candidate successor molecules* conditioned on how
many steps of the episode remain, so every featurizer produces a dict of
tensors keyed by the argument names its network expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np
import torch

from droptimus.chem.features import ATOM_FEATURE_DIM, fingerprint_matrix, graph_batch
from droptimus.config import BOND_CHANNELS, MORGAN_BITS


class Featurizer(ABC):
    """Builds network input tensors from molecules plus a horizon."""

    #: Name recorded in checkpoints so a model is always loaded with the
    #: featurizer it was trained with.
    kind: str = "abstract"

    def __init__(self, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}.")
        self.max_steps = max_steps

    def _horizon(
        self, steps_remaining: Sequence[int], device: torch.device
    ) -> torch.Tensor:
        """Return the horizon feature, normalized to roughly [0, 1]."""
        values = np.asarray(steps_remaining, dtype=np.float32) / float(self.max_steps)
        return torch.from_numpy(values).to(device).unsqueeze(1)

    @abstractmethod
    def batch(
        self,
        smiles: Sequence[str],
        steps_remaining: Sequence[int],
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        """Return the tensors needed to score ``smiles``."""

    @staticmethod
    def _check_lengths(smiles: Sequence[str], steps_remaining: Sequence[int]) -> None:
        if len(smiles) != len(steps_remaining):
            raise ValueError(
                f"Got {len(smiles)} molecules but {len(steps_remaining)} horizons."
            )


class FingerprintFeaturizer(Featurizer):
    """Morgan-fingerprint features, as used by MolDQN itself."""

    kind = "fingerprint"

    @property
    def input_dim(self) -> int:
        return MORGAN_BITS + 1

    def batch(
        self,
        smiles: Sequence[str],
        steps_remaining: Sequence[int],
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        self._check_lengths(smiles, steps_remaining)
        fingerprints = torch.from_numpy(fingerprint_matrix(tuple(smiles))).to(device)
        return {
            "fingerprints": fingerprints,
            "horizon": self._horizon(steps_remaining, device),
        }


class GraphFeaturizer(Featurizer):
    """Padded graph tensors for the relational GNN encoder."""

    kind = "gnn"

    def __init__(self, max_steps: int, max_atoms: int) -> None:
        super().__init__(max_steps)
        if max_atoms < 1:
            raise ValueError(f"max_atoms must be >= 1, got {max_atoms}.")
        self.max_atoms = max_atoms

    @property
    def node_dim(self) -> int:
        return ATOM_FEATURE_DIM

    @property
    def n_bond_channels(self) -> int:
        return len(BOND_CHANNELS)

    def batch(
        self,
        smiles: Sequence[str],
        steps_remaining: Sequence[int],
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        self._check_lengths(smiles, steps_remaining)
        nodes, adjacency, mask = graph_batch(tuple(smiles))
        return {
            "nodes": torch.from_numpy(nodes).to(device),
            "adjacency": torch.from_numpy(adjacency).to(device),
            "mask": torch.from_numpy(mask).to(device),
            "horizon": self._horizon(steps_remaining, device),
        }


def build_featurizer(kind: str, max_steps: int, max_atoms: int) -> Featurizer:
    """Construct the featurizer named ``kind``.

    Raises:
        ValueError: for an unknown encoder name.
    """
    normalized = kind.strip().lower()
    if normalized == "fingerprint":
        return FingerprintFeaturizer(max_steps=max_steps)
    if normalized == "gnn":
        return GraphFeaturizer(max_steps=max_steps, max_atoms=max_atoms)
    raise ValueError(
        f"Unknown encoder {kind!r}; choose 'fingerprint' or 'gnn'."
    )
