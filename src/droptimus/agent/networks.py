"""Q-value heads.

Both networks map a *candidate molecule* plus the remaining-step horizon to a
single scalar Q(s, a); the action space is state-dependent, so the agent scores
candidates rather than emitting a fixed-size action vector.

* :class:`FingerprintQNetwork` -- the MolDQN architecture: an MLP on a Morgan
  fingerprint concatenated with the horizon.
* :class:`GraphQNetwork` -- a relational graph convolution (one weight matrix
  per bond type) with masked mean+max pooling, then the same MLP head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def _mlp(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    """Build a ReLU MLP ending in a single linear output unit."""
    layers: list[nn.Module] = []
    width = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(width, size))
        layers.append(nn.ReLU())
        width = size
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Interface: consume a featurizer batch, return one Q value per molecule."""

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class FingerprintQNetwork(QNetwork):
    """MLP over [Morgan fingerprint | horizon]."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.mlp = _mlp(input_dim, hidden_sizes)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        features = torch.cat([batch["fingerprints"], batch["horizon"]], dim=1)
        return self.mlp(features).squeeze(-1)


class RelationalGraphLayer(nn.Module):
    """One R-GCN layer: a self transform plus one transform per bond type."""

    def __init__(self, input_dim: int, output_dim: int, n_relations: int) -> None:
        super().__init__()
        self.self_transform = nn.Linear(input_dim, output_dim)
        self.relation_transforms = nn.ModuleList(
            nn.Linear(input_dim, output_dim, bias=False) for _ in range(n_relations)
        )

    def forward(
        self, nodes: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # nodes: (B, A, F); adjacency: (B, R, A, A); mask: (B, A)
        out = self.self_transform(nodes)
        for relation, transform in enumerate(self.relation_transforms):
            messages = transform(nodes)  # (B, A, F')
            out = out + torch.bmm(adjacency[:, relation], messages)
        out = torch.relu(out)
        return out * mask.unsqueeze(-1).to(out.dtype)


class GraphQNetwork(QNetwork):
    """Relational GNN encoder with a Q-value MLP head."""

    def __init__(
        self,
        node_dim: int,
        n_bond_channels: int,
        hidden_dim: int,
        n_layers: int,
        head_hidden: Sequence[int],
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        dims = [node_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            RelationalGraphLayer(dims[i], dims[i + 1], n_bond_channels)
            for i in range(n_layers)
        )
        # mean pool + max pool + horizon
        self.head = _mlp(2 * hidden_dim + 1, head_hidden)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        nodes = batch["nodes"]
        adjacency = batch["adjacency"]
        mask = batch["mask"]
        hidden = nodes
        for layer in self.layers:
            hidden = layer(hidden, adjacency, mask)

        float_mask = mask.unsqueeze(-1).to(hidden.dtype)
        counts = float_mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (hidden * float_mask).sum(dim=1) / counts
        very_negative = torch.finfo(hidden.dtype).min
        max_pool = hidden.masked_fill(~mask.unsqueeze(-1), very_negative).max(dim=1).values
        # A molecule with no atoms cannot occur, but guard the fill value anyway.
        max_pool = torch.where(
            counts > 0, max_pool, torch.zeros_like(max_pool)
        )
        pooled = torch.cat([mean_pool, max_pool, batch["horizon"]], dim=1)
        return self.head(pooled).squeeze(-1)


def build_network(encoder: str, featurizer: object, hidden_sizes: Sequence[int],
                  gnn_hidden: int, gnn_layers: int) -> QNetwork:
    """Construct the Q-network matching ``encoder``.

    Raises:
        ValueError: for an unknown encoder name.
    """
    normalized = encoder.strip().lower()
    if normalized == "fingerprint":
        return FingerprintQNetwork(
            input_dim=featurizer.input_dim,  # type: ignore[attr-defined]
            hidden_sizes=hidden_sizes,
        )
    if normalized == "gnn":
        return GraphQNetwork(
            node_dim=featurizer.node_dim,  # type: ignore[attr-defined]
            n_bond_channels=featurizer.n_bond_channels,  # type: ignore[attr-defined]
            hidden_dim=gnn_hidden,
            n_layers=gnn_layers,
            head_hidden=hidden_sizes[-2:] if len(hidden_sizes) >= 2 else hidden_sizes,
        )
    raise ValueError(f"Unknown encoder {encoder!r}; choose 'fingerprint' or 'gnn'.")
