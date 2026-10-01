"""Device selection and seeding."""

from __future__ import annotations

import random

import numpy as np
import torch

from droptimus.errors import ConfigError

VALID_DEVICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(requested: str = "auto") -> torch.device:
    """Return the torch device to use.

    ``auto`` prefers Apple MPS, then CUDA, then CPU.

    Raises:
        ConfigError: if an unknown device is named, or a named accelerator is
            unavailable on this machine.
    """
    name = (requested or "auto").strip().lower()
    if name not in VALID_DEVICES:
        raise ConfigError(
            f"Unknown device {requested!r}; choose one of {', '.join(VALID_DEVICES)}."
        )
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ConfigError(
            "MPS was requested but is not available on this machine; use --device cpu."
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ConfigError(
            "CUDA was requested but is not available on this machine; use --device cpu."
        )
    return torch.device(name)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and torch so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on this machine
        torch.cuda.manual_seed_all(seed)
