"""dr.optimus --- MolDQN-style molecule optimization on open benchmarks.

A reproduction of Zhou et al. (2019), "Optimization of Molecules via Deep
Reinforcement Learning" (Scientific Reports 9:10752): a double-DQN agent edits
molecular graphs under RDKit valence validation and is scored on the standard
benchmark objectives (QED, penalized logP, and a similarity-constrained
variant).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
