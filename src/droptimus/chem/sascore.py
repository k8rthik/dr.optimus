"""Access to RDKit's contributed synthetic-accessibility scorer.

``sascorer`` ships inside RDKit's Contrib tree rather than as an importable
module, so it has to be loaded from the filesystem. We do that once, here, and
raise a clear error if the Contrib tree is missing from the installation.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from types import ModuleType

from rdkit import Chem, RDConfig

from droptimus.errors import ObjectiveError


@lru_cache(maxsize=1)
def _load_sascorer() -> ModuleType:
    path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
    if not os.path.exists(path):
        raise ObjectiveError(
            "RDKit's Contrib SA_Score module was not found at "
            f"{path!r}. Reinstall rdkit (the wheel bundles Contrib) to use "
            "objectives that need synthetic accessibility."
        )
    spec = importlib.util.spec_from_file_location("droptimus_sascorer", path)
    if spec is None or spec.loader is None:
        raise ObjectiveError(f"Could not load the SA scorer from {path!r}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_accessibility(mol: Chem.Mol) -> float:
    """Return the Ertl--Schuffenhauer SA score (1 = easy, 10 = hard)."""
    return float(_load_sascorer().calculateScore(mol))
