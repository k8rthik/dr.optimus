"""SMILES parsing, validation and similarity.

Every entry point into the package that accepts a molecule string funnels
through :func:`parse_smiles`, so an invalid molecule always surfaces as an
:class:`~droptimus.errors.InvalidSmilesError` with the offending string
attached.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from droptimus.config import MORGAN_BITS, MORGAN_RADIUS
from droptimus.errors import InvalidSmilesError

# RDKit prints parse failures to stderr; we surface them as exceptions instead.
RDLogger.DisableLog("rdApp.*")

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse and sanitize ``smiles``.

    Raises:
        InvalidSmilesError: if the input is not a string, is blank, cannot be
            parsed, or fails RDKit sanitization (e.g. impossible valences).
    """
    if not isinstance(smiles, str):
        raise InvalidSmilesError(repr(smiles), "expected a string")
    stripped = smiles.strip()
    if not stripped:
        raise InvalidSmilesError(smiles, "the string is empty")

    mol = Chem.MolFromSmiles(stripped, sanitize=False)
    if mol is None:
        raise InvalidSmilesError(stripped, "RDKit could not parse the structure")
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise InvalidSmilesError(stripped, f"sanitization failed ({exc})") from exc
    return mol


def mol_to_smiles(mol: Chem.Mol) -> str:
    """Return the canonical SMILES of an already-sanitized molecule."""
    return Chem.MolToSmiles(mol)


def canonical_smiles(smiles: str) -> str:
    """Return the RDKit canonical form of ``smiles``."""
    return mol_to_smiles(parse_smiles(smiles))


def is_valid_smiles(smiles: str) -> bool:
    """Return whether ``smiles`` parses and sanitizes cleanly."""
    try:
        parse_smiles(smiles)
    except InvalidSmilesError:
        return False
    return True


def atom_count(smiles: str) -> int:
    """Return the number of heavy (non-hydrogen) atoms in ``smiles``."""
    return parse_smiles(smiles).GetNumAtoms()


@lru_cache(maxsize=200_000)
def _fingerprint_cached(smiles: str) -> DataStructs.ExplicitBitVect:
    return _MORGAN_GENERATOR.GetFingerprint(parse_smiles(smiles))


def morgan_fingerprint(smiles: str) -> DataStructs.ExplicitBitVect:
    """Return the Morgan (ECFP4-style) bit vector for ``smiles``."""
    return _fingerprint_cached(smiles)


def tanimoto_similarity(reference: str, candidate: str) -> float:
    """Return the Tanimoto similarity of two molecules' Morgan fingerprints."""
    return float(
        DataStructs.TanimotoSimilarity(
            morgan_fingerprint(reference), morgan_fingerprint(candidate)
        )
    )


def clear_caches() -> None:
    """Drop memoized fingerprints. Used by tests and long-running processes."""
    _fingerprint_cached.cache_clear()
