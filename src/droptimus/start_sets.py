"""Choosing the molecules a run starts episodes from.

A "start set" is a named collection of start molecules. Which one a run uses
changes what the agent can learn: starting every episode from a single carbon
atom teaches the agent to *build* a molecule, while starting from a set of
drug-sized molecules teaches it to *edit* one. The novelty reference set lives
here too, because it is the same question asked at evaluation time: which
molecules count as already known.

Kept out of the CLI module so that the choice is usable from a notebook or a
script without going through argparse.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem

from droptimus.chem.molecule import canonical_smiles
from droptimus.datasets import (
    DEFAULT_DATA_DIR,
    load_fixture,
    load_zinc_smiles,
    sample_molecules,
    zinc800_logp,
)
from droptimus.errors import DrOptimusError

LOGGER = logging.getLogger("droptimus.start_sets")

#: The start sets a run may name.
START_SET_CHOICES = ("single", "fixture", "zinc-sample", "zinc800-logp")


def parse_atom_types(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated element list such as ``"C,N,O"``.

    Raises:
        DrOptimusError: if the list is empty or names something RDKit does not
            recognize as an element.
    """
    elements = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not elements:
        raise DrOptimusError(
            "--atom-types must name at least one element, e.g. 'C,N,O'."
        )
    table = Chem.GetPeriodicTable()
    for element in elements:
        try:
            table.GetAtomicNumber(element)
        except RuntimeError as exc:
            raise DrOptimusError(
                f"{element!r} in --atom-types is not an element symbol."
            ) from exc
    return elements


def resolve_start_set(
    kind: str,
    count: int,
    single: str | None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    seed: int = 0,
) -> tuple[str, ...]:
    """Return the start molecules named by ``kind``.

    Args:
        kind: one of :data:`START_SET_CHOICES`.
        count: how many molecules to take, where the set is larger.
        single: the molecule to use when ``kind`` is ``"single"``.
        data_dir: where ZINC250k lives, for the sets derived from it.
        seed: makes ``"zinc-sample"`` reproducible.

    Raises:
        DrOptimusError: for an unknown kind, a missing start molecule, or a
            dataset that has not been downloaded.
    """
    if kind == "single":
        if not single:
            raise DrOptimusError("--start-set single needs --start '<SMILES>'.")
        return (canonical_smiles(single),)
    if kind == "fixture":
        molecules = load_fixture()
        return molecules[: min(count, len(molecules))]
    if kind == "zinc-sample":
        molecules = load_zinc_smiles(data_dir)
        return sample_molecules(molecules, min(count, len(molecules)), seed=seed)
    if kind == "zinc800-logp":
        LOGGER.info(
            "loading the %d lowest-penalized-logP ZINC molecules "
            "(ranking all 249k takes ~2 minutes the first time, then it is cached)",
            count,
        )
        return zinc800_logp(data_dir, count=count)
    raise DrOptimusError(
        f"Unknown start set {kind!r}; choose one of {', '.join(START_SET_CHOICES)}."
    )


def novelty_reference(
    size: int, data_dir: str | Path = DEFAULT_DATA_DIR
) -> tuple[str, ...]:
    """Return the molecules that count as already known, for novelty.

    Prefers ZINC250k. If it has not been downloaded, falls back to the
    200-molecule sample shipped with the package and says so, because silently
    measuring novelty against a 200-molecule reference would make novelty look
    far better than it is.
    """
    if size < 1:
        return ()
    try:
        return load_zinc_smiles(data_dir, limit=size)
    except DrOptimusError:
        LOGGER.warning(
            "ZINC250k is not available; measuring novelty against the "
            "200-molecule sample shipped with the package instead, which will "
            "overstate novelty. Run `droptimus download` for the full set."
        )
        return load_fixture()
