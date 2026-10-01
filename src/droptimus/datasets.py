"""Benchmark molecule sets.

The reference set for this benchmark is ZINC250k --- the 249,456-molecule
"clean" ZINC subset used by Kusner et al. (2017), Jin et al. (2018), You et al.
(2018) and Zhou et al. (2019). It is downloaded on demand into a gitignored
directory; nothing large is committed.

Two derived sets matter:

* **ZINC800-logP** --- the 800 molecules with the lowest penalized logP. This is
  the start set for the similarity-constrained improvement task.
* a uniform random sample, used as start molecules for unconstrained runs and
  as the novelty reference.
"""

from __future__ import annotations

import csv
import random
import urllib.error
import urllib.request
from pathlib import Path

from droptimus.chem.molecule import canonical_smiles
from droptimus.errors import DrOptimusError, InvalidSmilesError

ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
)
DEFAULT_DATA_DIR = Path("data")
ZINC_FILENAME = "zinc250k.csv"
#: Size of the constrained-optimization start set, following the literature.
ZINC800_SIZE = 800

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "zinc_sample.smi"


class DatasetError(DrOptimusError):
    """Raised when a benchmark dataset is missing or unreadable."""


def zinc_path(data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Return the expected location of the ZINC250k CSV."""
    return Path(data_dir) / ZINC_FILENAME


def download_zinc250k(
    data_dir: Path | str = DEFAULT_DATA_DIR, force: bool = False
) -> Path:
    """Download ZINC250k if it is not already present. Returns its path.

    Raises:
        DatasetError: if the download fails.
    """
    destination = zinc_path(data_dir)
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    try:
        urllib.request.urlretrieve(ZINC_URL, temporary)  # noqa: S310 - fixed https URL
    except (urllib.error.URLError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise DatasetError(
            f"Could not download ZINC250k from {ZINC_URL}: {exc}. Download it "
            f"manually and save it as {destination}."
        ) from exc
    temporary.replace(destination)
    return destination


def load_zinc_smiles(
    data_dir: Path | str = DEFAULT_DATA_DIR, limit: int | None = None
) -> tuple[str, ...]:
    """Load canonical SMILES from the ZINC250k CSV.

    Raises:
        DatasetError: if the file is absent or has no usable ``smiles`` column.
    """
    source = zinc_path(data_dir)
    if not source.exists():
        raise DatasetError(
            f"ZINC250k is not present at {source}. Run "
            "`droptimus download` to fetch it."
        )
    molecules: list[str] = []
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "smiles" not in reader.fieldnames:
            raise DatasetError(
                f"{source} has no 'smiles' column (found {reader.fieldnames})."
            )
        for row in reader:
            raw = (row.get("smiles") or "").strip()
            if not raw:
                continue
            try:
                molecules.append(canonical_smiles(raw))
            except InvalidSmilesError:
                continue  # a handful of rows fail sanitization; skip them
            if limit is not None and len(molecules) >= limit:
                break
    if not molecules:
        raise DatasetError(f"No valid molecules were read from {source}.")
    return tuple(molecules)


def lowest_scoring(
    molecules: tuple[str, ...], score: object, count: int = ZINC800_SIZE
) -> tuple[str, ...]:
    """Return the ``count`` molecules with the lowest value of ``score``."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}.")
    ranked = sorted(molecules, key=score)  # type: ignore[arg-type]
    return tuple(ranked[:count])


def sample_molecules(
    molecules: tuple[str, ...], count: int, seed: int = 0
) -> tuple[str, ...]:
    """Return a reproducible uniform sample (without replacement)."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}.")
    if count > len(molecules):
        raise ValueError(
            f"Cannot sample {count} molecules from a set of {len(molecules)}."
        )
    return tuple(random.Random(seed).sample(list(molecules), count))


def write_smiles(molecules: tuple[str, ...], path: Path | str) -> Path:
    """Write one SMILES per line, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(molecules) + "\n")
    return destination


def read_smiles(path: Path | str) -> tuple[str, ...]:
    """Read one SMILES per line, skipping blanks and ``#`` comments.

    Raises:
        DatasetError: if the file is missing or contains no molecules.
        InvalidSmilesError: if a line is not a parseable molecule.
    """
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"No SMILES file at {source}.")
    molecules = tuple(
        canonical_smiles(line)
        for line in source.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not molecules:
        raise DatasetError(f"{source} contains no molecules.")
    return molecules


def load_fixture() -> tuple[str, ...]:
    """Load the committed 200-molecule ZINC sample used by the tests."""
    return read_smiles(FIXTURE_PATH)
