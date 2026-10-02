#!/usr/bin/env python
"""Splice the generated results tables into README.md.

The README's tables are never edited by hand: this script replaces everything
between the RESULTS markers with the output of make_report.py, so what is
committed is exactly what was measured.

Usage:  python scripts/make_report.py runs > runs/RESULTS.md
        python scripts/update_readme.py runs/RESULTS.md
"""

from __future__ import annotations

import sys
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def splice(readme: str, tables: str) -> str:
    """Return ``readme`` with the region between the markers replaced.

    Raises:
        ValueError: if the markers are missing or out of order.
    """
    start = readme.find(START)
    end = readme.find(END)
    if start < 0 or end < 0:
        raise ValueError(f"README.md must contain {START} and {END}.")
    if end < start:
        raise ValueError(f"{END} appears before {START} in README.md.")
    return (
        readme[: start + len(START)]
        + "\n\n"
        + tables.strip()
        + "\n\n"
        + readme[end:]
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: update_readme.py <results.md> [readme.md]", file=sys.stderr)
        return 2
    tables_path = Path(argv[1])
    readme_path = Path(argv[2]) if len(argv) > 2 else Path("README.md")
    if not tables_path.exists():
        print(f"error: no results file at {tables_path}", file=sys.stderr)
        return 2
    if not readme_path.exists():
        print(f"error: no README at {readme_path}", file=sys.stderr)
        return 2
    try:
        updated = splice(readme_path.read_text(), tables_path.read_text())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    readme_path.write_text(updated)
    print(f"spliced {tables_path} into {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
