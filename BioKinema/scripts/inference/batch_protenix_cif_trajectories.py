#!/usr/bin/env python3
"""
Legacy entry point for batch Protenix-CIF inference.

This script forwards to ``batch_structure_trajectories.py``. If the first
argument is not ``protenix`` or ``bioemu``, ``protenix`` is inserted so old
commands keep working::

  python scripts/batch_protenix_cif_trajectories.py \\
    --protenix-results-root ... --output-root ... --checkpoint-path ...

is equivalent to::

  python scripts/batch_structure_trajectories.py protenix \\
    --protenix-results-root ... --output-root ... --checkpoint-path ...

For BioEmu PDB+XTC, use ``batch_structure_trajectories.py bioemu`` (or pass
``bioemu`` as the first argument to this wrapper).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_batch_module():
    here = Path(__file__).resolve().parent
    path = here / "batch_structure_trajectories.py"
    spec = importlib.util.spec_from_file_location(
        "batch_structure_trajectories", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = _load_batch_module()
    argv = sys.argv[1:]
    if argv and argv[0] not in ("protenix", "bioemu"):
        argv = ["protenix"] + argv
    mod.main(argv)


if __name__ == "__main__":
    main()
