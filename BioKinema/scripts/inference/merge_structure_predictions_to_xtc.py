#!/usr/bin/env python3
"""
Convert BioKinema trajectory CIFs to one PDB + one XTC **per UniProt**.

Under each ``<category>/<uniprot>/``, every ``<run_name>/predictions/*.cif`` is
collected (all BioKinema runs for that protein), ordered by ``run_name`` then by
`_f<number>_` within each run, and merged into a single trajectory::

  {uniprot}_topology.pdb
  {uniprot}_trajectory.xtc

Layout (same whether upstream BioKinema used Protenix or BioEmu inputs)::

  <results_root>/<category>/<uniprot>/<run_name>/predictions/*.cif

With ``--output-root DIR`` (or in-place without it), merged files go **directly under
the benchmark (category) folder** — no per–UniProt subfolder::

  DIR/<category>/{uniprot}_topology.pdb
  DIR/<category>/{uniprot}_trajectory.xtc

Without ``--output-root``, writes under ``<results_root>/<category>/`` (not under
``<uniprot>/``).

Requires: mdtraj, numpy.

Example::

  python scripts/merge_structure_predictions_to_xtc.py \\
    --results-root ./biokinema_trajectories_bioemu \\
    --output-root ./trajectories_bioemu_xtc \\
    --jobs 8
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np


def _frame_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"bioemu_sample_(\d+)", path.name, re.IGNORECASE)
    if m:
        return (int(m.group(1)), path.name)
    m = re.search(r"sample_(\d+)", path.name, re.IGNORECASE)
    if m:
        return (int(m.group(1)), path.name)
    return (-1, path.name)


def _biokinema_cif_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"_f(\d+)_", path.name)
    if m:
        return (int(m.group(1)), path.name)
    return _frame_sort_key(path)


def find_biokinema_prediction_cifs(pred_dir: Path) -> list[Path]:
    """CIFs in a single BioKinema ``predictions/`` directory."""
    cifs: list[Path] = []
    for p in pred_dir.glob("*.cif"):
        if "summary" in p.name.lower():
            continue
        cifs.append(p)
    return sorted(cifs, key=_biokinema_cif_sort_key)


def collect_uniprot_cifs(uid_dir: Path) -> list[Path]:
    """All CIF paths under ``uid_dir/<run>/predictions/``, ordered by run then frame."""
    cifs: list[Path] = []
    for run_dir in sorted(uid_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        pred = run_dir / "predictions"
        if not pred.is_dir():
            continue
        cifs.extend(find_biokinema_prediction_cifs(pred))
    return cifs


def iter_uniprot_cif_groups(
    root: Path, categories: set[str] | None = None
):
    """Yield (category, uniprot_id, uid_dir, list[all_cif_paths])."""
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        if categories is not None and cat_dir.name not in categories:
            continue
        for uid_dir in sorted(cat_dir.iterdir()):
            if not uid_dir.is_dir():
                continue
            uid = uid_dir.name
            cifs = collect_uniprot_cifs(uid_dir)
            if not cifs:
                continue
            yield cat_dir.name, uid, uid_dir, cifs


def convert_cif_to_xtc(
    cif_files: list[Path] | Iterable[Path],
    topology_out_path: os.PathLike[str] | str,
    trajectory_out_path: os.PathLike[str] | str,
) -> bool:
    """
    Merge multiple CIF files into one PDB topology and one XTC trajectory (mdtraj).

    First file defines atom ordering; all structures must match that topology.
    """
    import mdtraj as md  # type: ignore

    paths = [Path(p) for p in cif_files]
    if not paths:
        return False

    cif_strs = [str(p.resolve()) for p in paths]

    ref_structure = md.load(cif_strs[0])
    num_atoms = ref_structure.n_atoms
    bfactors_for_save = np.zeros((1, num_atoms))
    ref_structure.save_pdb(
        os.fspath(topology_out_path), bfactors=bfactors_for_save
    )

    full_trajectory = md.load(cif_strs)
    full_trajectory.save_xtc(os.fspath(trajectory_out_path))

    return True


def _merge_uniprot_worker(
    packed: tuple[tuple[str, ...], str, str, str],
) -> tuple[str, str | None]:
    """Multiprocessing worker: (cif paths, pdb out, xtc out, label) -> (label, err or None)."""
    cif_paths, pdb_out, xtc_out, label = packed
    try:
        convert_cif_to_xtc([Path(p) for p in cif_paths], pdb_out, xtc_out)
        return (label, None)
    except Exception as e:
        return (label, repr(e))


def _resolve_job_count(jobs: int) -> int:
    if jobs <= 0:
        return max(1, os.cpu_count() or 4)
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge all BioKinema prediction CIFs per UniProt into one PDB + XTC "
            "(output of batch_structure_trajectories.py)."
        ),
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help=(
            "BioKinema results directory: parent of <category>/ folders "
            "(e.g. biokinema_trajectories_protenix or biokinema_trajectories_bioemu)."
        ),
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Optional. Write "
            "DIR/<category>/{uniprot}_topology.pdb and {uniprot}_trajectory.xtc "
            "(flat under each benchmark name). If omitted, write under "
            "<results_root>/<category>/."
        ),
    )
    ap.add_argument(
        "--categories",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Only these top-level category folder names (default: all).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_topology.pdb / *_trajectory.xtc.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions only; do not write files.",
    )
    ap.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Parallel processes (one UniProt per worker). "
            "Default 0 = use all CPU cores; use 1 to force sequential."
        ),
    )
    args = ap.parse_args(argv)

    root = args.results_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    cat_filter = set(args.categories) if args.categories else None
    n_ok = 0
    n_skip = 0
    out_root = args.output_root.resolve() if args.output_root is not None else None

    tasks: list[tuple[tuple[str, ...], str, str, str]] = []

    for category, uid, uid_dir, cifs in iter_uniprot_cif_groups(root, cat_filter):
        folder_path = (out_root if out_root is not None else root) / category
        source_pdb = folder_path / f"{uid}_topology.pdb"
        source_xtc = folder_path / f"{uid}_trajectory.xtc"

        if (
            not args.overwrite
            and source_pdb.is_file()
            and source_xtc.is_file()
        ):
            print(f"[skip] exists: {source_pdb} + {source_xtc}", file=sys.stderr)
            n_skip += 1
            continue

        label = f"{category}/{uid}"
        print(
            f"[merge] {label}: {len(cifs)} CIF (all runs) -> {source_pdb}",
            file=sys.stderr,
            flush=True,
        )
        if args.dry_run:
            n_ok += 1
            continue

        folder_path.mkdir(parents=True, exist_ok=True)
        cif_strs = tuple(str(p.resolve()) for p in cifs)
        tasks.append((cif_strs, str(source_pdb), str(source_xtc), label))

    if args.dry_run:
        print(
            f"Done. Would write {n_ok} protein(s), skipped {n_skip} (already present)."
        )
        return 0

    if not tasks:
        print(f"Done. Wrote 0 protein(s), skipped {n_skip} (already present).")
        return 0

    max_workers = min(len(tasks), _resolve_job_count(args.jobs))
    use_pool = args.jobs != 1 and len(tasks) > 1 and max_workers > 1
    failures: list[tuple[str, str]] = []
    n_ok = 0

    if use_pool:
        print(
            f"[pool] {len(tasks)} protein(s), {max_workers} worker process(es)",
            file=sys.stderr,
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_merge_uniprot_worker, t): t for t in tasks}
            for fut in as_completed(futs):
                label, err = fut.result()
                if err is None:
                    n_ok += 1
                    print(f"[ok] {label}", file=sys.stderr, flush=True)
                else:
                    failures.append((label, err))
                    print(f"[fail] {label}: {err}", file=sys.stderr, flush=True)
    else:
        for t in tasks:
            label, err = _merge_uniprot_worker(t)
            if err is None:
                n_ok += 1
                print(f"[ok] {label}", file=sys.stderr, flush=True)
            else:
                failures.append((label, err))
                print(f"[fail] {label}: {err}", file=sys.stderr, flush=True)

    print(
        f"Done. Wrote {n_ok} protein(s), skipped {n_skip} (already present)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
