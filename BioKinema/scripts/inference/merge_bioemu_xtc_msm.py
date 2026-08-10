#!/usr/bin/env python3
"""Custom merger for MSM bioemu runs.

For each system under <inference_root>/<category>/<uniprot>/:
  - load every sample dir's *_pred_coordinates.npy: shape (2, N_sample, N_atom, 3)
  - take generated frame [1, :, :, :] -> (N_sample, N_atom, 3) per init
  - stack across all init pdbs -> (N_init * N_sample, N_atom, 3)
  - prepend the N_init init structures (from f0 of any cif, or from GT_coordinates)
  - write <output_root>/<category>/<uniprot>.xtc + <uniprot>.pdb (matching name
    so bioemu_benchmarks.samples.find_samples_in_dir picks them up).

The topology pdb is taken from the first init cif converted by mdtraj.
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np


def find_first_cif(sample_dir: Path) -> Path | None:
    for c in sorted((sample_dir / "predictions").glob("*_f0_*.cif")):
        if "summary" not in c.name.lower():
            return c
    return None


def merge_uniprot(uid_dir: Path, out_xtc: Path, out_pdb: Path,
                  protenix_root: Path | None = None,
                  category: str = "domainmotion") -> tuple[int, int]:
    import mdtraj as md  # local import: env-specific

    sample_dirs = sorted(p for p in uid_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not sample_dirs:
        raise RuntimeError(f"no sample dirs in {uid_dir}")

    init_xyz = []      # (N_init, N_atom, 3) from PROTENIX CIFs directly
    gen_xyz = []       # list of (N_sample, N_atom, 3)
    topo = None
    uid = uid_dir.name

    # Resolve Protenix init CIF dir for this UID
    px_pred_dir = None
    if protenix_root is not None:
        px_pred_dir = protenix_root / category / uid / uid / "seed_101" / "predictions"
        if not px_pred_dir.is_dir():
            print(f"  WARN: protenix preds dir missing for {uid}: {px_pred_dir}", file=sys.stderr)
            px_pred_dir = None

    for sd in sample_dirs:
        pred_files = list(sd.glob("*_pred_coordinates.npy"))
        if not pred_files:
            print(f"  WARN: no pred in {sd.name}", file=sys.stderr)
            continue
        arr = np.load(pred_files[0])  # (2, N_sample, N_atom, 3)
        if arr.ndim != 4 or arr.shape[0] < 2:
            print(f"  WARN: unexpected shape {arr.shape} in {sd.name}", file=sys.stderr)
            continue
        gen_xyz.append(arr[1])  # generated frame, all N_sample

        if topo is None:
            cif = find_first_cif(sd)
            if cif is None:
                raise RuntimeError(f"no f0 cif in {sd} for topology")
            t = md.load(str(cif))
            topo = t.topology

    if not gen_xyz:
        raise RuntimeError(f"no usable generated samples for {uid}")

    # Now load the 20 Protenix init CIFs (sample_0..sample_19) directly
    if px_pred_dir is not None:
        for k in range(20):
            cif = px_pred_dir / f"{uid}_seed_101_sample_{k}.cif"
            if not cif.is_file():
                print(f"  WARN: protenix init missing {cif.name}", file=sys.stderr)
                continue
            t = md.load(str(cif))
            if t.xyz.shape[1] != topo.n_atoms:
                print(f"  WARN: protenix init {cif.name} has {t.xyz.shape[1]} atoms != topo {topo.n_atoms}", file=sys.stderr)
                continue
            # mdtraj loads xtc in nm; cif in nm too (mdtraj converts to nm by default)
            # but pred_coordinates.npy is angstrom -> we'll convert later uniformly.
            # Convert protenix cif xyz (nm) to angstrom so it lives in the same unit as pred_coordinates:
            init_xyz.append(t.xyz[0] * 10.0)

    if not init_xyz:
        raise RuntimeError(f"no protenix init frames loaded for {uid}")

    init_arr = np.stack(init_xyz, axis=0)                # (N_init, N_atom, 3) angstrom
    gen_arr = np.concatenate(gen_xyz, axis=0)            # (N_init*N_sample, N_atom, 3) angstrom
    all_arr = np.concatenate([init_arr, gen_arr], axis=0)

    # mdtraj xtc expects nanometres; pred_coordinates.npy is angstrom (Protenix convention)
    xyz_nm = all_arr.astype(np.float32) / 10.0
    traj = md.Trajectory(xyz_nm, topology=topo)

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    traj[0].save_pdb(str(out_pdb))
    traj.save_xtc(str(out_xtc))
    return init_arr.shape[0], gen_arr.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-root", type=Path, required=True,
                    help="biokinema_<tag>/ dir produced by batch_structure_trajectories.py")
    ap.add_argument("--output-root", type=Path, required=True,
                    help="dir to receive <category>/<uniprot>.xtc + .pdb")
    ap.add_argument("--categories", nargs="*", default=["domainmotion"])
    ap.add_argument("--protenix-root", type=Path,
                    default=Path("/cto_studio/xtalpi_lab/fengbin/Protenix_v1.0.0/protenix_results"),
                    help="Root of Protenix CIF outputs; <root>/<category>/<UID>/<UID>/seed_101/predictions/*.cif")
    args = ap.parse_args()

    for cat in args.categories:
        cat_in = args.inference_root / cat
        if not cat_in.is_dir():
            print(f"SKIP missing {cat_in}"); continue
        cat_out = args.output_root / cat
        cat_out.mkdir(parents=True, exist_ok=True)
        n_ok = 0; n_err = 0
        for uid_dir in sorted(p for p in cat_in.iterdir() if p.is_dir()):
            uid = uid_dir.name
            out_xtc = cat_out / f"{uid}_trajectory.xtc"
            out_pdb = cat_out / f"{uid}_topology.pdb"
            try:
                n_init, n_gen = merge_uniprot(uid_dir, out_xtc, out_pdb,
                                              protenix_root=args.protenix_root,
                                              category=cat)
                print(f"  [{cat}/{uid}] init={n_init} gen={n_gen} total={n_init+n_gen}")
                n_ok += 1
            except Exception as e:
                print(f"  [ERR {cat}/{uid}] {e}", file=sys.stderr)
                n_err += 1
        print(f"{cat}: ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
