#!/usr/bin/env python3
"""Build <UID>_topology.pdb + <UID>_trajectory.xtc using ONLY the 20 Protenix
init CIFs per system. Useful as a baseline to evaluate what the BioKinema
generated frames add (or remove) on top of Protenix init coverage.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protenix-root", type=Path,
                    default=Path("/cto_studio/xtalpi_lab/fengbin/Protenix_v1.0.0/protenix_results"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--categories", nargs="*", default=["domainmotion"])
    args = ap.parse_args()

    import mdtraj as md

    for cat in args.categories:
        cat_in = args.protenix_root / cat
        if not cat_in.is_dir():
            print(f"SKIP missing {cat_in}"); continue
        cat_out = args.output_root / cat
        cat_out.mkdir(parents=True, exist_ok=True)
        n_ok = 0; n_err = 0
        for uid_dir in sorted(p for p in cat_in.iterdir() if p.is_dir()):
            uid = uid_dir.name
            pred_dir = uid_dir / uid / "seed_101" / "predictions"
            if not pred_dir.is_dir():
                print(f"  SKIP {uid}: no preds"); continue
            cifs = sorted(c for c in pred_dir.glob(f"{uid}_seed_101_sample_*.cif") if "summary" not in c.name)
            if not cifs:
                print(f"  SKIP {uid}: no CIFs"); continue
            xyz = []
            topo = None
            for cif in cifs:
                try:
                    t = md.load(str(cif))
                except Exception as e:
                    print(f"  WARN {cif.name}: {e}"); continue
                if topo is None: topo = t.topology
                if t.xyz.shape[1] != topo.n_atoms:
                    print(f"  WARN {cif.name} atom mismatch"); continue
                xyz.append(t.xyz[0])  # nm
            if not xyz:
                print(f"  SKIP {uid}: no frames loaded"); continue
            traj = md.Trajectory(np.stack(xyz, axis=0).astype(np.float32), topology=topo)
            out_xtc = cat_out / f"{uid}_trajectory.xtc"
            out_pdb = cat_out / f"{uid}_topology.pdb"
            traj[0].save_pdb(str(out_pdb))
            traj.save_xtc(str(out_xtc))
            print(f"  [{cat}/{uid}] frames={len(xyz)}")
            n_ok += 1
        print(f"{cat}: ok={n_ok}")


if __name__ == "__main__":
    main()
