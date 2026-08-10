#!/usr/bin/env python
"""
Fig 4c  --  Pin1 allosteric loop (apo vs holo).

Produces two PDFs:
  fig4c_pin1_rmsf.pdf                         per-residue RMSF (apo vs holo, + Delta)
  fig4c_pin1_rmsd_align1-999_calc60-70.pdf    loop 60-70 RMSD-to-frame0,
                                              whole-protein CA superposition (align 1-999),
                                              x-axis in us (0.0, 0.3, ..., 1.5; no "us" label)

Usage:
  python plot_pin1.py --traj_dir <dir> [--out_dir .]

<traj_dir> must contain:  Pin1_apo/predictions/  and  Pin1_holo/predictions/
(the output of  inference/run_pin1.sh).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import lib_pin1 as L

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True,
                    help="dir with Pin1_apo/ and Pin1_holo/ (each holding predictions/*.cif)")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    holo = os.path.join(args.traj_dir, "Pin1_holo", "predictions")
    apo  = os.path.join(args.traj_dir, "Pin1_apo",  "predictions")

    # ---- per-residue RMSF (order [holo, apo] -> holo=blue, apo=red, Delta=apo-holo) ----
    print("########## Pin1 RMSF ##########", flush=True)
    rmsf = L.read_data([holo, apo])
    L.make_plot(rmsf, output_file=os.path.join(args.out_dir, "fig4c_pin1_rmsf.pdf"))

    # ---- loop 60-70 RMSD to frame 0, whole-protein align (1-999) ----
    print("########## Pin1 RMSD (align 1-999, calc 60-70) ##########", flush=True)
    rmsd = L.calculate_rmsd_data([holo, apo], align_residue_range=(1, 999),
                                 calc_residue_range=(60, 70))
    L.plot_rmsd(rmsd, ylabel="Å",
                output_file=os.path.join(args.out_dir, "fig4c_pin1_rmsd_align1-999_calc60-70.pdf"))
    print("DONE plot_pin1")
