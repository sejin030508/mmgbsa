#!/usr/bin/env python
"""
Fig 4b  --  ADK induced fit (open <-> closed), apo and holo.

For each of {apo, holo} produces three PDFs:
  fig4b_adk_rmsd_<tag>.pdf      RMSD-to-1AKE(closed) and RMSD-to-4AKE(open) vs time (0-5 us)
  fig4b_adk_scatter_<tag>.pdf   density scatter of RMSD-to-1AKE vs RMSD-to-4AKE
  fig4b_adk_rmsf_<tag>.pdf      per-residue RMSF over the 4-5 us window only

RMSD is computed by TM-align against the two reference conformers (shipped in refs/).

Usage:
  python plot_adk.py --traj_dir <dir> [--out_dir .] [--workers 40]

<traj_dir> must contain:  ADK_apo/predictions/  and  ADK_holo/predictions/
(the output of  inference/run_adk.sh).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import lib_adk as L

HERE = os.path.dirname(os.path.abspath(__file__))
CONF_CLOSED = os.path.join(HERE, "refs", "1ake_a.cif")  # 1AKE = closed
CONF_OPEN   = os.path.join(HERE, "refs", "4ake_a.cif")  # 4AKE = open

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True,
                    help="dir with ADK_apo/ and ADK_holo/ (each holding predictions/*.cif)")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--conf_closed", default=CONF_CLOSED)
    ap.add_argument("--conf_open", default=CONF_OPEN)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for tag in ("apo", "holo"):
        folder = os.path.join(args.traj_dir, f"ADK_{tag}", "predictions")
        print(f"########## ADK {tag} ##########", flush=True)
        # main() returns: rmsd-to-conf_a list, rmsd-to-conf_b list, RMSF (4-5us window)
        ra, rb, rf = L.main(args.conf_closed, args.conf_open, folder, args.workers)
        L.plot_rmsd_from_main(ra, rb, label_a="RMSD to 1AKE", label_b="RMSD to 4AKE",
                              output_file=os.path.join(args.out_dir, f"fig4b_adk_rmsd_{tag}.pdf"))
        L.plot_rmsd_scatter(ra, rb, name="ADK",
                            xlabel="RMSD to 1AKE", ylabel="RMSD to 4AKE",
                            output_file=os.path.join(args.out_dir, f"fig4b_adk_scatter_{tag}.pdf"))
        L.make_rmsf_plot([rf], [""],
                         output_file=os.path.join(args.out_dir, f"fig4b_adk_rmsf_{tag}.pdf"))
    print("DONE plot_adk")
