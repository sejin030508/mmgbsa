#!/usr/bin/env python
"""
Fig 2a  --  MISATO-OOD ligand physical stability.

For every protein-ligand system it compares the ligand bond lengths / bond angles
in each generated frame against the reference geometry (get_gt_mol, from
BIOKINEMA_LIGAND_BONDS_DIR), then plots the per-time-segment error distributions:
  misato_bond_length_error.pdf
  misato_bond_angle_error.pdf

All data is kept; the y-axis is simply limited to [0, 0.2] so the rare large-error
tails fall outside the view (no points are removed from the distributions).

Usage:
  python plot_misato.py --traj_dir <dir> [--out_dir .]

<traj_dir> must contain one subfolder per system, each with predictions/*.cif
(the output of  inference/run_misato_ood.sh). Requires BIOKINEMA_LIGAND_BONDS_DIR.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import pandas as pd, seaborn as sns
import lib_misato as L

BIN_SIZE = 10     # 101-frame trajs @10ns (=1us) -> ten 0.1-us segments
Y_LIMIT = 0.2     # y-axis cap; tails above this are clipped from view (data kept)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True,
                    help="dir with one subfolder per system (each holding predictions/*.cif)")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    subdirs = sorted(os.path.join(args.traj_dir, d) for d in os.listdir(args.traj_dir)
                     if os.path.isdir(os.path.join(args.traj_dir, d)))
    print(f"MISATO systems: {len(subdirs)}", flush=True)

    dfs = []
    for i, sd in enumerate(subdirs):
        if i % 10 == 0:
            print(f"  {i}/{len(subdirs)} {os.path.basename(sd)}", flush=True)
        rows = L.process_single_system(sd, ligand_name="LIG")
        if rows:
            dfs.append(pd.DataFrame(rows))
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"aggregated {len(final_df)} points (all kept; y-axis capped at {Y_LIMIT})\n", flush=True)

    # ---- time segments + plot (y-axis limited to [0, Y_LIMIT]; nothing dropped) ----
    final_df["Time Segment"] = final_df["Frame Index"].apply(
        lambda x: f"{(x // BIN_SIZE) * BIN_SIZE}-{(x // BIN_SIZE) * BIN_SIZE + BIN_SIZE}")
    order = sorted(final_df["Time Segment"].unique(), key=lambda s: int(s.split("-")[0]))
    sns.set_style("ticks")
    pal = sns.color_palette("coolwarm", n_colors=len(order))

    L.plot_metric(final_df, "Bond Length Error-Å", "bond_length_error",
                  order, pal, args.out_dir, md_value=0.042, ylabel="Å",
                  y_min=0.0, y_max=Y_LIMIT)
    L.plot_metric(final_df, "Bond Angle Error-radians", "bond_angle_error",
                  order, pal, args.out_dir, md_value=0.048, ylabel="radians",
                  y_min=0.0, y_max=Y_LIMIT)
    print("DONE plot_misato")
