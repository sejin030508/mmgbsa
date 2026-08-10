#!/usr/bin/env python
"""Aggregated per-system result table (the supplementary table). Reads data_cache.pkl
and writes per_system_table.csv into KT_OUT_DIR, with a final cross-system MEAN row.

Columns (per CATH2-OOD system):
  rho_P  : Pearson correlation of log10(MFPT) between MD and BioKinema
  rho_S  : Spearman rank correlation of MFPT
  KL     : KL divergence between MD and BioKinema reversible-MLE MSM stationary distributions
  pi_MAE : per-state mean absolute error of the stationary distribution
  W2     : root-mean Wasserstein-2 distance across the five slowest TICA dimensions
  n_st   : number of MSM states populated by BioKinema (pi_bk > 1e-4), out of K=10
"""
import os, sys, csv, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import CACHE_PATH, OUT_DIR


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 2 else float("nan")


def spearman(a, b):
    if len(a) < 2:
        return float("nan")
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def main():
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    rows = []
    for s in data["systems"]:
        mf = data["mfpt_per_system"][s]
        md = np.asarray(mf["md_v"], float); bk = np.asarray(mf["bk_v"], float)
        st = data["stat_per_system"][s]
        pi_md = np.asarray(st["pi_md"], float); pi_bk = np.asarray(st["pi_bk"], float)
        rows.append({
            "system": s.replace("cath2_", "").upper(),
            "rho_P": pearson(np.log10(md), np.log10(bk)),
            "rho_S": spearman(md, bk),
            "KL": float(data["kl_per_system"][s]),
            "pi_MAE": float(np.mean(np.abs(pi_md - pi_bk))),
            "W2": float(data["w2_per_system"][s]),
            "n_st": int((pi_bk > 1e-4).sum()),
        })

    fields = ["system", "rho_P", "rho_S", "KL", "pi_MAE", "W2", "n_st"]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "per_system_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})
        mean = {"system": "MEAN"}
        for k in ["rho_P", "rho_S", "KL", "pi_MAE", "W2"]:
            mean[k] = f"{np.mean([r[k] for r in rows]):.4f}"
        mean["n_st"] = f"{np.mean([r['n_st'] for r in rows]):.1f}"
        w.writerow(mean)
    print(f"wrote {path}  ({len(rows)} systems)")
    print(f"MEAN: rho_P={np.mean([r['rho_P'] for r in rows]):.3f}  "
          f"rho_S={np.mean([r['rho_S'] for r in rows]):.3f}  "
          f"KL={np.mean([r['KL'] for r in rows]):.3f}  "
          f"pi_MAE={np.mean([r['pi_MAE'] for r in rows]):.3f}  "
          f"W2={np.mean([r['W2'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
