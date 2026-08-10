#!/usr/bin/env python
"""Main-text MFPT figure: pooled mean-first-passage-time joint plot (MD vs BioKinema),
identical in style to the manuscript main figure. Reads data_cache.pkl (built by
prepare_data.py) and writes mfpt_jointplot.{pdf,png} into KT_OUT_DIR."""
import os, sys, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import load_style_settings, CACHE_PATH, OUT_DIR, mfpt_jointplot


def main():
    load_style_settings()
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    fig = plt.figure(figsize=(4.4, 4.4))
    gs = gridspec.GridSpec(1, 1, figure=fig)
    mfpt_jointplot(fig, gs[0, 0], data)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, dpi in [("pdf", None), ("png", 200)]:
        p = os.path.join(OUT_DIR, f"mfpt_jointplot.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
