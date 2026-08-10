#!/usr/bin/env python
"""figure_kinetics_supplement — per-system rows for all 20 CATH2 systems.

Each row contains 6 columns:
  [vertical name | structure | MFPT | MSM stationary (1.5× wide) | MD TICA | BK TICA]

All plots have matching height; titles aligned.  Two pages of 10 rows each.
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import (
    load_style_settings, C_MD, C_BK, C_BAR, C_GRAY,
    CLUSTER_CMAP, all_spines, despine, hide_log_minor, nice_ticks,
    get_density, _label_offsets,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.environ.get("KT_CACHE", os.path.join(HERE, "data_cache.pkl"))
STRUCT_DIR = os.environ.get("KT_STRUCT_DIR", os.path.join(HERE, "..", "data", "structures"))
OUT_DIR = os.environ.get("KT_OUT_DIR", os.path.join(HERE, "_out"))

# ── Layout parameters ────────────────────────────────────────────────────
# Page split: 3 pages of 7, 7, 6 systems (sums to 20 total)
PAGE_SIZES = [7, 7, 6]
N_PAGES = len(PAGE_SIZES)
ROW_H = 1.5             # each row's plotting area height (inches)
COL_BASE = 1.5          # standard column width
NAME_W = 0.25           # vertical PDB-ID column
STRUCT_W = COL_BASE
MFPT_W = COL_BASE
MSM_W = COL_BASE * 2.0  # MSM is 2× wider per user request
TICA_W = COL_BASE
H_GAP_INNER = 0.42      # wspace within row
V_GAP = 0.40            # gap between rows (inches)


# ─── per-row drawing helpers ──────────────────────────────────────────────
def draw_name(ax, pdb_id):
    """Single vertical PDB ID, centered against the structure cell."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, pdb_id, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#222",
            rotation=90, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def draw_structure(ax, sysn, show_title=False):
    png = os.path.join(STRUCT_DIR, f"{sysn}.png")
    if os.path.exists(png):
        img = np.array(Image.open(png).convert("RGB"))
        ax.imshow(img)
    if show_title:
        ax.set_title("Structure", fontsize=12, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_box_aspect(1)


def draw_mfpt(ax, info, show_title=False):
    md = info["md_v"]; bk = info["bk_v"]
    if len(md) == 0: ax.set_axis_off(); return
    lmd, lbk = np.log10(md), np.log10(bk)
    pad = 0.22
    xlo = 10 ** (lmd.min() - pad); xhi = 10 ** (lmd.max() + pad)
    ylo = 10 ** (lbk.min() - pad); yhi = 10 ** (lbk.max() + pad)
    ax.scatter(md, bk, s=12, color=C_BAR, alpha=0.6, edgecolors="none",
               rasterized=True)
    if len(md) >= 3:
        a, b_ = np.polyfit(lmd, lbk, 1)
        ax.plot([xlo, xhi], 10**(a*np.log10([xlo, xhi])+b_), "--",
                color="#333", lw=0.8, alpha=0.7)
        rho = float(np.corrcoef(lmd, lbk)[0, 1])
        ax.text(0.95, 0.06, f"$\\rho$ = {rho:.2f}",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("MD MFPT (ns)", fontsize=8)
    ax.set_ylabel("BK MFPT (ns)", fontsize=8)
    if show_title:
        ax.set_title("MFPT", fontsize=12, pad=6)
    ax.tick_params(labelsize=7)
    ax.grid(False)
    hide_log_minor(ax)
    ax.set_box_aspect(1)
    all_spines(ax)


def draw_stat(ax, pi_md, pi_bk, kl, show_title=False, show_legend=False):
    """Same height as 1×1 plots but 1.5× wider via set_box_aspect."""
    x = np.arange(len(pi_md)); w = 0.38
    ax.bar(x - w/2, pi_md, width=w, color=C_MD, edgecolor="black",
           linewidth=0.3, label="MD")
    ax.bar(x + w/2, pi_bk, width=w, color=C_BK, edgecolor="black",
           linewidth=0.3, label="BioKinema")
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x], fontsize=7)
    ax.set_xlabel("MSM state", fontsize=8)
    ax.set_ylabel("π", fontsize=8)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6]); ax.set_ylim(0, 0.7)
    if show_title:
        ax.set_title("MSM stationary", fontsize=12, pad=6)
    ax.text(0.97, 0.94, f"$D_{{\\mathrm{{KL}}}}$ = {kl:.2f}",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top")
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.93),
                  fontsize=6.5, handlelength=0.7, labelspacing=0.15,
                  borderpad=0.15, ncol=2, columnspacing=0.5)
    ax.tick_params(labelsize=7)
    # 2× wider than tall: box_aspect = h/w = 1/2
    ax.set_box_aspect(1.0 / 2.0)
    despine(ax)


def draw_tica(ax, pool, centers, xlim, ylim, w2=None,
              show_title=False, title_text="", show_ylabel=True):
    """Density TICA scatter (single ensemble) + cluster centres."""
    dens = get_density(pool)
    order = np.argsort(dens)
    ax.scatter(pool[order, 0], pool[order, 1], s=4, alpha=0.7,
               c=dens[order], cmap="mako_r", vmin=-0.05, vmax=1.0,
               linewidths=0, rasterized=True)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("tIC1", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("tIC2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_box_aspect(1)
    if show_title:
        ax.set_title(title_text, fontsize=12, pad=6)
    all_spines(ax)
    if w2 is not None:
        ax.text(0.97, 0.94, f"$W_2$ = {w2:.2f}",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
                color="#222")
    # Cluster centres + numbered labels
    ax.figure.canvas.draw()
    colors = [CLUSTER_CMAP(i % 10) for i in range(len(centers))]
    ax.scatter(centers[:, 0], centers[:, 1], s=18, c=colors,
               linewidths=0, marker="o", zorder=4)
    offs = _label_offsets(centers, ax)
    for i, (cx, cy) in enumerate(centers):
        dx, dy = offs[i]
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
        ax.annotate(str(i), (cx, cy), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=6, color=colors[i], fontweight="bold",
                    ha=ha, va=va, zorder=5)


# ─── one row per system ──────────────────────────────────────────────────
def draw_row(fig, gs_row, sysn, data, show_titles=False, show_stat_legend=False):
    """Render one system row into the given gridspec row.
    Columns: [vert name | struct | MFPT | MSM stationary | MD TICA | BK TICA]"""
    short = sysn.replace("cath2_", "")
    pdb_id = short.upper()

    gs_in = gridspec.GridSpecFromSubplotSpec(
        1, 6, subplot_spec=gs_row,
        width_ratios=[NAME_W, STRUCT_W, MFPT_W, MSM_W, TICA_W, TICA_W],
        wspace=H_GAP_INNER,
    )
    ax_name = fig.add_subplot(gs_in[0, 0])
    draw_name(ax_name, pdb_id)

    ax_struct = fig.add_subplot(gs_in[0, 1])
    draw_structure(ax_struct, sysn, show_title=show_titles)

    ax_mfpt = fig.add_subplot(gs_in[0, 2])
    draw_mfpt(ax_mfpt, data["mfpt_per_system"][sysn], show_title=show_titles)

    ax_stat = fig.add_subplot(gs_in[0, 3])
    draw_stat(ax_stat,
              data["stat_per_system"][sysn]["pi_md"],
              data["stat_per_system"][sysn]["pi_bk"],
              data["kl_per_system"][sysn],
              show_title=show_titles,
              show_legend=show_stat_legend)

    tica = data["tica_per_system"][sysn]
    pool_all = np.concatenate([tica["md_pool"], tica["bk_pool"]], 0)
    x0, x1 = pool_all[:, 0].min(), pool_all[:, 0].max()
    y0, y1 = pool_all[:, 1].min(), pool_all[:, 1].max()
    pad_x = (x1 - x0) * 0.05; pad_y = (y1 - y0) * 0.05
    xlim = (x0 - pad_x, x1 + pad_x); ylim = (y0 - pad_y, y1 + pad_y)

    w2 = data["w2_per_system"][sysn]
    ax_md = fig.add_subplot(gs_in[0, 4])
    draw_tica(ax_md, tica["md_pool"], tica["cluster_centers_2d"], xlim, ylim,
              w2=None, show_title=show_titles, title_text="MD", show_ylabel=True)
    ax_bk = fig.add_subplot(gs_in[0, 5])
    draw_tica(ax_bk, tica["bk_pool"], tica["cluster_centers_2d"], xlim, ylim,
              w2=w2, show_title=show_titles, title_text="BioKinema",
              show_ylabel=False)


# ─── main: 2 pages, 10 systems each ─────────────────────────────────────
def main():
    load_style_settings()
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    systems = sorted(data["systems"])
    # Split into PAGE_SIZES chunks
    pages = []
    idx = 0
    for sz in PAGE_SIZES:
        pages.append(systems[idx:idx + sz])
        idx += sz

    # Total page width: name + 4 plots-of-COL_BASE + 1.5×COL_BASE + 5 gaps + small margin
    page_w = (NAME_W + STRUCT_W + MFPT_W + MSM_W + TICA_W * 2
              + H_GAP_INNER * 5 * COL_BASE / 5 + 0.5)
    # Empirically a bit wider works better
    page_w = max(page_w, 9.0)

    for p_idx, sys_chunk in enumerate(pages):
        n = len(sys_chunk)
        page_h = 0.15 + n * ROW_H + (n - 1) * V_GAP + 0.15
        fig = plt.figure(figsize=(page_w, page_h))
        outer = gridspec.GridSpec(
            n, 1, figure=fig,
            hspace=V_GAP / ROW_H,
            left=0.025, right=0.985,
            top=1 - 0.05 / page_h,
            bottom=0.05 / page_h,
        )
        for r_idx, sysn in enumerate(sys_chunk):
            draw_row(fig, outer[r_idx, 0], sysn, data,
                     show_titles=(r_idx == 0),
                     show_stat_legend=(r_idx == 0))
        out_pdf = os.path.join(OUT_DIR, f"figure_kinetics_sup_page{p_idx+1}.pdf")
        out_png = os.path.join(OUT_DIR, f"figure_kinetics_sup_page{p_idx+1}.png")
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
        plt.close(fig)
        print(f"saved {out_pdf}  ({n} systems, {page_w:.1f}×{page_h:.1f}\")")


if __name__ == "__main__":
    main()
