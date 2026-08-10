#!/usr/bin/env python
"""figure_kinetics — Protein dynamics (a) + thermodynamics (b).

Style follows user's plot_style.py: Helvetica, font_size 11, axes.linewidth 1.0,
all-black axes/text. MD blue = #356a9f, BK red = #ac3d48.

Panel A — Protein dynamics
  L: MFPT jointplot (KDE fill + scatter overlay, fig4a style; independent x/y
     lim per data; ρ pooled log-Pearson annotated).
  R: 2 cases (1A1WA00 + 1amuB02), each 3×2:
       [struct] [MD tIC scatter coloured by MSM state] [BK tIC scatter]
       [per-protein MFPT scatter] [TICA-1 ACF] [TICA-2 ACF (shared y)]
     All 4 ACF panels show τ legend. TICA labels: tIC1 / tIC2.

Panel B — Protein thermodynamics
  L: 2 bar charts under one shared title "Distribution distance with MD":
     left = Stationary KL div, right = TICA W2 dist; mean line centred.
  R: 2 cases (1bobA03 + 1ciiA03), each 2×2:
       [struct]              [MD tIC density scatter] [BK tIC density scatter]
       [residue contact map] [stationary distribution]

Each sub-element is also saved standalone in figure_kinetics_elements/.
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("KT_OUT_DIR", os.path.join(HERE, "_out"))
CACHE_PATH = os.environ.get("KT_CACHE", os.path.join(OUT_DIR, "data_cache.pkl"))


# ════════════════════════════════════════════════════════════════════════
# Style — matches plot_style.py used by fig2-5
# ════════════════════════════════════════════════════════════════════════
FONT_DIR = "/cto_studio/xtalpi_lab/fengbin/Helvetica"

def load_style_settings():
    for f in ["Helvetica.ttf", "Helvetica-Bold.ttf", "Helvetica-Oblique.ttf",
              "Helvetica-BoldOblique.ttf", "helvetica-light-587ebe5a59211.ttf"]:
        p = os.path.join(FONT_DIR, f)
        if os.path.exists(p):
            try: fm.fontManager.addfont(p)
            except Exception: pass
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 11,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

load_style_settings()

# Colors
C_MD = "#356a9f"
C_BK = "#ac3d48"
C_BAR = "#356a9f"
C_GRAY = "#444"
C_PLACEHOLDER = "#eaeaec"
C_PLACEHOLDER_EDGE = "#b0b0b0"
CLUSTER_CMAP = plt.get_cmap("tab10")


# ─── helpers ──────────────────────────────────────────────────────────────
def all_spines(ax, lw=1.0):
    for s in ("top", "bottom", "left", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color("black")

def despine(ax, which=("top", "right")):
    for s in which: ax.spines[s].set_visible(False)

def hide_log_minor(ax, axes_=("x", "y")):
    """Hide minor ticks on log axes (per user spec)."""
    for a in axes_:
        ax.tick_params(axis=a, which="minor", length=0)

def nice_ticks(lo, hi, n=4):
    import math
    rng = hi - lo
    if rng <= 0: return [lo, hi]
    raw = rng / (n - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda x: abs(x - raw / mag)) * mag
    eps = step * 1e-6
    a = math.floor((lo + eps) / step) * step
    b = math.ceil((hi - eps) / step) * step
    n_ = int(round((b - a) / step))
    return [round(a + i * step, 8) for i in range(n_ + 1)]

def get_density(v):
    """fig4a style: gaussian KDE density per scatter point."""
    return gaussian_kde(np.transpose(v))(np.transpose(v))


# ════════════════════════════════════════════════════════════════════════
# Panel A — MFPT jointplot (fig4a style)
# ════════════════════════════════════════════════════════════════════════
def mfpt_jointplot(fig, gs_block, data):
    md = np.asarray(data["mfpt_pooled_md"], float)
    bk = np.asarray(data["mfpt_pooled_bk"], float)
    lmd, lbk = np.log10(md), np.log10(bk)

    # Independent x/y limits based on data (NOT forced equal)
    pad = 0.18
    xlo = 10 ** (lmd.min() - pad); xhi = 10 ** (lmd.max() + pad)
    ylo = 10 ** (lbk.min() - pad); yhi = 10 ** (lbk.max() + pad)

    gs_in = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_block,
        width_ratios=[1.0, 0.18], height_ratios=[0.18, 1.0],
        wspace=0.04, hspace=0.04,
    )
    ax_top = fig.add_subplot(gs_in[0, 0])
    ax_main = fig.add_subplot(gs_in[1, 0])
    ax_right = fig.add_subplot(gs_in[1, 1])

    # marginals: KDE fill (in log space, projected to linear-log axis)
    xs = np.linspace(np.log10(xlo), np.log10(xhi), 300)
    ys = np.linspace(np.log10(ylo), np.log10(yhi), 300)
    kx = gaussian_kde(lmd, bw_method=0.3)(xs)
    ky = gaussian_kde(lbk, bw_method=0.3)(ys)
    ax_top.fill_between(10 ** xs, 0, kx, color=C_BAR, alpha=1.0, linewidth=0)
    ax_top.set_xscale("log"); ax_top.set_xlim(xlo, xhi); ax_top.set_axis_off()
    ax_right.fill_betweenx(10 ** ys, 0, ky, color=C_BAR, alpha=1.0, linewidth=0)
    ax_right.set_yscale("log"); ax_right.set_ylim(ylo, yhi); ax_right.set_axis_off()

    # main: 2D KDE fill (Blues cmap, thresh=0.05, levels=10) + black scatter overlay
    sns.kdeplot(x=md, y=bk, ax=ax_main, log_scale=(True, True),
                fill=True, color=C_BAR, cmap="Blues",
                thresh=0.05, levels=10, bw_adjust=0.5,
                clip=[(xlo, xhi), (ylo, yhi)])
    ax_main.scatter(md, bk, s=15, alpha=0.4, color="black", linewidth=0,
                    rasterized=True)

    # Linear fit in log space (NO x=y line per user)
    a, b_ = np.polyfit(lmd, lbk, 1)
    fit_x = np.array([xlo, xhi])
    fit_y = 10 ** (a * np.log10(fit_x) + b_)
    ax_main.plot(fit_x, fit_y, "--", color="#333", lw=1.5, alpha=0.6, zorder=4)

    # ρ uses per-system mean of log Pearson (matches official SUMMARY = 0.80)
    ps_pearson = []
    for s, info in data["mfpt_per_system"].items():
        m, b__ = info["md_v"], info["bk_v"]
        if len(m) < 3: continue
        ps_pearson.append(np.corrcoef(np.log10(m), np.log10(b__))[0, 1])
    rho = float(np.mean(ps_pearson))
    # NO box around the annotation (per user request)
    ax_main.text(0.96, 0.06, f"$\\rho$ = {rho:.2f}\n$N$ = {len(md):,}",
                 transform=ax_main.transAxes, fontsize=10,
                 va="bottom", ha="right")

    ax_main.set_xscale("log"); ax_main.set_yscale("log")
    ax_main.set_xlim(xlo, xhi); ax_main.set_ylim(ylo, yhi)
    ax_main.set_xlabel("MD  MFPT (ns)", fontsize=10)
    ax_main.set_ylabel("BioKinema  MFPT (ns)", fontsize=10)
    # Match ticks to case sub-panels
    ax_main.tick_params(labelsize=9)
    ax_main.grid(False)
    hide_log_minor(ax_main)
    all_spines(ax_main)


# ════════════════════════════════════════════════════════════════════════
# Panel B — per-case elements
# ════════════════════════════════════════════════════════════════════════
def _label_offsets(centers, ax):
    """For each cluster centre, pick one of 8 offset directions that
    minimises proximity to any other centre. Returns list of (dx_pt, dy_pt)
    offsets in display points."""
    # 8 candidate directions in display coordinates (points)
    R = 4.5   # label distance from centre, in points (closer than before)
    dirs = [(R, 0), (R*0.7, R*0.7), (0, R), (-R*0.7, R*0.7),
            (-R, 0), (-R*0.7, -R*0.7), (0, -R), (R*0.7, -R*0.7)]
    # Convert centres to display coordinates so distance computations are
    # meaningful regardless of axis scale.
    inv = ax.transData
    disp = np.array([inv.transform(c) for c in centers])
    n = len(centers)
    offsets = []
    used = []   # display positions of placed labels so far
    for i in range(n):
        best = None
        best_score = -np.inf
        for d in dirs:
            lx = disp[i, 0] + d[0]
            ly = disp[i, 1] + d[1]
            # Score: min distance to all other CENTRES and previously PLACED labels
            d_centres = np.min(np.linalg.norm(
                disp - np.array([lx, ly]), axis=1))
            if used:
                d_labels = np.min(np.linalg.norm(
                    np.array(used) - np.array([lx, ly]), axis=1))
            else:
                d_labels = R * 4
            score = d_centres + 0.6 * d_labels
            if score > best_score:
                best_score = score
                best = d
                best_pos = (lx, ly)
        offsets.append(best)
        used.append(best_pos)
    return offsets
