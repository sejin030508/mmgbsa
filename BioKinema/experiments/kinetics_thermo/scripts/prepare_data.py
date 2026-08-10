#!/usr/bin/env python
"""Prepare cached data for fig_kinetics_thermo.

Reads MSM pickles + BK pred coords, computes per-system:
  - pi_md (MSM reversible MLE), pi_bk (BK reversible MLE) -> KL
  - ACF(TICA-1), ACF(TICA-2) full curves -> per-system + aggregate
  - MFPT matrices (MD, BK) -> per-system + pooled scatter

Output: data_cache.pkl  (single dict; load once and plot)
"""
import os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_msm_utils import (
    load_msm_cache, load_bk_ca_coords, project_bk, estimate_T)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Paths are configured via environment variables (set by run_analysis.sh):
MSM_DIR = os.environ.get("KT_MSM_DIR", os.path.join(_HERE, "..", "data", "msm_cache"))
TRAJ_DIR = os.environ["KT_TRAJ_DIR"]            # required: the biokinema_trajs/ dir
OUT_DIR = os.environ.get("KT_OUT_DIR", _HERE)
CACHE_PATH = os.environ.get("KT_CACHE", os.path.join(OUT_DIR, "data_cache.pkl"))

DT_NS = 10.0
K = 10
MAX_LAG = 100   # 100 frames * 10 ns = 1000 ns
DIMS = [0, 1]
NBINS_FES = 60


# ─── MSM math ─────────────────────────────────────────────────────────────
def row_normalize(T):
    T = np.clip(np.asarray(T, float), 0, None)
    r = T.sum(1, keepdims=True); r[r == 0] = 1.0
    return T / r


def largest_connected_set(C):
    C = np.asarray(C, float); K_ = C.shape[0]
    adj = C > 0
    reach = adj.copy()
    for _ in range(K_):
        nxt = reach | (reach @ adj)
        if np.array_equal(nxt, reach): break
        reach = nxt
    np.fill_diagonal(reach, True)
    mutual = reach & reach.T
    seen = np.zeros(K_, bool); best = np.array([], int)
    for i in range(K_):
        if seen[i]: continue
        comp = np.where(mutual[i])[0]
        seen[comp] = True
        if len(comp) > len(best): best = comp
    return best


def reversible_mle_stationary(C, max_iter=5000, tol=1e-12):
    C = np.asarray(C, float); K_ = C.shape[0]
    pi = np.zeros(K_)
    active = largest_connected_set(C)
    if len(active) == 0: return np.full(K_, 1.0 / K_)
    if len(active) == 1: pi[active[0]] = 1.0; return pi
    Cs = C[np.ix_(active, active)]
    ci = Cs.sum(1)
    X = Cs + Cs.T; S = Cs + Cs.T
    for _ in range(max_iter):
        xi = X.sum(1)
        denom = (ci[:, None] / xi[:, None]) + (ci[None, :] / xi[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            Xn = S / denom
        Xn[~np.isfinite(Xn)] = 0.0
        if np.max(np.abs(Xn - X)) < tol:
            X = Xn; break
        X = Xn
    xi = X.sum(1)
    pi[active] = xi / xi.sum()
    return pi


def mfpt_all_pairs(T, lag_ns, reachable):
    Tn = row_normalize(T); n = Tn.shape[0]
    M = np.full((n, n), np.nan)
    idx_reach = np.where(reachable)[0]
    for j in range(n):
        if not reachable[j]: continue
        others = [i for i in idx_reach if i != j]
        if not others: M[j, j] = 0.0; continue
        A = np.eye(len(others)) - Tn[np.ix_(others, others)]
        b = np.ones(len(others))
        try: m = np.linalg.solve(A, b)
        except np.linalg.LinAlgError: m, *_ = np.linalg.lstsq(A, b, rcond=None)
        for k, i in enumerate(others): M[i, j] = m[k] * lag_ns
        M[j, j] = 0.0
    return M


def kl_div(p, q, alpha=1e-3):
    p = np.asarray(p, float); q = np.asarray(q, float); K_ = len(p)
    p = (p + alpha) / (p.sum() + alpha * K_)
    q = (q + alpha) / (q.sum() + alpha * K_)
    return float(np.sum(p * np.log(p / q)))


def per_dim_w2(z_md, z_bk):
    """Root-mean Wasserstein-2 across TICA dims."""
    from scipy import stats
    n_dims = min(z_md.shape[1], z_bk.shape[1])
    vals = []
    for d in range(n_dims):
        try:
            w = stats.wasserstein_distance(z_md[:, d], z_bk[:, d])
            vals.append(w ** 2)
        except Exception:
            pass
    return float(np.sqrt(np.mean(vals))) if vals else float("nan")


def assign_clusters(coords, centers):
    """coords [N, D] -> [N] integer cluster labels (nearest center)."""
    coords = np.asarray(coords, float); centers = np.asarray(centers, float)
    nd = centers.shape[1]
    if coords.shape[1] < nd:
        raise ValueError(f"coords have {coords.shape[1]} dims, centers need {nd}")
    coords = coords[:, :nd]
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return d2.argmin(axis=1).astype(np.int16)


# ─── ACF ──────────────────────────────────────────────────────────────────
def _biased_autocov(x, max_lag):
    n = len(x)
    return np.array([np.dot(x[: n - k], x[k:]) for k in range(max_lag + 1)]) / n


def acf_ensemble(signals_1d, max_lag):
    sigs = [np.asarray(s, float) for s in signals_1d if len(s) >= max_lag + 1]
    if not sigs: return np.full(max_lag + 1, np.nan)
    gmean = np.concatenate(sigs).mean()
    accs = [_biased_autocov(s - gmean, max_lag) for s in sigs]
    g = np.mean(accs, axis=0)
    return g / g[0] if g[0] > 1e-12 else np.zeros(max_lag + 1)


def tau_1e(acf):
    thr = 1.0 / np.e
    for i in range(1, len(acf)):
        if acf[i] <= thr:
            frac = (acf[i - 1] - thr) / max(acf[i - 1] - acf[i], 1e-12)
            return (i - 1 + frac) * DT_NS
    return (len(acf) - 1) * DT_NS


def project_tica_traj(ca, msm):
    pi = msm["pair_indices_i"]; pj = msm["pair_indices_j"]
    mu = msm["tica_mean"].astype(np.float64)
    comp = msm["tica_components"].astype(np.float64)
    d = np.linalg.norm(ca[:, pi, :] - ca[:, pj, :], axis=-1).astype(np.float64)
    return (d - mu) @ comp


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    systems = sorted(
        f.replace("_msm.pkl", "") for f in os.listdir(MSM_DIR)
        if f.endswith("_msm.pkl") and os.path.isdir(os.path.join(TRAJ_DIR, f.replace("_msm.pkl", "")))
    )
    print(f"Found {len(systems)} systems")

    out = {
        "systems": [],
        "kl_per_system": {},        # sys -> float
        "w2_per_system": {},        # sys -> float (root-mean per-dim 1D Wasserstein)
        "acf_per_system": {},       # sys -> {dim: (bk_acf, md_acf)}
        "lags_ns": np.arange(MAX_LAG + 1) * DT_NS,
        "tau_per_system": {},       # sys -> {dim: (tau_bk, tau_md)}
        "mfpt_pooled_md": [],
        "mfpt_pooled_bk": [],
        "mfpt_per_system": {},      # sys -> {"md_v": ..., "bk_v": ..., "n": int}
        "stat_per_system": {},      # sys -> {"pi_md": ..., "pi_bk": ...}
        "its_per_system": {},       # sys -> {"md": [t_2, t_3, ...], "bk": [...], "lag_ns": float}
        "edge_pooled_md": [],
        "edge_pooled_bk": [],
        "edge_per_system": {},      # sys -> {"md": ndarray, "bk": ndarray, "mae": float}
    }

    pooled_mfpt_md, pooled_mfpt_bk = [], []
    pooled_edge_md, pooled_edge_bk = [], []
    for i, sysn in enumerate(systems):
        msm_pkl = os.path.join(MSM_DIR, f"{sysn}_msm.pkl")
        if not os.path.exists(msm_pkl):
            continue
        try:
            msm = load_msm_cache(msm_pkl)
        except Exception as e:
            print(f"  {sysn}: failed to load MSM ({e})"); continue
        ca_list, n_ca = load_bk_ca_coords(os.path.join(TRAJ_DIR, sysn))
        if not ca_list or n_ca != msm["n_ca_full"]:
            print(f"  {sysn}: no BK / n_ca mismatch"); continue

        lag_frames = int(msm["lagtime_frames"])
        lag_ns = lag_frames * DT_NS
        T_md = row_normalize(msm["by_k"][K]["transition_matrix"])
        pi_md = np.asarray(msm["by_k"][K]["stationary_dist"], float); pi_md /= pi_md.sum()

        dtrajs, _ = project_bk(ca_list, msm)
        T_bk, C_bk, empty = estimate_T(dtrajs, K, lag_frames)
        T_bk = row_normalize(T_bk)
        visited = ~empty
        pi_bk = reversible_mle_stationary(C_bk)

        kl_v = kl_div(pi_md, pi_bk)
        out["kl_per_system"][sysn] = kl_v
        out["stat_per_system"][sysn] = {"pi_md": pi_md, "pi_bk": pi_bk}

        # ACF — use BK trajs (re-project to TICA — we already have it via project_bk's helper, but estimate_T discarded coords)
        bk_tica = [project_tica_traj(ca, msm) for ca in ca_list]

        # ── Wasserstein-2 distance (per-dim 1D, root-mean) ────────────
        md_full = np.concatenate([np.asarray(msm["tica_coords_by_traj"][k_])
                                  for k_ in sorted(msm["tica_coords_by_traj"].keys())], 0)
        bk_full = np.concatenate(bk_tica, 0)
        out["w2_per_system"][sysn] = per_dim_w2(md_full, bk_full)
        d_res = {}; tau_res = {}
        for d in DIMS:
            bk_sig = [t[:, d] for t in bk_tica]
            md_sig = []
            for k_ in sorted(msm["tica_coords_by_traj"].keys()):
                t = np.asarray(msm["tica_coords_by_traj"][k_])[:, d]
                if len(t) >= MAX_LAG + 1:
                    md_sig.append(t)
            bk_acf = acf_ensemble(bk_sig, MAX_LAG)
            md_acf = acf_ensemble(md_sig, MAX_LAG)
            d_res[d] = (bk_acf, md_acf)
            tau_res[d] = (tau_1e(bk_acf), tau_1e(md_acf))
        out["acf_per_system"][sysn] = d_res
        out["tau_per_system"][sysn] = tau_res

        # MFPT
        M_md = mfpt_all_pairs(T_md, lag_ns, reachable=np.ones(K, bool))
        M_bk = mfpt_all_pairs(T_bk, lag_ns, reachable=visited)
        mask = np.isfinite(M_md) & np.isfinite(M_bk) & (~np.eye(K, dtype=bool))
        mask &= (M_md > 0) & (M_bk > 0)
        md_v = M_md[mask]; bk_v = M_bk[mask]
        out["mfpt_per_system"][sysn] = {"md_v": md_v, "bk_v": bk_v, "n": int(len(md_v))}
        pooled_mfpt_md.append(md_v); pooled_mfpt_bk.append(bk_v)

        # ─── Implied timescales: t_i = -lag / ln |lambda_i| ────────────
        # Use top-(K-1) non-trivial eigenvalues, sorted by magnitude.
        def _its(T):
            eigvals = np.linalg.eigvals(T)
            mags = np.abs(eigvals)
            mags = mags[mags < 0.9999]                 # drop stationary mode
            mags = np.sort(mags)[::-1]
            mags = mags[mags > 1e-8]
            return -lag_ns / np.log(mags)
        its_md = _its(T_md); its_bk = _its(T_bk)
        n_take = min(len(its_md), len(its_bk), 5)      # store top 5 slow modes
        out["its_per_system"][sysn] = {
            "md": its_md[:n_take], "bk": its_bk[:n_take], "lag_ns": lag_ns}

        # ─── Edge transition frequencies (10-ns step) ──────────────────
        # Off-diagonal edges populated in MD
        ei, ej = np.where((np.eye(K) == 0) & (T_md > 0))
        edge_md = T_md[ei, ej]; edge_bk = T_bk[ei, ej]
        edge_mae = float(np.mean(np.abs(edge_md - edge_bk))) if len(ei) else np.nan
        out["edge_per_system"][sysn] = {"md": edge_md, "bk": edge_bk, "mae": edge_mae}
        pooled_edge_md.append(edge_md); pooled_edge_bk.append(edge_bk)

        # ─── Per-system TICA pools + cluster centers (needed for supplement) ──
        # Lightweight: only 2D coordinates and labels, no trajectories.
        md_keys = sorted(msm["tica_coords_by_traj"].keys())
        md_full = np.concatenate([np.asarray(msm["tica_coords_by_traj"][k_])
                                  for k_ in md_keys], 0)
        bk_full = np.concatenate(bk_tica, 0)
        centers = np.asarray(msm["by_k"][K]["cluster_centers"], float)
        out.setdefault("tica_per_system", {})[sysn] = {
            "md_pool": md_full[:, :2].astype(np.float32),
            "bk_pool": bk_full[:, :2].astype(np.float32),
            "md_pool_lbl": assign_clusters(md_full, centers),
            "bk_pool_lbl": assign_clusters(bk_full, centers),
            "cluster_centers_2d": centers[:, :2].astype(np.float32),
        }

        out["systems"].append(sysn)
        print(f"  [{i+1}/{len(systems)}] {sysn}: KL={kl_v:.3f} tau_md={tau_res[0][1]:.0f}ns "
              f"tau_bk={tau_res[0][0]:.0f}ns  n_mfpt={len(md_v)}")

    out["mfpt_pooled_md"] = np.concatenate(pooled_mfpt_md)
    out["mfpt_pooled_bk"] = np.concatenate(pooled_mfpt_bk)
    out["edge_pooled_md"] = np.concatenate(pooled_edge_md)
    out["edge_pooled_bk"] = np.concatenate(pooled_edge_bk)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"\nSaved cache: {CACHE_PATH}  ({os.path.getsize(CACHE_PATH)/1e6:.1f} MB)")
    print(f"Total systems: {len(out['systems'])}")
    print(f"Mean KL: {np.mean(list(out['kl_per_system'].values())):.3f}")
    print(f"Mean MFPT log10 MAE: "
          f"{np.mean(np.abs(np.log10(out['mfpt_pooled_md']) - np.log10(out['mfpt_pooled_bk']))):.3f}")


if __name__ == "__main__":
    main()
