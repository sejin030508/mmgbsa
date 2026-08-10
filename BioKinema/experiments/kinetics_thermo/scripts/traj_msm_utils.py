#!/usr/bin/env python
"""Pyemma-free primitives shared by the kinetics/thermo trajectory analyses.

These are lifted verbatim from compute_bk_mfpt.py (which imports pyemma at module load and
therefore only runs under the pyemma env). Keeping them here lets the analysis scripts run
under the protenix env (py3.12), which is required to read protocol-5 MSM pickles.
"""
import os
import pickle

import numpy as np


def load_msm_cache(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def load_bk_ca_coords(sys_dir):
    """Return (list of (T, n_ca, 3) arrays in Angstrom, n_ca).

    Layout: <sys_dir>/seed_<sid>/start_<sid>/start_<sid>_pred_coordinates.npy with shape
    [T, S, A, 3] (Angstrom). Each of the S samples becomes a separate trajectory.
    MSM TICA was built in Angstrom -> DO NOT convert units.
    """
    import mdtraj as md
    if not os.path.isdir(sys_dir):
        return [], None
    seed_dirs = sorted(d for d in os.listdir(sys_dir)
                       if d.startswith("seed_") and os.path.isdir(os.path.join(sys_dir, d)))
    coords_list = []
    n_ca = None
    ca_indices = None
    for sd in seed_dirs:
        seed_path = os.path.join(sys_dir, sd)
        subdirs = sorted(d for d in os.listdir(seed_path)
                         if d.startswith("start_") and
                         os.path.isdir(os.path.join(seed_path, d)))
        if not subdirs:
            continue
        data_dir = os.path.join(seed_path, subdirs[0])
        npys = [f for f in os.listdir(data_dir) if f.endswith("_pred_coordinates.npy")]
        if not npys:
            continue
        coords = np.load(os.path.join(data_dir, npys[0]))  # [T, S, A, 3] Angstrom
        if ca_indices is None:
            pred_dir = os.path.join(data_dir, "predictions")
            if os.path.isdir(pred_dir):
                cifs = sorted(f for f in os.listdir(pred_dir) if f.endswith(".cif"))
                if cifs:
                    ref = md.load(os.path.join(pred_dir, cifs[0]))
                    ca_indices = ref.topology.select("name CA")
                    n_ca = len(ca_indices)
        if ca_indices is None:
            continue
        ca_xyz = coords[..., ca_indices, :]   # [T, S, n_ca, 3]
        T, S, _, _ = ca_xyz.shape
        for s in range(S):
            coords_list.append(ca_xyz[:, s, :, :])
    return coords_list, n_ca


def project_bk(ca_coord_list, msm_cache, K=10):
    """Return (list of [T,] dtrajs of K-cluster labels, list of [T, n_tica] TICA coords)."""
    pi = msm_cache["pair_indices_i"]
    pj = msm_cache["pair_indices_j"]
    mu = msm_cache["tica_mean"].astype(np.float64)
    comp = msm_cache["tica_components"].astype(np.float64)
    centers = msm_cache["by_k"][K]["cluster_centers"].astype(np.float64)
    dtrajs, tica_list = [], []
    for ca in ca_coord_list:
        d = np.linalg.norm(ca[:, pi] - ca[:, pj], axis=-1)  # [T, N_pairs]
        x = (d - mu) @ comp                                  # [T, n_tica]
        dist_sq = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        dtrajs.append(dist_sq.argmin(axis=1).astype(np.int64))
        tica_list.append(x)
    return dtrajs, tica_list


def estimate_T(dtrajs, K, lag_frames):
    """Count-based transition matrix at given lag (no detailed balance enforcement).
    Returns (T, C, empty_mask). Empty rows get a self-transition so the MSM stays valid."""
    C = np.zeros((K, K), dtype=np.float64)
    for tr in dtrajs:
        if len(tr) <= lag_frames:
            continue
        a = tr[:-lag_frames]; b = tr[lag_frames:]
        np.add.at(C, (a, b), 1.0)
    row = C.sum(axis=1, keepdims=True)
    empty = row.flatten() == 0
    row[row == 0] = 1.0
    T = C / row
    for k in np.where(empty)[0]:
        T[k] = 0; T[k, k] = 1.0
    return T, C, empty
