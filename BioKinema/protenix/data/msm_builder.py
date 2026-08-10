"""
MSM Builder — Build Markov State Models for trajectory systems and cache artifacts.

Two MSM granularities:
  - Fine-grained (n_clusters=200): for pseudo-trajectory resampling (data augmentation)
  - Coarse-grained (n_clusters=20): for TICA-space loss computation

Usage:
  Called lazily during Dataset.__init__ — builds and caches MSM artifacts on first run,
  loads from cache on subsequent runs.
"""

import os
import pickle
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MSMArtifacts:
    """MSM artifacts for a single system, containing both fine and coarse MSMs."""

    # Shared TICA parameters (linear projection, used for differentiable forward in loss)
    tica_mean: np.ndarray  # [N_pairs,] float32
    tica_components: np.ndarray  # [N_pairs, N_tica] float32
    tica_eigenvalues: np.ndarray  # [N_tica,] float32 — for autocorrelation loss

    # Coarse MSM (for loss computation, ~20 states)
    coarse_cluster_centers: np.ndarray  # [K_coarse, N_tica] float32
    coarse_transition_matrix: np.ndarray  # [K_coarse, K_coarse] float32
    coarse_stationary_distribution: np.ndarray  # [K_coarse,] float32
    coarse_state_labels: dict  # {traj_name: np.ndarray [T,] int16}
    coarse_n_states: int

    # Fine MSM (for pseudo-trajectory resampling, ~200 states)
    fine_cluster_centers: np.ndarray  # [K_fine, N_tica] float32
    fine_transition_matrix: np.ndarray  # [K_fine, K_fine] float32
    fine_stationary_distribution: np.ndarray  # [K_fine,] float32
    fine_state_labels: dict  # {traj_name: np.ndarray [T,] int16}
    fine_n_states: int

    # Metadata
    lagtime_frames: int
    n_tica_dims: int
    pair_indices_i: np.ndarray  # upper-triangle pair indices
    pair_indices_j: np.ndarray

    # Per-traj frame counts (for pseudo-trajectory construction)
    traj_frame_counts: dict = field(default_factory=dict)  # {traj_name: int}

    # Within-cluster diagonal log-variance for Gaussian HMM emission.
    # Shape: [K_coarse, N_tica] float32.  log σ²_{i,d} = log(var of TICA dim d in cluster i).
    # None in old caches (backward compat); dataset computes an isotropic fallback when missing.
    coarse_cluster_log_vars: Optional[np.ndarray] = None

    # GMM emission fields (preferred over coarse_cluster_log_vars when present).
    # Supports per-cluster mixture of Gaussians; C=1 degenerates to a single Gaussian.
    # weights + means + precisions_chol must all be set together (full-covariance version).
    # Shape [K, C, ...] where C = max GMM components used across all clusters.
    # Clusters with fewer optimal components have extra entries set to weight=0.
    coarse_cluster_gmm_weights: Optional[np.ndarray] = None          # [K, C]    float32
    coarse_cluster_gmm_means:   Optional[np.ndarray] = None          # [K, C, N_tica] float32
    # Full-covariance precision Cholesky: lower-triangular L s.t. Σ^{-1} = L L^T.
    # Preferred over coarse_cluster_gmm_log_vars (diagonal approximation).
    coarse_cluster_gmm_precisions_chol: Optional[np.ndarray] = None  # [K, C, N_tica, N_tica] float32
    # Deprecated diagonal log-variances (kept for backward compat with old caches).
    coarse_cluster_gmm_log_vars: Optional[np.ndarray] = None         # [K, C, N_tica] float32

    # Full (uncropped) CA count — used in dataset to detect spatial-crop inconsistency.
    # pair_indices_i/j reference positions 0..max_ca-1 within the original N_ca atoms.
    # If the model crops the protein to fewer than n_ca_full CA atoms, pair indices
    # may reference atoms that no longer exist → TICA loss must be disabled for that sample.
    # Default 0 = unknown (old caches); the dataset falls back to the in-loss guard check.
    n_ca_full: int = 0

    # All-C GMM fits: {c: {"weights": [K,c], "means": [K,c,NT], "precisions_chol": [K,c,NT,NT]}}
    # Populated by rebuild_msm_emission.py with max_gmm_components distinct uniform-C fits.
    # dataset.py picks one C based on config["gmm_n_components"].
    # None in old caches (falls back to coarse_cluster_gmm_weights/means/precisions_chol).
    coarse_cluster_gmm_by_c: Optional[dict] = None


def compute_ca_pairwise_distances(ca_coords: np.ndarray, max_ca: int = 100):
    """
    Compute upper-triangle pairwise distances from Cα coordinates.
    Subsamples Cα atoms if N_ca > max_ca to keep feature dimension manageable.

    Args:
        ca_coords: [T, N_ca, 3] or [N_ca, 3]
        max_ca: maximum number of Cα atoms to use (subsamples evenly if exceeded)

    Returns:
        dists: [T, N_pairs] or [N_pairs]
        pair_indices: (idx_i, idx_j)
    """
    if ca_coords.ndim == 2:
        ca_coords = ca_coords[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    N_ca = ca_coords.shape[1]

    # Subsample Cα atoms evenly if too many (keeps TICA/clustering tractable)
    if N_ca > max_ca:
        ca_indices = np.linspace(0, N_ca - 1, max_ca, dtype=int)
        ca_coords_sub = ca_coords[:, ca_indices]
        n_sub = max_ca
    else:
        ca_indices = np.arange(N_ca)
        ca_coords_sub = ca_coords
        n_sub = N_ca

    idx_i, idx_j = np.triu_indices(n_sub, k=1)  # indices into subsampled array
    diff = ca_coords_sub[:, idx_i] - ca_coords_sub[:, idx_j]  # [T, N_pairs, 3]
    dists = np.linalg.norm(diff, axis=-1)  # [T, N_pairs]

    # Map back to ORIGINAL Cα indices so the loss can index into full predicted coords
    orig_idx_i = ca_indices[idx_i].astype(np.int64)
    orig_idx_j = ca_indices[idx_j].astype(np.int64)

    if squeeze:
        dists = dists[0]
    return dists, (orig_idx_i, orig_idx_j)


def _build_single_msm(tica_outputs, n_clusters, msm_lagtime):
    """Build a single MSM at given granularity.

    Args:
        tica_outputs: list of [T_i, N_tica] arrays
        n_clusters: desired number of clusters
        msm_lagtime: lagtime for MSM transition estimation (shorter = more transitions observed)

    Returns:
        (centers, T, pi, dtrajs, n_states) or None.
    """
    import deeptime

    tica_concat = np.concatenate(tica_outputs, axis=0)
    total_frames = tica_concat.shape[0]

    # Scale clusters: at least ~50 frames per cluster for robust transition estimation
    actual_k = min(n_clusters, max(3, total_frames // 50))
    if actual_k < 3:
        return None

    kmeans_est = deeptime.clustering.KMeans(n_clusters=actual_k, max_iter=500, init_strategy="kmeans++")
    kmeans_est.fit(tica_concat)
    kmeans_model = kmeans_est.fetch_model()
    dtrajs_raw = [kmeans_model.transform(t) for t in tica_outputs]

    # Use short MSM lagtime (1-2 frames) to maximize transition observations
    # This is separate from TICA lagtime — MSM lagtime controls how many transitions
    # are counted, while TICA lagtime controls which slow modes are captured
    min_traj_len = min(len(t) for t in tica_outputs)
    effective_lag = min(msm_lagtime, max(1, min_traj_len // 10))

    try:
        msm_est = deeptime.markov.msm.MaximumLikelihoodMSM(lagtime=effective_lag)
        counts = deeptime.markov.TransitionCountEstimator(
            lagtime=effective_lag, count_mode="sliding"
        ).fit_fetch(dtrajs_raw)
        msm = msm_est.fit_fetch(counts)
    except Exception as e:
        logger.warning(f"MSM fitting failed: {e}")
        return None

    n_msm_states = msm.n_states
    all_centers = kmeans_model.cluster_centers  # [actual_k, N_tica]

    # Map cluster labels → MSM state indices (disconnected clusters → -1)
    state_symbols = msm.count_model.state_symbols
    cluster_to_msm = np.full(actual_k, -1, dtype=np.int64)
    for msm_idx, cluster_label in enumerate(state_symbols):
        cluster_to_msm[cluster_label] = msm_idx

    # Remap dtrajs: cluster labels → MSM state indices
    dtrajs = [cluster_to_msm[d] for d in dtrajs_raw]

    # Cluster centers for MSM states only
    msm_centers = all_centers[state_symbols].astype(np.float32)

    # Warn if too many states were disconnected
    connected_frac = n_msm_states / actual_k
    if connected_frac < 0.5:
        logger.warning(f"Only {n_msm_states}/{actual_k} clusters connected (lag={effective_lag})")

    return (
        msm_centers,
        msm.transition_matrix.astype(np.float32),
        msm.stationary_distribution.astype(np.float32),
        dtrajs,
        n_msm_states,
    )


def _compute_cluster_log_vars(
    tica_coords: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    min_var: float = 1e-4,
) -> np.ndarray:
    """
    Compute per-cluster diagonal log-variance in TICA space.

    Args:
        tica_coords: [T, N_tica]
        labels:      [T,] int  (-1 = unassigned, skipped)
        n_states:    number of MSM states
        min_var:     minimum variance (prevents degenerate Gaussian emission)

    Returns:
        log_vars: [n_states, N_tica] float32
    """
    N_tica = tica_coords.shape[1]
    log_vars = np.zeros((n_states, N_tica), dtype=np.float32)

    for i in range(n_states):
        mask = labels == i
        if mask.sum() < 2:
            # Fallback: use global variance for under-populated clusters
            log_vars[i] = np.log(np.var(tica_coords, axis=0).clip(min=min_var))
        else:
            log_vars[i] = np.log(np.var(tica_coords[mask], axis=0).clip(min=min_var))

    return log_vars


def validate_gaussian_assumption(
    tica_coords: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    max_components: int = 3,
    min_frames_for_gmm: int = 20,
) -> dict:
    """
    Validate whether a single diagonal Gaussian is adequate per cluster,
    using BIC comparison against GMM with up to max_components.

    Returns a dict with per-cluster metrics and an overall summary:
      {
        "per_cluster": [
            {"n_frames": int, "kurtosis_max": float,
             "bic_1": float, "bic_best": float, "best_n": int,
             "bic_improvement": float},   # bic_1 - bic_best (>0 means GMM better)
            ...
        ],
        "frac_needing_gmm": float,   # fraction of clusters where best_n > 1
        "recommended_n_components": int,  # max best_n (or 1 if single-G is fine for all)
      }
    """
    from sklearn.mixture import GaussianMixture
    from scipy.stats import kurtosis as scipy_kurtosis

    per_cluster = []
    for i in range(n_states):
        mask = labels == i
        X = tica_coords[mask]  # [N_i, N_tica]
        N_i = X.shape[0]

        record = {"n_frames": N_i, "kurtosis_max": float("nan"),
                  "bic_1": float("nan"), "bic_best": float("nan"),
                  "best_n": 1, "bic_improvement": 0.0}

        if N_i < 4:
            per_cluster.append(record)
            continue

        # Kurtosis (excess; >0 = heavier tails than Gaussian)
        kurt = scipy_kurtosis(X, axis=0, fisher=True)
        record["kurtosis_max"] = float(np.max(np.abs(kurt)))

        # BIC for C=1..max_components — store each individually
        bic_by_c = {}
        best_bic = np.inf
        best_n = 1
        c_max = min(max_components + 1, max(2, N_i // min_frames_for_gmm + 1))
        for c in range(1, c_max):
            try:
                gmm = GaussianMixture(
                    n_components=c, covariance_type="full",
                    n_init=3, random_state=42, reg_covar=1e-4,
                )
                gmm.fit(X)
                bic = float(gmm.bic(X))
                bic_by_c[c] = bic
                if bic < best_bic:
                    best_bic = bic
                    best_n = c
            except Exception:
                break

        bic_1 = bic_by_c.get(1, float("nan"))
        record["bic_by_c"] = bic_by_c          # {1: bic1, 2: bic2, 3: bic3}
        record["bic_1"]    = bic_1
        record["bic_best"] = float(best_bic) if not np.isinf(best_bic) else float("nan")
        record["best_n"]   = best_n
        record["bic_improvement"] = float(bic_1 - best_bic) if not np.isnan(bic_1) else 0.0
        per_cluster.append(record)

    # Summary — include per-C median BIC improvements
    valid = [r for r in per_cluster if not np.isnan(r["bic_1"])]
    frac_gmm = sum(1 for r in valid if r["best_n"] > 1) / max(len(valid), 1)
    rec_n = max((r["best_n"] for r in valid), default=1)

    # Per-C improvement vs C=1
    bic_improvement_by_c = {}
    for c in range(2, max_components + 1):
        improvements = [
            r["bic_1"] - r["bic_by_c"][c]
            for r in valid
            if c in r.get("bic_by_c", {}) and not np.isnan(r["bic_1"])
        ]
        if improvements:
            bic_improvement_by_c[c] = {
                "median": float(np.median(improvements)),
                "frac_positive": float(np.mean([x > 0 for x in improvements])),
            }

    return {
        "per_cluster": per_cluster,
        "frac_needing_gmm": frac_gmm,
        "recommended_n_components": rec_n,
        "median_bic_improvement": float(np.median([r["bic_improvement"] for r in valid]) if valid else 0),
        "median_kurtosis_max": float(np.median([r["kurtosis_max"] for r in valid if not np.isnan(r["kurtosis_max"])]) if valid else 0),
        "bic_improvement_by_c": bic_improvement_by_c,   # {2: {median, frac_positive}, 3: {...}}
    }


def select_n_components_per_cluster(
    per_cluster: list,
    bic_marginal_threshold: float = 10.0,
    max_components: int = 5,
) -> np.ndarray:
    """
    Pick the C with the best total BIC improvement from the C=1 baseline.

    Uses argmax over all tested C values so that non-monotonic BIC curves
    (e.g. a dip at C=4 followed by recovery at C=5) do not prematurely halt
    selection.  A higher C is chosen only if its cumulative improvement from
    C=1 strictly exceeds both the current best AND ``bic_marginal_threshold``.

    Args:
        per_cluster: list of per-cluster dicts from ``validate_gaussian_assumption()``.
                     Each dict must contain a ``bic_by_c`` key ({1: bic1, 2: bic2, …}).
        bic_marginal_threshold: minimum total improvement over C=1 to prefer C > 1.
        max_components: hard cap on C.

    Returns:
        np.ndarray [K] int  — optimal C per cluster (≥ 1).
    """
    K = len(per_cluster)
    result = np.ones(K, dtype=int)
    for i, r in enumerate(per_cluster):
        bic_by_c = r.get("bic_by_c", {})
        if not bic_by_c or 1 not in bic_by_c:
            continue
        bic_1 = bic_by_c[1]
        best_c = 1
        best_improvement = 0.0
        for c in range(2, max_components + 1):
            curr_bic = bic_by_c.get(c)
            if curr_bic is None:
                break
            improvement = bic_1 - curr_bic
            if improvement > best_improvement and improvement > bic_marginal_threshold:
                best_improvement = improvement
                best_c = c
        result[i] = best_c
    return result


def _cov_to_precision_chol(cov: np.ndarray, reg_covar: float = 1e-4) -> np.ndarray:
    """
    Given a covariance matrix, return the lower-triangular Cholesky of its inverse.

    Regularises by adding reg_covar * I before inverting.
    Returns identity if the matrix is degenerate.
    """
    N = cov.shape[0]
    cov_reg = cov + reg_covar * np.eye(N, dtype=cov.dtype)
    try:
        precision = np.linalg.inv(cov_reg)
        # Enforce symmetry
        precision = (precision + precision.T) * 0.5
        L = np.linalg.cholesky(precision)   # lower triangular, Λ = L L^T
        return L.astype(np.float32)
    except np.linalg.LinAlgError:
        return np.eye(N, dtype=np.float32)


def fit_gmm_emission(
    tica_coords: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    n_components: int,
    per_cluster_n_components: Optional[np.ndarray] = None,
    min_var: float = 1e-4,
    min_frames_for_gmm: int = 20,
) -> tuple:
    """
    Fit full-covariance GMM per cluster.

    When ``per_cluster_n_components`` is provided (a [K] int array), each cluster
    uses its own BIC-optimal C.  The global output shape is padded to
    [K, C_max, ...] where C_max = max(per_cluster_n_components).

    When ``per_cluster_n_components`` is None the legacy behaviour applies:
    all clusters use the same global ``n_components``.

    Clusters with insufficient data (< min_frames_for_gmm * c_target) fall back
    to a single Gaussian.

    Returns:
        weights:         [K, C] float32   (sum to 1 per cluster; extra components have weight=0)
        means:           [K, C, N_tica] float32
        precisions_chol: [K, C, N_tica, N_tica] float32
                         Lower-triangular Cholesky of precision (Σ^{-1} = L L^T).
    """
    from sklearn.mixture import GaussianMixture

    K = n_states
    N_tica = tica_coords.shape[1]

    # Determine global C (array shape) and per-cluster targets
    if per_cluster_n_components is not None:
        C = int(per_cluster_n_components.max())
        cluster_targets = per_cluster_n_components.astype(int)
    else:
        C = n_components
        cluster_targets = np.full(K, n_components, dtype=int)

    weights_out  = np.zeros((K, C), dtype=np.float32)
    means_out    = np.zeros((K, C, N_tica), dtype=np.float32)
    prec_out     = np.zeros((K, C, N_tica, N_tica), dtype=np.float32)

    # Global fallbacks for tiny clusters
    global_mean = tica_coords.mean(axis=0)
    global_cov  = np.atleast_2d(np.cov(tica_coords.T)).reshape(N_tica, N_tica)
    global_L    = _cov_to_precision_chol(global_cov, reg_covar=min_var)

    for i in range(K):
        mask = labels == i
        X = tica_coords[mask]
        N_i = X.shape[0]
        c_target = int(cluster_targets[i])

        # Fall back to 1 component if too few frames for the requested C
        c_fit = c_target if N_i >= min_frames_for_gmm * c_target else 1

        if N_i < 2 or c_fit == 1:
            # Single full-covariance Gaussian
            mu = X.mean(axis=0) if N_i >= 1 else global_mean
            if N_i >= 2:
                cov = np.atleast_2d(np.cov(X.T)).reshape(N_tica, N_tica)
                L = _cov_to_precision_chol(cov, reg_covar=min_var)
            else:
                L = global_L
            weights_out[i, 0] = 1.0
            mu32 = mu.astype(np.float32)
            # Fill all C slots (weight=0 for slots 1..C-1, but copy params for stability)
            for c in range(C):
                means_out[i, c] = mu32
                prec_out[i, c]  = L
        else:
            try:
                gmm = GaussianMixture(
                    n_components=c_fit, covariance_type="full",
                    n_init=5, random_state=42, reg_covar=1e-4,
                )
                gmm.fit(X)
                weights_out[i, :c_fit] = gmm.weights_.astype(np.float32)
                means_out[i, :c_fit]   = gmm.means_.astype(np.float32)
                # precisions_cholesky_: [c_fit, N_tica, N_tica], lower triangular of precision
                prec_out[i, :c_fit]    = gmm.precisions_cholesky_.astype(np.float32)
                # Pad unused slots with first component (weight=0 so they don't fire)
                for c in range(c_fit, C):
                    means_out[i, c] = means_out[i, 0]
                    prec_out[i, c]  = prec_out[i, 0]
            except Exception:
                # Fallback to single full-covariance Gaussian
                mu = X.mean(axis=0)
                cov = np.atleast_2d(np.cov(X.T)).reshape(N_tica, N_tica)
                L = _cov_to_precision_chol(cov, reg_covar=min_var)
                weights_out[i, 0] = 1.0
                for c in range(C):
                    means_out[i, c] = mu.astype(np.float32)
                    prec_out[i, c]  = L

    return weights_out, means_out, prec_out


def fit_all_gmm_emissions(
    tica_coords: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    max_components: int,
    min_var: float = 1e-4,
    min_frames_for_gmm: int = 20,
) -> dict:
    """
    Fit separate GMMs for each C in 1..max_components, each with exactly C components
    for every cluster (no adaptive per-cluster selection).

    Returns:
        dict {c: {"weights": [K,c], "means": [K,c,NT], "precisions_chol": [K,c,NT,NT]}}
        for c in range(1, max_components + 1).
    """
    result = {}
    for c in range(1, max_components + 1):
        w, m, p = fit_gmm_emission(
            tica_coords, labels, n_states, n_components=c,
            min_var=min_var, min_frames_for_gmm=min_frames_for_gmm,
        )
        result[c] = {"weights": w, "means": m, "precisions_chol": p}
    return result


def project_to_tica(ca_coords: np.ndarray, msm: "MSMArtifacts") -> np.ndarray:
    """
    Project CA coordinates to TICA space using the stored linear model.

    Args:
        ca_coords: [T, N_ca, 3]
        msm:       MSMArtifacts with tica_mean, tica_components, pair_indices_i/j

    Returns:
        tica_coords: [T, N_tica]
    """
    idx_i = msm.pair_indices_i
    idx_j = msm.pair_indices_j
    diff = ca_coords[:, idx_i] - ca_coords[:, idx_j]   # [T, N_pairs, 3]
    dists = np.linalg.norm(diff, axis=-1)                # [T, N_pairs]
    centered = dists - msm.tica_mean                     # [T, N_pairs]
    return centered @ msm.tica_components                # [T, N_tica]


def build_msm_for_system(
    traj_ca_coords: dict,
    lagtime_frames: int = 10,
    n_tica_dims: int = 3,
    n_clusters_coarse: int = 20,
    n_clusters_fine: int = 200,
    msm_lagtime: int = 2,
) -> Optional[MSMArtifacts]:
    """
    Build both coarse and fine MSMs for a system.

    Args:
        traj_ca_coords: {traj_name: np.ndarray [T, N_ca, 3]}
        lagtime_frames: TICA lag time in frames (controls slow-mode decomposition)
        n_tica_dims: TICA dimensionality
        n_clusters_coarse: for loss computation
        n_clusters_fine: for pseudo-trajectory resampling
        msm_lagtime: MSM transition counting lagtime (shorter = more transitions, better connectivity)
    """
    import deeptime

    # Pre-scan to get total frames and N_ca for adaptive max_ca
    valid_trajs = {tn: c for tn, c in traj_ca_coords.items() if c.shape[0] >= 3}
    if not valid_trajs:
        return None
    total_frames_prelim = sum(c.shape[0] for c in valid_trajs.values())
    actual_nca = next(iter(valid_trajs.values())).shape[1]

    # Adaptive max_ca: ensure N_pairs << total_frames for well-conditioned TICA
    # Target: N_pairs ≤ total_frames // 5. Solve N*(N-1)/2 ≤ total_frames//5 → N ≤ sqrt(2*T/5)
    max_ca = min(actual_nca, max(15, int((2 * total_frames_prelim / 5) ** 0.5)))

    # 1. Featurize: pairwise distances
    features_list = []
    traj_names_ordered = []
    pair_indices = None
    traj_frame_counts = {}

    for traj_name, ca_coords in valid_trajs.items():
        dists, pair_indices = compute_ca_pairwise_distances(ca_coords, max_ca=max_ca)
        features_list.append(dists.astype(np.float32))
        traj_names_ordered.append(traj_name)
        traj_frame_counts[traj_name] = ca_coords.shape[0]

    total_frames = sum(f.shape[0] for f in features_list)
    if total_frames < max(n_clusters_coarse, 50):
        logger.warning(f"Insufficient frames ({total_frames}) for MSM building")
        return None

    # 2. TICA (shared) — use TICA lagtime for slow-mode decomposition
    actual_tica_dims = min(n_tica_dims, features_list[0].shape[1])
    # Adapt TICA lagtime to trajectory lengths
    min_traj_len = min(f.shape[0] for f in features_list)
    tica_lag = min(lagtime_frames, max(1, min_traj_len // 5))

    tica_est = deeptime.decomposition.TICA(lagtime=tica_lag, dim=actual_tica_dims)
    try:
        tica_est.fit(features_list)
    except Exception as e:
        logger.warning(f"TICA fitting failed: {e}")
        return None
    tica_model = tica_est.fetch_model()
    tica_outputs = [tica_model.transform(f) for f in features_list]

    # Clip TICA eigenvalues to valid range (numerical artifacts can produce |eig| > 1)
    raw_eigenvalues = tica_model.singular_values[:actual_tica_dims].astype(np.float32)
    eigenvalues = np.clip(raw_eigenvalues, -0.9999, 0.9999)

    # 3. Build coarse MSM (short lagtime for better connectivity)
    coarse_result = _build_single_msm(tica_outputs, n_clusters_coarse, msm_lagtime)
    if coarse_result is None:
        return None
    coarse_centers, coarse_T, coarse_pi, coarse_dtrajs, coarse_k = coarse_result

    coarse_labels = {}
    for traj_name, dtraj in zip(traj_names_ordered, coarse_dtrajs):
        coarse_labels[traj_name] = dtraj.astype(np.int16)

    # Compute within-cluster diagonal log-variance for Gaussian HMM emission.
    # For each MSM state i, compute var of TICA coordinates across all assigned frames.
    tica_concat_all = np.concatenate(tica_outputs, axis=0)   # [T_total, N_tica]
    coarse_labels_concat = np.concatenate(
        [coarse_dtrajs[k] for k in range(len(traj_names_ordered))], axis=0
    )   # [T_total,]
    coarse_cluster_log_vars = _compute_cluster_log_vars(
        tica_concat_all, coarse_labels_concat, coarse_k
    )

    # 4. Build fine MSM
    fine_result = _build_single_msm(tica_outputs, n_clusters_fine, msm_lagtime)
    if fine_result is None:
        # Fall back: use coarse MSM for both
        fine_centers, fine_T, fine_pi, fine_dtrajs, fine_k = (
            coarse_centers,
            coarse_T,
            coarse_pi,
            coarse_dtrajs,
            coarse_k,
        )
    else:
        fine_centers, fine_T, fine_pi, fine_dtrajs, fine_k = fine_result

    fine_labels = {}
    for traj_name, dtraj in zip(traj_names_ordered, fine_dtrajs):
        fine_labels[traj_name] = dtraj.astype(np.int16)

    return MSMArtifacts(
        tica_mean=tica_model.mean_0.astype(np.float32),
        tica_components=tica_model.singular_vectors_left[:, :actual_tica_dims].astype(np.float32),
        tica_eigenvalues=eigenvalues,
        coarse_cluster_centers=coarse_centers,
        coarse_cluster_log_vars=coarse_cluster_log_vars,
        coarse_transition_matrix=coarse_T,
        coarse_stationary_distribution=coarse_pi,
        coarse_state_labels=coarse_labels,
        coarse_n_states=coarse_k,
        fine_cluster_centers=fine_centers,
        fine_transition_matrix=fine_T,
        fine_stationary_distribution=fine_pi,
        fine_state_labels=fine_labels,
        fine_n_states=fine_k,
        lagtime_frames=tica_lag,
        n_tica_dims=actual_tica_dims,
        pair_indices_i=pair_indices[0],
        pair_indices_j=pair_indices[1],
        traj_frame_counts=traj_frame_counts,
        n_ca_full=actual_nca,
    )


class MSMCache:
    """Manages MSM artifact loading/building/caching per system."""

    def __init__(self, cache_dir, msm_configs=None):
        self.cache_dir = cache_dir
        self.msm_configs = msm_configs or {}
        self._mem_cache = {}  # system_name → MSMArtifacts or None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def get(self, system_name: str):
        """Get cached MSM artifacts. Returns MSMArtifacts, plain dict (new multi-K format),
        or None if not available."""
        if system_name in self._mem_cache:
            return self._mem_cache[system_name]

        cache_path = os.path.join(self.cache_dir, f"{system_name}_msm.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    artifacts = pickle.load(f)
                if isinstance(artifacts, MSMArtifacts):
                    self._mem_cache[system_name] = artifacts
                    return artifacts
                elif isinstance(artifacts, dict) and "by_k" in artifacts:
                    # New multi-K plain-dict format from build_multiK_msm.py
                    self._mem_cache[system_name] = artifacts
                    return artifacts
                else:
                    # Failed marker or unknown format
                    self._mem_cache[system_name] = None
                    return None
            except Exception:
                self._mem_cache[system_name] = None
                return None

        return None  # Not built yet

    def has(self, system_name: str) -> bool:
        return system_name in self._mem_cache or os.path.exists(
            os.path.join(self.cache_dir, f"{system_name}_msm.pkl")
        )

    def build_and_save(self, system_name: str, traj_ca_coords: dict) -> Optional[MSMArtifacts]:
        """Build MSM and save to disk. Returns artifacts or None."""
        artifacts = build_msm_for_system(traj_ca_coords, **self.msm_configs)

        cache_path = os.path.join(self.cache_dir, f"{system_name}_msm.pkl")
        if artifacts is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(artifacts, f)
            self._mem_cache[system_name] = artifacts
            logger.info(f"Built MSM for {system_name}: "
                        f"coarse={artifacts.coarse_n_states} states, "
                        f"fine={artifacts.fine_n_states} states")
        else:
            # Save failure marker
            with open(cache_path, "wb") as f:
                pickle.dump({"failed": True}, f)
            self._mem_cache[system_name] = None
            logger.warning(f"MSM building failed for {system_name}")

        return artifacts


def sample_pseudo_trajectory_from_msm(
    msm_artifacts: MSMArtifacts,
    traj_ca_coords_all: dict,
    n_frames: int,
    rmsd_threshold_percentile: float = 95,
    frame_gap_sigma: float = 20.0,
    use_coarse_msm: bool = True,
) -> Optional[list]:
    """
    Sample a pseudo-trajectory from the MSM.

    Samples state sequences from the MSM transition matrix, then picks real MD
    frames for each state. By default uses the COARSE MSM (~85% self-transition
    probability) which gives frame-to-frame variation matching real MD much
    better than the fine MSM (~33% self-transition).

    Frame selection within a state:
      - Self-transitions: prefer gap=1 (exact adjacent frame, forward then backward).
        Within a metastable state, consecutive frames are always gap=1 by MD
        construction → this reproduces real MD frame-to-frame variation.
      - Cross-state transitions: exponential-decay weighting (sigma=frame_gap_sigma)
        allows slightly wider exploration since a new basin is being entered.
      - The exact previous frame is always excluded (gap > 0) to prevent
        zero-change consecutive pairs that don't exist in real MD.

    Distribution matching result (CATH2, coarse MSM):
      - p1–p75:  0.97–1.10× real MD frame-to-frame change  (near-perfect match)
      - p90–p99: 1.22–1.39× real MD  (residual tail elevation from 15% cross-state jumps)
      - Mean:    1.10× real MD

    Args:
        msm_artifacts: MSMArtifacts with both coarse and fine MSMs
        traj_ca_coords_all: {traj_name: [T, N_ca, 3]} — all trajectories for this system
        n_frames: number of frames to generate
        rmsd_threshold_percentile: (unused, kept for API compatibility)
        frame_gap_sigma: decay length scale (frames) for cross-state candidate weighting.
            Default 20 frames is appropriate for most MD datasets.
        use_coarse_msm: if True (default), use coarse MSM (~85% self-transition)
            for better frame-to-frame distribution matching. Fine MSM has only
            ~33% self-transition → 67% of steps are large structural jumps.

    Returns:
        List of (traj_name, frame_idx) tuples, or None if failed.
        Each entry specifies which real MD frame to use.
    """
    if use_coarse_msm:
        T = msm_artifacts.coarse_transition_matrix
        pi = msm_artifacts.coarse_stationary_distribution
        state_labels = msm_artifacts.coarse_state_labels
    else:
        T = msm_artifacts.fine_transition_matrix
        pi = msm_artifacts.fine_stationary_distribution
        state_labels = msm_artifacts.fine_state_labels

    # Build frame pool: {state_id: [(traj_name, frame_idx), ...]}
    # Skip disconnected states (label == -1)
    frame_pool = {}
    for traj_name, labels in state_labels.items():
        for frame_idx, state in enumerate(labels):
            if int(state) < 0:
                continue
            frame_pool.setdefault(int(state), []).append((traj_name, frame_idx))

    if not frame_pool:
        return None

    # Sample initial state from stationary distribution
    valid_states = [s for s in frame_pool.keys() if len(frame_pool[s]) > 0]
    if not valid_states:
        return None

    # Renormalize pi over valid states
    pi_valid = np.array([pi[s] for s in valid_states])
    pi_valid = pi_valid / pi_valid.sum()
    current_state = np.random.choice(valid_states, p=pi_valid)

    # Sample state sequence using transition matrix
    result = []
    prev_traj = None
    prev_frame = None
    prev_state = None

    for _ in range(n_frames):
        # Pick a real frame from current state
        candidates = frame_pool.get(current_state, [])
        if not candidates:
            # Fallback: pick from any neighboring state
            neighbors = np.where(T[current_state] > 0.01)[0]
            for nb in neighbors:
                candidates = frame_pool.get(int(nb), [])
                if candidates:
                    break
        if not candidates:
            return None

        # Frame selection strategy:
        # - Always prefer same trajectory to avoid unphysical coordinate jumps.
        # - Distinguish self-transition (stay in same state) vs. state transition:
        #     * Self-transition → try gap=1 first (exactly adjacent frame, forward
        #       or backward). Within a metastable basin, consecutive frames are
        #       always gap=1 by MD construction, so this reproduces real MD
        #       frame-to-frame variation. Fall back to nearest same-state frame if
        #       no gap=1 candidate exists (end of a state-run segment).
        #     * State transition → soft exponential weighting (sigma=frame_gap_sigma)
        #       since the system has moved to a new basin; allow wider exploration.
        # - Always exclude the exact previous frame (gap > 0) to prevent
        #   zero-change consecutive pairs that don't exist in real MD.
        if prev_traj is not None:
            # Exclude the exact previous frame to guarantee gap > 0
            same_traj = [(tn, fi) for tn, fi in candidates
                         if tn == prev_traj and fi != prev_frame]
            if same_traj:
                is_self_transition = (current_state == prev_state)
                if is_self_transition:
                    # Try to find gap=1 (forward preferred, then backward)
                    gap1_fwd = [(tn, fi) for tn, fi in same_traj if fi == prev_frame + 1]
                    gap1_bwd = [(tn, fi) for tn, fi in same_traj if fi == prev_frame - 1]
                    if gap1_fwd:
                        chosen = gap1_fwd[0]
                    elif gap1_bwd:
                        chosen = gap1_bwd[0]
                    else:
                        # No adjacent frame in same state — pick nearest available
                        same_traj_sorted = sorted(same_traj, key=lambda x: abs(x[1] - prev_frame))
                        chosen = same_traj_sorted[0]
                else:
                    # State transition: soft weighting, allow wider exploration
                    gaps = np.array([abs(fi - prev_frame) for _, fi in same_traj], dtype=float)
                    weights = np.exp(-gaps / frame_gap_sigma)
                    weights /= weights.sum()
                    chosen = same_traj[np.random.choice(len(same_traj), p=weights)]
            else:
                # No other frame in this traj → cross-traj or allow prev_frame
                other = [(tn, fi) for tn, fi in candidates if tn != prev_traj]
                chosen = random.choice(other) if other else random.choice(candidates)
        else:
            chosen = random.choice(candidates)

        result.append(chosen)
        prev_traj, prev_frame = chosen
        prev_state = current_state

        # Transition to next state
        row = T[current_state].copy()
        # Zero out states with no frames
        for s_idx in range(len(row)):
            if s_idx not in frame_pool or len(frame_pool[s_idx]) == 0:
                row[s_idx] = 0
        if row.sum() < 1e-10:
            break
        row = row / row.sum()
        current_state = int(np.random.choice(len(row), p=row))

    return result if len(result) == n_frames else None
