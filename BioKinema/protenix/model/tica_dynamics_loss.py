"""
TICA-space Dynamics Losses — HMM-based formulation.

Three loss components sharing a single TICA projection:
  1. HMMForwardLoss: -log P(z_{0:N-1}) via forward algorithm with Gaussian emission
       — captures transition dynamics AND thermodynamics (π) simultaneously via path probability
  2. PopulationLoss: KL(traj_pop || q_avg)
       — compares model's state distribution against the TRAJECTORY's own empirical population
       — NOT against MSM stationary distribution π (which spans the full conformational space
         while a single trajectory legitimately stays in a small region)
  3. AutocorrelationLoss: MSE(C_pred(dt), λ^n) — relaxation timescales

All are differentiable through:
    pred_coords → Cα pairwise distances → TICA projection → Gaussian emission → HMM → loss

Gaussian emission (replaces SoftStateAssigner):
    P(z_t | s_t = i) = N(z_t ; μ_i, diag(σ²_i))
    μ_i, σ²_i: precomputed per-cluster from MD data — no temperature hyperparameter.

HMM forward algorithm (exact, O(N_frame * K²)):
    Marginalizes over all K^N state paths, accounting for temporal correlations.
    Avoids the independence assumption in the outer-product approximation.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Diagnostic dump (toggled via env var TICA_DIAG_DUMP_DIR) ──────────────────
# When set, each call to TICADynamicsLoss.forward saves intermediates to:
#   {TICA_DIAG_DUMP_DIR}/{traj_name}__{uuid}.npz
# Set in val/eval path only; unset in training steps.
def _tica_diag_dump_dir():
    return os.environ.get("TICA_DIAG_DUMP_DIR", "")


# ── TICA Projector ────────────────────────────────────────────────────────────

class TICAProjector(nn.Module):
    """
    Differentiable TICA projection from Cα coordinates to TICA space.

    Chain: Cα coords → pairwise distances (rotation-invariant) → TICA projection (linear)
    Parameters (tica_mean, tica_components) are frozen constants from offline MSM building.
    """

    def forward(self, ca_coords, tica_mean, tica_components, pair_indices_i, pair_indices_j):
        """
        Args:
            ca_coords:        [N_frame, N_sample, N_ca, 3]
            tica_mean:        [N_pairs,]
            tica_components:  [N_pairs, N_tica]
            pair_indices_i/j: [N_pairs,]  long

        Returns:
            tica_coords: [N_frame, N_sample, N_tica]
        """
        diff = ca_coords[:, :, pair_indices_i] - ca_coords[:, :, pair_indices_j]
        pair_dists = torch.norm(diff, dim=-1)              # [N_frame, N_sample, N_pairs]
        centered = pair_dists - tica_mean
        return torch.einsum("...p,pd->...d", centered, tica_components)


# ── Gaussian / GMM Emission ───────────────────────────────────────────────────

def _gaussian_log_emission(z, centers, log_vars):
    """
    Single diagonal Gaussian log-probability per cluster (backward compat).

    Args:
        z:        [N_frame, N_sample, N_tica]
        centers:  [K, N_tica]
        log_vars: [K, N_tica]

    Returns:
        log_emit: [N_frame, N_sample, K]
    """
    diff = z.unsqueeze(-2) - centers      # [N_frame, N_sample, K, N_tica]
    log_emit = -0.5 * (
        diff ** 2 / log_vars.exp() + log_vars + math.log(2.0 * math.pi)
    ).sum(-1)
    return log_emit   # [N_frame, N_sample, K]


def _gmm_log_emission(z, gmm_weights, gmm_means, gmm_log_vars):
    """
    Diagonal GMM log-probability per cluster (legacy backward-compat path).

    Args:
        z:             [N_frame, N_sample, N_tica]
        gmm_weights:   [K, C]
        gmm_means:     [K, C, N_tica]
        gmm_log_vars:  [K, C, N_tica]

    Returns:
        log_emit: [N_frame, N_sample, K]
    """
    z_exp = z.unsqueeze(-2).unsqueeze(-2)          # [NF, NS, 1, 1, NT]
    mu    = gmm_means.unsqueeze(0).unsqueeze(0)    # [1,  1,  K, C, NT]
    lv    = gmm_log_vars.unsqueeze(0).unsqueeze(0) # [1,  1,  K, C, NT]
    diff  = z_exp - mu                             # [NF, NS, K, C, NT]
    log_gauss = -0.5 * (
        diff ** 2 / lv.exp() + lv + math.log(2.0 * math.pi)
    ).sum(-1)
    log_w = torch.log(gmm_weights.clamp(min=1e-10)).unsqueeze(0).unsqueeze(0)
    return torch.logsumexp(log_gauss + log_w, dim=-1)


def _gmm_log_emission_full(z, gmm_weights, gmm_means, gmm_precisions_chol):
    """
    Full-covariance GMM log-probability per cluster.

    Uses the Cholesky decomposition of the precision matrix for an efficient,
    numerically stable Mahalanobis distance computation:

        (x-μ)^T Σ^{-1} (x-μ) = ||L^T (x-μ)||²   where  Σ^{-1} = L L^T

    Args:
        z:                   [N_frame, N_sample, N_tica]
        gmm_weights:         [K, C]
        gmm_means:           [K, C, N_tica]
        gmm_precisions_chol: [K, C, N_tica, N_tica]  lower-triangular L  (Σ^{-1}=LL^T)

    Returns:
        log_emit: [N_frame, N_sample, K]
    """
    NT = z.shape[-1]

    # diff: [NF, NS, K, C, NT]
    diff = z.unsqueeze(-2).unsqueeze(-2) - gmm_means.unsqueeze(0).unsqueeze(0)

    # Mahalanobis: y = L^T @ diff,  mahal = ||y||²
    # L: [K, C, NT, NT] → broadcast [1, 1, K, C, NT, NT]
    L = gmm_precisions_chol.unsqueeze(0).unsqueeze(0)        # [1, 1, K, C, NT, NT]
    diff_col = diff.unsqueeze(-1)                             # [NF, NS, K, C, NT, 1]
    y = torch.matmul(L.transpose(-1, -2), diff_col)          # [NF, NS, K, C, NT, 1]
    mahal = (y.squeeze(-1) ** 2).sum(-1)                     # [NF, NS, K, C]

    # log|Σ| = −2 · Σ_d log L_{dd}   (L is lower triangular, diagonal > 0)
    log_det_sigma = -2.0 * gmm_precisions_chol.diagonal(
        dim1=-2, dim2=-1
    ).clamp(min=1e-8).log().sum(-1)                           # [K, C]

    log_gauss = -0.5 * (
        mahal
        + log_det_sigma.unsqueeze(0).unsqueeze(0)
        + NT * math.log(2.0 * math.pi)
    )                                                         # [NF, NS, K, C]

    log_w = torch.log(gmm_weights.clamp(min=1e-10)).unsqueeze(0).unsqueeze(0)
    return torch.logsumexp(log_gauss + log_w, dim=-1)         # [NF, NS, K]


# ── Transition Matrix Power ───────────────────────────────────────────────────

def _compute_log_T_power(T, k, pi):
    """
    Compute log(T^k) where k = actual_frame_spacing / lagtime_frames.

    By design (msm_lagtime == lagtime_frames, and training spacing is always a
    multiple of lagtime_frames), k is always a positive integer — no fractional
    powers needed. Uses torch.linalg.matrix_power for efficiency.

    pi is unused (kept for API compatibility).
    """
    k_int = max(1, int(round(k)))
    if k_int == 1:
        return torch.log(T.clamp(min=1e-30))
    Tk = torch.linalg.matrix_power(T.double(), k_int).to(T.dtype)
    # Re-normalize rows against numerical drift
    Tk = Tk.clamp(min=0.0)
    Tk = Tk / Tk.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return torch.log(Tk.clamp(min=1e-30))


# ── HMM Forward Algorithm ─────────────────────────────────────────────────────

def hmm_forward_log(log_pi, log_Tk, log_emit):
    """
    HMM forward algorithm in log space — exact marginal log-likelihood.

    Computes log P(z_{0:N-1}) by summing over all K^N state paths in O(N·K²).

    Recursion:
        log α_0[j] = log π(j) + log P(z_0 | j)
        log α_{t+1}[j] = log P(z_{t+1}|j) + logsumexp_i(log α_t[i] + log T^k[i,j])
        log P(z_{0:N-1}) = logsumexp_j(log α_{N-1}[j])

    Args:
        log_pi:   [K]           log initial state distribution
        log_Tk:   [K, K]        log T^k[i, j]  (i=row → j=col)
        log_emit: [N_frame, K]  log emission probabilities for one sample

    Returns:
        log_likelihood: scalar
    """
    log_alpha = log_pi + log_emit[0]      # [K]

    for t in range(1, log_emit.shape[0]):
        # log_alpha[i] + log_Tk[i, j] → [K, K], logsumexp over i (dim=0) → [K]
        log_sum = torch.logsumexp(log_alpha.unsqueeze(1) + log_Tk, dim=0)
        log_alpha = log_emit[t] + log_sum

    return torch.logsumexp(log_alpha, dim=0)   # scalar


# ── Main Loss Module ──────────────────────────────────────────────────────────

class TICADynamicsLoss(nn.Module):
    """
    HMM-based TICA dynamics loss combining three components.

    The HMM is defined by:
        Hidden states  s_t ∈ {0,...,K-1}            (MSM coarse states)
        Initial dist.  π                             (MSM stationary distribution)
        Transitions    T^k                           (MSM transition matrix raised to frame gap)
        Emission       P(z_t|s_t=i) = N(z_t;μ_i,Σ_i)  (data-derived Gaussian)
    """

    def __init__(
        self,
        weight_transition=1.0,
        weight_population=0.1,
        weight_autocorrelation=0.5,
        max_acf_lag=5,
        log_var_min=-6.0,   # σ² ≥ e^{-6} ≈ 0.0025 Å²  (prevents degenerate emission)
        # Backward-compat: ignored kwargs from old API
        temperature=None,
    ):
        super().__init__()
        self.projector = TICAProjector()
        self.weight_transition = weight_transition
        self.weight_population = weight_population
        self.weight_autocorrelation = weight_autocorrelation
        self.max_acf_lag = max_acf_lag
        self.log_var_min = log_var_min

    # ── Sub-losses ────────────────────────────────────────────────────────────

    def _hmm_forward_loss(self, log_pi, log_Tk, log_emit):
        """
        -log P(z_{0:N-1}) averaged over N_sample.

        Args:
            log_pi:   [K]
            log_Tk:   [K, K]
            log_emit: [N_frame, N_sample, K]

        Returns:
            scalar  (positive; lower = trajectory more likely under MSM)
        """
        N_frame, N_sample, K = log_emit.shape
        if N_frame < 2:
            return torch.zeros(1, device=log_pi.device).squeeze()

        log_lls = torch.stack([
            hmm_forward_log(log_pi, log_Tk, log_emit[:, s, :])
            for s in range(N_sample)
        ])
        return -log_lls.mean() / log_emit.shape[0]

    def _population_loss(self, log_emit, traj_population, stationary_distribution):
        """
        KL(traj_pop || q_avg): match the model trajectory's time-averaged soft
        state distribution to THIS trajectory's empirical distribution traj_pop,
        using the MSM stationary distribution π as the Bayesian prior.

        Per-frame soft assignment (Bayesian posterior with π as prior):
            q_t[k] ∝ P(z_t | s=k) · π[k]
        Time / sample average:
            q_avg[k] = <q_t[k]>
        Loss:
            KL(traj_pop || q_avg)

        Why π as prior (and traj_pop as target):
          Using traj_pop as BOTH prior and target is information leakage —
          a uniform emission trivially gives q_avg = traj_pop and loss = 0.
          π is the global equilibrium distribution — a natural uninformative prior
          that is independent of the target, so emission carries the entire signal.

          Since any state visited by this trajectory must have π[k] > 0,
          supp(traj_pop) ⊆ supp(π). The posterior q_avg therefore stays positive
          wherever traj_pop is positive, so KL(traj_pop || q_avg) is always finite —
          no numerical guards needed.

        Args:
            log_emit:                [N_frame, N_sample, K]
            traj_population:         [K]  empirical distribution of this trajectory (target)
            stationary_distribution: [K]  MSM stationary π (Bayesian prior)

        Returns:
            KL(traj_pop || q_avg)  — scalar, ≥ 0
        """
        log_pi = torch.log(stationary_distribution.clamp(min=1e-8))     # [K]
        log_posterior = log_emit + log_pi                                # [NF, NS, K]
        log_q = log_posterior - torch.logsumexp(log_posterior, dim=-1, keepdim=True)
        q_avg = log_q.exp().mean(dim=(0, 1)).clamp(min=1e-8)            # [K]
        return F.kl_div(q_avg.log(), traj_population, reduction="sum", log_target=False)

    _collapse_dbg_count = 0

    def _autocorrelation_loss(self, tica_coords, tica_eigenvalues, frame_time_gap_ratio,
                              _diag_traj_name=None, _diag_pair_dists=None):
        """
        MSE between empirical and analytical TICA autocorrelation.
        Analytical: C_i(n·τ) = λ_i^n  (eigenvectors decorrelate exponentially).

        Numerical guard: TICA dims with near-zero variance in the predicted trajectory
        produce c_pred/var_z blow-up. We floor var_z and mask out degenerate dims so a
        single collapsed dim cannot dominate the loss.

        Args:
            tica_coords:          [N_frame, N_sample, N_tica]
            tica_eigenvalues:     [N_tica,]
            frame_time_gap_ratio: float  (actual_spacing / lagtime_frames; always a positive integer)
        """
        N_frame = tica_coords.shape[0]
        if N_frame < 3:
            return torch.zeros(1, device=tica_coords.device).squeeze()

        mean_z = tica_coords.mean(dim=0, keepdim=True)
        centered = tica_coords - mean_z
        var_z = centered.var(dim=0)    # [N_sample, N_tica]

        var_floor = 1e-3
        valid = (var_z > var_floor).float()                    # [N_sample, N_tica]
        var_z_safe = var_z.clamp(min=var_floor)
        n_valid = valid.sum().clamp(min=1.0)

        # ── Root-cause diagnostic: print only when variance collapses ──
        if (
            var_z.min().item() < var_floor
            and TICADynamicsLoss._collapse_dbg_count < 20
        ):
            try:
                import torch.distributed as _dist
                _rank = _dist.get_rank() if _dist.is_initialized() else 0
            except Exception:
                _rank = 0
            n_bad = int((var_z < var_floor).sum().item())
            n_total = int(var_z.numel())
            worst_sample, worst_dim = torch.unravel_index(
                var_z.argmin(), var_z.shape
            )
            ws, wd = int(worst_sample.item()), int(worst_dim.item())
            raw_traj = tica_coords[:, ws, wd].detach().float().cpu().tolist()
            col_per_dim = ((var_z < var_floor).sum(dim=0)).detach().cpu().tolist()
            comp_norm = tica_eigenvalues.detach().float().cpu().tolist()
            print(
                f"\n[ACF-COLLAPSE rank={_rank} "
                f"traj={_diag_traj_name!r} "
                f"N_frame={N_frame} N_sample={tica_coords.shape[1]} N_tica={tica_coords.shape[2]}] "
                f"bad_dims={n_bad}/{n_total}  "
                f"worst=(sample={ws}, dim={wd}) var={var_z[ws, wd].item():.3g}\n"
                f"  raw tica_coords over frames (sample={ws}, dim={wd}): {raw_traj}\n"
                f"  #collapsed samples per dim (len=N_tica): {col_per_dim}\n"
                f"  eigenvalues: {comp_norm}\n"
                f"  frame_time_gap_ratio: {frame_time_gap_ratio}",
                flush=True,
            )
            if _diag_pair_dists is not None:
                pd = _diag_pair_dists[:, ws, :].detach().float()  # [N_frame, N_pairs]
                pd_std_per_pair = pd.std(dim=0)
                print(
                    f"  pair_dists: shape={tuple(pd.shape)} "
                    f"std_per_pair[min/median/max]="
                    f"{pd_std_per_pair.min().item():.3g}/"
                    f"{pd_std_per_pair.median().item():.3g}/"
                    f"{pd_std_per_pair.max().item():.3g} "
                    f"frame_var_sum={(pd.var(dim=0).sum()).item():.3g}",
                    flush=True,
                )
            TICADynamicsLoss._collapse_dbg_count += 1

        total, count = torch.zeros(1, device=tica_coords.device).squeeze(), 0
        for dt in range(1, min(N_frame, self.max_acf_lag + 1)):
            c_pred = (centered[:-dt] * centered[dt:]).mean(dim=0)    # [N_sample, N_tica]
            c_pred_norm = c_pred / var_z_safe

            n_lags = dt * frame_time_gap_ratio
            c_target = tica_eigenvalues ** n_lags    # [N_tica,]

            sq = ((c_pred_norm - c_target.unsqueeze(0)) ** 2) * valid
            total = total + sq.sum() / n_valid
            count += 1

        return total / max(count, 1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pred_coordinate,               # [N_frame, N_sample, N_atom, 3]
        noise_level,                   # [N_frame, N_sample]  (kept for API compat)
        atom_to_tokatom_idx,           # [N_atom,]
        is_ligand,                     # [N_atom,]
        msm_state_labels,              # [N_frame,] int  (kept for API compat, unused in HMM)
        msm_tica_mean,                 # [N_pairs,]
        msm_tica_components,           # [N_pairs, N_tica]
        msm_tica_eigenvalues,          # [N_tica,]
        msm_cluster_centers,           # [K, N_tica]      (used only for single-Gaussian fallback)
        msm_cluster_log_vars,          # [K, N_tica]      (single-Gaussian fallback)
        msm_transition_matrix,         # [K, K]
        msm_stationary_distribution,   # [K,]
        msm_traj_population,           # [K,]
        msm_pair_indices_i,            # [N_pairs,]  long
        msm_pair_indices_j,            # [N_pairs,]  long
        frame_time_gap_ratio=1.0,
        per_sample_scale=None,
        _diag_traj_name=None,
    ):
        device = pred_coordinate.device
        zero = torch.zeros(1, device=device, requires_grad=True).squeeze()
        zero_metrics = {
            "hmm_loss": zero.detach(),
            "population_loss": zero.detach(),
            "autocorrelation_loss": zero.detach(),
        }

        # ── 1. Extract Cα ─────────────────────────────────────────────────
        ca_mask = (atom_to_tokatom_idx == 1) & (~is_ligand.bool())
        n_ca = ca_mask.sum().item()
        if n_ca == 0:
            return zero, zero_metrics

        pred_ca = pred_coordinate[:, :, ca_mask, :]   # [N_frame, N_sample, N_ca, 3]

        max_idx = max(msm_pair_indices_i.max().item(), msm_pair_indices_j.max().item())
        if max_idx >= n_ca:
            return zero, zero_metrics

        # ── 2. TICA projection ────────────────────────────────────────────
        tica_coords = self.projector(
            pred_ca,
            msm_tica_mean.to(device),
            msm_tica_components.to(device),
            msm_pair_indices_i.to(device),
            msm_pair_indices_j.to(device),
        )   # [N_frame, N_sample, N_tica]

        # Pair-distances for diagnostic (cheap recompute — only used when collapse triggers)
        _pi = msm_pair_indices_i.to(device)
        _pj = msm_pair_indices_j.to(device)
        _pair_dists_dbg = torch.norm(pred_ca[:, :, _pi] - pred_ca[:, :, _pj], dim=-1)

        # ── 3. Emission — single diagonal Gaussian per cluster ──
        # GMM (full-cov / diag) paths intentionally disabled: emission uses only
        # the per-cluster mean + diagonal variance from msm_cluster_centers /
        # msm_cluster_log_vars. Upstream may still pass GMM tensors; they are ignored here.
        centers  = msm_cluster_centers.to(device)
        log_vars = msm_cluster_log_vars.to(device).clamp(min=self.log_var_min)
        log_emit = _gaussian_log_emission(tica_coords, centers, log_vars)

        # ── 4. Log T^k ────────────────────────────────────────────────────
        T = msm_transition_matrix.to(device)
        pi = msm_stationary_distribution.to(device)
        log_pi = torch.log(pi.clamp(min=1e-10))
        log_Tk = _compute_log_T_power(T, frame_time_gap_ratio, pi)

        # ── 5. Three losses ───────────────────────────────────────────────
        hmm_loss = self._hmm_forward_loss(log_pi, log_Tk, log_emit)
        if torch.isnan(hmm_loss) or torch.isinf(hmm_loss):
            hmm_loss = zero

        # Diagnostic decomposition (logged-only, not added to total_loss):
        #   emission-only NLL: -mean_t logsumexp_k(log_emit[t,k] + log_pi[k]) / N_frame
        #     → likelihood of frames under independent-state assumption with prior π
        #   transition contribution: hmm_loss - emission_only_nll
        #     → extra penalty from the joint trajectory coupling through T^k
        with torch.no_grad():
            _log_post = log_emit + log_pi                              # [NF, NS, K]
            _frame_ll = torch.logsumexp(_log_post, dim=-1)             # [NF, NS]
            hmm_emission_loss = (-_frame_ll.mean()).detach()
            if torch.isnan(hmm_emission_loss) or torch.isinf(hmm_emission_loss):
                hmm_emission_loss = zero.detach()
            hmm_transition_loss = (hmm_loss.detach() - hmm_emission_loss)

        pop_loss = self._population_loss(
            log_emit, msm_traj_population.to(device), pi
        )
        if torch.isnan(pop_loss) or torch.isinf(pop_loss):
            pop_loss = zero

        acf_loss = self._autocorrelation_loss(
            tica_coords, msm_tica_eigenvalues.to(device), frame_time_gap_ratio,
            _diag_traj_name=_diag_traj_name,
            _diag_pair_dists=_pair_dists_dbg,
        )
        if torch.isnan(acf_loss) or torch.isinf(acf_loss):
            acf_loss = zero

        total_loss = (
            self.weight_transition * hmm_loss
            + self.weight_population * pop_loss
            + self.weight_autocorrelation * acf_loss
        )

        metrics = {
            "hmm_loss": hmm_loss.detach(),
            "hmm_emission_loss": hmm_emission_loss,
            "hmm_transition_loss": hmm_transition_loss,
            "population_loss": pop_loss.detach(),
            "autocorrelation_loss": acf_loss.detach(),
        }

        # ── Diagnostic dump ───────────────────────────────────────────────
        _dump_dir = _tica_diag_dump_dir()
        if _dump_dir:
            try:
                import numpy as _np
                import uuid as _uuid
                os.makedirs(_dump_dir, exist_ok=True)

                # Recompute q_avg (same formula as _population_loss)
                _log_pi = torch.log(pi.clamp(min=1e-8))
                _log_post = log_emit + _log_pi
                _log_q = _log_post - torch.logsumexp(_log_post, dim=-1, keepdim=True)
                _q_t = _log_q.exp()                          # [NF, NS, K]
                _q_avg = _q_t.mean(dim=(0, 1)).clamp(min=1e-8)  # [K]

                traj_pop = msm_traj_population.to(device)    # [K]
                tag = str(_diag_traj_name) if _diag_traj_name else "traj"
                tag = tag.replace("/", "_").replace(" ", "_")
                uid = _uuid.uuid4().hex[:8]
                _path = os.path.join(_dump_dir, f"{tag}__{uid}.npz")

                _np.savez_compressed(
                    _path,
                    traj_name=_np.asarray(str(_diag_traj_name)),
                    # Targets / priors (ground truth side)
                    traj_population=traj_pop.detach().float().cpu().numpy(),
                    stationary_distribution=pi.detach().float().cpu().numpy(),
                    transition_matrix=T.detach().float().cpu().numpy(),
                    tica_eigenvalues=msm_tica_eigenvalues.detach().float().cpu().numpy(),
                    msm_state_labels=msm_state_labels.detach().cpu().numpy(),
                    # Predicted state distribution
                    q_avg=_q_avg.detach().float().cpu().numpy(),
                    q_per_frame_mean_over_samples=_q_t.mean(dim=1).detach().float().cpu().numpy(),  # [NF, K]
                    # Emission & TICA coords
                    log_emit=log_emit.detach().float().cpu().numpy(),     # [NF, NS, K]
                    tica_coords=tica_coords.detach().float().cpu().numpy(),  # [NF, NS, NT]
                    # Loss values
                    hmm_loss=hmm_loss.detach().float().cpu().numpy(),
                    population_loss=pop_loss.detach().float().cpu().numpy(),
                    autocorrelation_loss=acf_loss.detach().float().cpu().numpy(),
                    frame_time_gap_ratio=_np.float32(frame_time_gap_ratio),
                )
            except Exception as _e:
                print(f"[TICA-DIAG-DUMP] failed: {_e}", flush=True)

        return total_loss, metrics


# ── Per-frame TICA MSE Loss ───────────────────────────────────────────────────
