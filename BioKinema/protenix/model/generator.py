# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at

#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional

import torch
import math
from protenix.model.utils import centre_random_augmentation
from protenix.metrics.rmsd import weighted_rigid_align
from tqdm import tqdm
import random
from protenix.model.utils import expand_at_dim


class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list

def sample_diffusion_train_eval(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    N_frame: int = 121,
    window_size: int = 50,
    no_first_frame: bool = False,
    no_history: bool = True,
    first_frame_noise: float = 4e-4,
) -> torch.Tensor:
    """Per-frame windowed reverse-diffusion sampler used at training-time eval
    (validation / test), where ``frame0_coordinate`` is the GT first frame.

    Frame 0 is anchored to the (near-clean) initial structure; the remaining
    ``N_frame - 1`` frames are generated jointly within sliding windows of
    ``window_size`` via an EDM predictor-corrector loop. Real inference uses the
    hierarchical :func:`sample_diffusion` instead.

    Returns:
        torch.Tensor: denoised coordinates [N_frame, N_sample, N_atom, 3].
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init: frame 0 = GT initial structure, frames 1.. = sigma_max EDM noise
        x_l = torch.zeros(size=(N_frame, N_atom, 3), device=device, dtype=dtype)
        x_l[0] = input_feature_dict["frame0_coordinate"].unsqueeze(0)
        coordinate_mask = input_feature_dict["frame0_coordinate_mask"]
        x_l = centre_random_augmentation(
            x_input_coords=x_l, N_sample=chunk_n_sample, mask=coordinate_mask, s_trans=0.,
        ).to(dtype)  # [N_frame, N_sample, N_atom, 3]
        x_l[0, :, coordinate_mask == 0, :] = 0.
        x_l[1:] = noise_schedule[0] * torch.randn(
            size=(N_frame - 1, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )
        t_hat_all = torch.ones([N_frame, chunk_n_sample]).to(dtype).to(device) * -1

        for w_idx in range(0, math.ceil(float(N_frame - 1) / window_size)):
            w_start = w_idx * window_size + 1
            w_end = min((w_idx + 1) * window_size + 1, N_frame)
            # conditioned frames advertise the near-clean first-frame noise level
            t_hat_all[:w_start, :] = first_frame_noise

            # if using history, rigid-align already-generated frames to frame 0
            if not no_history:
                with torch.no_grad():
                    for sample_idx in range(chunk_n_sample):
                        x_original = x_l[:, sample_idx, :, :]
                        ref_frame = x_original[0]
                        x_aligned = torch.zeros_like(x_original)
                        x_aligned[0] = ref_frame
                        for i in range(1, w_start):
                            x_aligned[i] = weighted_rigid_align(
                                x=x_original[i], x_target=ref_frame,
                                atom_weight=coordinate_mask, stop_gradient=True,
                            )
                        x_l[:w_start, sample_idx, :, :] = x_aligned[:w_start, :, :]

            for _, (c_tau_last, c_tau) in enumerate(
                zip(noise_schedule[:-1], noise_schedule[1:])
            ):
                # predictor-corrector step (Alg.18): raise noise to t_hat, denoise, advance
                gamma = float(gamma0) if c_tau > gamma_min else 0
                t_hat = c_tau_last * (gamma + 1)
                delta_noise_level = torch.sqrt(t_hat ** 2 - c_tau_last ** 2)
                x_noisy = x_l[:w_end, :].clone()
                x_noisy[w_start:w_end, :] += noise_scale_lambda * delta_noise_level * torch.randn(
                    size=(w_end - w_start, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
                )
                t_hat_all[w_start:w_end, :] = t_hat
                t_hat_input = t_hat_all[:w_end, :].clone()

                if no_first_frame:
                    x_noisy_window = x_noisy[1:]
                    x_denoised = denoise_net(
                        x_noisy=x_noisy_window, t_hat_noise_level=t_hat_input[1:],
                        input_feature_dict=input_feature_dict, s_inputs=s_inputs,
                        s_trunk=s_trunk, z_trunk=z_trunk,
                        chunk_size=attn_chunk_size, inplace_safe=inplace_safe,
                    )
                    delta = (x_noisy_window - x_denoised) / t_hat
                    dt = c_tau - t_hat
                    x_l[w_start:w_end, :] = (x_noisy_window + step_scale_eta * dt * delta)[w_start-1:w_end-1, :]
                elif no_history:
                    x_noisy[0] += torch.randn_like(x_noisy[0], dtype=dtype) * first_frame_noise
                    x_noisy_window = x_noisy[w_start-1:w_end, :]
                    x_denoised = denoise_net(
                        x_noisy=x_noisy_window, t_hat_noise_level=t_hat_input[w_start-1:w_end, :],
                        input_feature_dict=input_feature_dict, s_inputs=s_inputs,
                        s_trunk=s_trunk, z_trunk=z_trunk,
                        chunk_size=attn_chunk_size, inplace_safe=inplace_safe,
                    )
                    delta = (x_noisy_window - x_denoised) / t_hat
                    dt = c_tau - t_hat
                    x_l[w_start:w_end, :] = (x_noisy_window + step_scale_eta * dt * delta)[1:, :]
                else:
                    x_denoised = denoise_net(
                        x_noisy=x_noisy, t_hat_noise_level=t_hat_input,
                        input_feature_dict=input_feature_dict, s_inputs=s_inputs,
                        s_trunk=s_trunk, z_trunk=z_trunk,
                        chunk_size=attn_chunk_size, inplace_safe=inplace_safe,
                    )
                    delta = (x_noisy - x_denoised) / t_hat
                    dt = c_tau - t_hat
                    x_l[w_start:w_end, :] = (x_noisy + step_scale_eta * dt * delta)[w_start:w_end, :]
        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in tqdm(range(no_chunks)):
            chunk_n_sample = (
                diffusion_chunk_size if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            x_l.append(_chunk_sample_diffusion(chunk_n_sample, inplace_safe=inplace_safe))
        x_l = torch.cat(x_l, -3)
    return x_l


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Trajectory-generation diffusion training (AF3 Appendix p.23 style).

    All frames of the trajectory window are rigid-aligned to frame 0, then a single
    random (R, t) augmentation is shared across frames (N_sample copies). Per-EDM
    noise is sampled either i.i.d. per frame or shared across frames (50/50); the
    first (and sometimes last) frame is set to near-zero noise to realize the
    forecasting / interpolation conditioning. Returns the denoised coordinates for
    the diffusion + trajectory losses.

    Returns:
        (x_gt_augment, x_denoised, sigma, gamma_bar, x_gt_augment) — gamma_bar is a
        zero placeholder kept for a stable return signature.
    """
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    traj_len = label_dict["traj_len"]
    traj = [label_dict[f"coordinate_{i}"] for i in range(traj_len)]
    x_original = torch.stack(traj, dim=0)

    # Align every frame to frame 0, then apply one shared (R, t) augmentation.
    with torch.no_grad():
        ref_frame = x_original[0]  # [N_atom, 3]
        x_aligned = torch.zeros_like(x_original)
        x_aligned[0] = ref_frame
        for i in range(1, x_original.shape[0]):
            x_aligned[i] = weighted_rigid_align(
                x=x_original[i], x_target=ref_frame,
                atom_weight=label_dict["coordinate_mask"].float(), stop_gradient=True,
            )
    x_gt_augment = centre_random_augmentation(
        x_input_coords=x_aligned, N_sample=N_sample, mask=None, s_trans=0.,
    ).to(dtype)  # [N_frame, N_sample, N_atom, 3]
    N_frame = x_gt_augment.shape[0]

    # EDM noise levels: 50% i.i.d. per frame, 50% shared across frames.
    if random.random() <= 1 / 2:
        sigma = noise_sampler(size=(N_frame, N_sample), device=device).to(dtype)
    else:
        sigma = noise_sampler(size=(N_sample,), device=device).to(dtype)
        sigma = expand_at_dim(sigma, dim=0, n=N_frame).clone()

    # Near-zero noise on the first (and sometimes last) frame => forecasting / interpolation.
    for idx in range(N_sample):
        if random.random() <= 1 / 2:
            sigma[0, idx] = 4e-4  # s_min for inference_noise_scheduler
            if random.random() <= 1 / 2:
                sigma[-1, idx] = 4e-4

    # i.i.d. EDM forward process: x_noisy = x_gt + sigma * eps
    epsilon_iid = torch.randn_like(x_gt_augment, dtype=dtype)
    x_noisy = x_gt_augment + epsilon_iid * sigma[..., None, None]
    gamma_bar = torch.zeros_like(sigma)  # placeholder (no temporal-coupling forward process)

    # Denoise [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_noisy, t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in range(no_chunks):
            sl = slice(i * diffusion_chunk_size, (i + 1) * diffusion_chunk_size)
            x_denoised.append(denoise_net(
                x_noisy=x_noisy[..., sl, :, :], t_hat_noise_level=sigma[..., sl],
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
            ))
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma, gamma_bar, x_gt_augment


# ============================================================================
# Inference-time samplers (hierarchical coarse/fine generation + conformation
# sampling), ported from the BioKinema inference tree. The training/eval
# sampler above is sample_diffusion_train_eval; these two drive real inference.
# ============================================================================

def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    first_frame_noise: float = 4e-4,
    # --- Hierarchical generation parameters ---
    W_H: int = 1,
    W_G: int = 10,
    coarse_frame_num: int = 20,
    coarse_interval: float = 5,
    fine_frame_num: int = 5,
    # --- Staircase diffusion ---
    steps_per_stage: int = 0,
    # --- AR stabilisation: noise injected into history/conditioned frames ---
    history_noise: float = 0.0,
    # --- advertised noise level (t_hat) for history frames, without corrupting coords ---
    history_t: float = 0.0,
) -> torch.Tensor:
    """
    Hierarchical trajectory generation via two-stage diffusion sampling.

    Stage 1 (Coarse Forecasting):
        Autoregressively generates coarse_frame_num frames (including x_0)
        with temporal spacing = coarse_interval.  Uses a sliding history
        window of size W_H and generates W_G frames per batch.

    Stage 2 (Fine Interpolation):
        For every pair of consecutive coarse frames, generates
        (fine_frame_num - 1) intermediate frames conditioned on both
        boundary frames.  Temporal spacing = coarse_interval / fine_frame_num.

    Total output frames = 1 + (coarse_frame_num - 1) * fine_frame_num.

    Args:
        denoise_net (Callable): Denoising network.
        input_feature_dict (dict): Input features.
            'time_interval' will be overridden per stage.
        s_inputs  (torch.Tensor): [..., N_tokens, c_s_inputs]
        s_trunk   (torch.Tensor): [..., N_tokens, c_s]
        z_trunk   (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): Decreasing noise levels [N_steps].
        N_sample (int): Number of trajectory samples to draw.
        gamma0, gamma_min, noise_scale_lambda, step_scale_eta:
            Diffusion hyper-parameters (Alg.18 in AF3).
        diffusion_chunk_size (Optional[int]): Chunk over N_sample for memory.
        inplace_safe (bool): Whether inplace ops are safe.
        attn_chunk_size (Optional[int]): Attention chunk size.
        first_frame_noise (float): Small noise level for conditioned frames.
        W_H (int): History window size  (coarse forecasting).
        W_G (int): Generation window size (coarse forecasting).
        coarse_frame_num (int): Number of coarse frames including x_0.
        coarse_interval (float): Temporal spacing between coarse frames.
        fine_frame_num (int): Number of sub-intervals per coarse interval.
            fine_interval = coarse_interval / fine_frame_num.

    Returns:
        torch.Tensor: Full trajectory
            [N_total_frames, N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    device = s_inputs.device
    dtype = s_inputs.dtype
    fine_interval = coarse_interval / fine_frame_num
    coordinate_mask = input_feature_dict["frame0_coordinate_mask"]

    # ================================================================
    #  Per-chunk worker  (handles one slice of N_sample)
    # ================================================================
    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):

        # History/conditioned frames have two decoupled knobs:
        #   cond_noise : std of Gaussian noise actually added to the history coords
        #                (history_noise>0 corrupts them; else tiny first_frame_noise).
        #   cond_t     : the noise level advertised to the net via t_hat (feeds the
        #                FourierEmbedding/AdaLN conditioning). history_t>0 lets us tell
        #                the model the history sits at sigma=history_t WITHOUT actually
        #                corrupting the coordinates -> modulates history influence only.
        #                Defaults to cond_noise (consistent EDM behaviour) when unset.
        cond_noise = history_noise if history_noise > 0 else first_frame_noise
        cond_t = history_t if history_t > 0 else cond_noise

        # ------------------------------------------------------------
        #  Core denoising loop (shared by both stages)
        # ------------------------------------------------------------
        def _denoise(x_all, cond_mask, feat_dict):
            """
            Run the full noise_schedule denoising on x_all.

            Args:
                x_all     : [F, S, A, 3]  frames × samples × atoms × 3
                cond_mask : [F] bool tensor  (True → conditioned / frozen)
                feat_dict : shallow copy of input_feature_dict with the
                            correct 'time_interval' already set.
            Returns:
                x_all with generated (non-conditioned) frames denoised.
            """
            gen_idx = torch.where(~cond_mask)[0]
            cond_idx = torch.where(cond_mask)[0]
            t_hat_all = torch.full(
                (x_all.shape[0], chunk_n_sample),
                first_frame_noise, dtype=dtype, device=device,
            )

            for c_tau_last, c_tau in zip(noise_schedule[:-1], noise_schedule[1:]):
                gamma = float(gamma0) if c_tau > gamma_min else 0.0
                t_hat = c_tau_last * (gamma + 1.0)
                d_sigma = torch.sqrt(t_hat ** 2 - c_tau_last ** 2)

                x_noisy = x_all.clone()

                # Stochastic noise on generated frames
                if gen_idx.numel() > 0:
                    x_noisy[gen_idx] += (
                        noise_scale_lambda * d_sigma
                        * torch.randn(
                            gen_idx.numel(), chunk_n_sample, N_atom, 3,
                            device=device, dtype=dtype,
                        )
                    )
                # Noise on conditioned/history frames (history_noise stabilises AR)
                if cond_idx.numel() > 0:
                    x_noisy[cond_idx] += (
                        cond_noise
                        * torch.randn_like(x_noisy[cond_idx])
                    )

                # Per-frame noise level (history advertised at cond_t)
                t_hat_all[~cond_mask] = t_hat
                t_hat_all[cond_mask] = cond_t

                x_denoised = denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat_all.clone(),
                    input_feature_dict=feat_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                )

                # Denoising update (only generated frames)
                delta = (x_noisy - x_denoised) / t_hat
                dt = c_tau - t_hat
                x_all[gen_idx] = (x_noisy + step_scale_eta * dt * delta)[gen_idx]

            return x_all

        # ------------------------------------------------------------
        #  Staircase denoising loop
        # ------------------------------------------------------------
        def _denoise_staircase(x_all, cond_mask, feat_dict, sps):
            """
            Staircase joint denoising.

            Each generated frame runs the full noise_schedule (n_steps steps),
            but frames are staggered: frame k (0-indexed among gen frames)
            starts at wall-clock step k * sps.  At each wall-clock step, all
            currently active frames share one forward pass through denoise_net,
            each carrying its own per-frame noise level.  Frames that have not
            yet started appear as pure noise; frames that are done are treated
            like conditioned frames (tiny noise level).

            Args:
                x_all    : [F, S, A, 3]  (history frames | gen frames, all initialised)
                cond_mask: [F] bool  (True = history / conditioned)
                feat_dict: feature dict with correct time_interval
                sps      : steps_per_stage  (stagger offset in schedule steps)
            """
            gen_idx  = torch.where(~cond_mask)[0]
            cond_idx = torch.where(cond_mask)[0]
            n_gen    = gen_idx.numel()
            n_steps  = len(noise_schedule) - 1   # number of denoising steps

            total_wall_steps = n_steps + (n_gen - 1) * sps
            max_simultaneous = n_steps // sps if sps > 0 else n_gen
            print(f"  [staircase] n_gen={n_gen} sps={sps} n_steps={n_steps} "
                  f"wall_steps={total_wall_steps} max_active≈{max_simultaneous}")

            # t_hat_all[frame, sample]: noise level seen by denoise_net
            t_hat_all = torch.full(
                (x_all.shape[0], chunk_n_sample),
                first_frame_noise, dtype=dtype, device=device,
            )
            # Not-yet-started gen frames start as pure noise
            for k in range(n_gen):
                t_hat_all[gen_idx[k]] = noise_schedule[0]

            for w in range(total_wall_steps):
                x_noisy   = x_all.clone()
                has_active = False

                # --- per generated frame ---
                for k in range(n_gen):
                    fi        = gen_idx[k]
                    sched_step = w - k * sps   # position in own 0..n_steps-1 schedule

                    if sched_step < 0:
                        # not yet started: pure noise, no update
                        # t_hat_all[fi] already = noise_schedule[0]
                        pass

                    elif sched_step >= n_steps:
                        # done: treat like a conditioned frame
                        t_hat_all[fi] = first_frame_noise
                        x_noisy[fi]   = x_all[fi] + (
                            first_frame_noise
                            * torch.randn_like(x_all[fi])
                        )

                    else:
                        # active: advance one step in its own schedule
                        has_active   = True
                        c_tau_last   = noise_schedule[sched_step]
                        gamma        = float(gamma0) if c_tau_last > gamma_min else 0.0
                        t_hat_k      = c_tau_last * (gamma + 1.0)
                        d_sigma_sq   = max(float(t_hat_k ** 2 - c_tau_last ** 2), 0.0)
                        if d_sigma_sq > 0:
                            x_noisy[fi] = x_all[fi] + (
                                noise_scale_lambda * d_sigma_sq ** 0.5
                                * torch.randn(
                                    chunk_n_sample, N_atom, 3,
                                    device=device, dtype=dtype,
                                )
                            )
                        t_hat_all[fi] = t_hat_k

                # --- conditioned (history) frames: history_noise (AR stabilisation) ---
                if cond_idx.numel() > 0:
                    x_noisy[cond_idx] = x_all[cond_idx] + (
                        cond_noise * torch.randn_like(x_all[cond_idx])
                    )
                    t_hat_all[cond_idx] = cond_t

                if not has_active:
                    continue

                # --- single joint forward pass ---
                x_denoised = denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat_all.clone(),
                    input_feature_dict=feat_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                )

                # --- update only active frames ---
                for k in range(n_gen):
                    fi        = gen_idx[k]
                    sched_step = w - k * sps
                    if sched_step < 0 or sched_step >= n_steps:
                        continue

                    c_tau_last = noise_schedule[sched_step]
                    c_tau      = noise_schedule[sched_step + 1]
                    gamma      = float(gamma0) if c_tau_last > gamma_min else 0.0
                    t_hat_k    = c_tau_last * (gamma + 1.0)

                    delta      = (x_noisy[fi] - x_denoised[fi]) / t_hat_k
                    dt         = c_tau - t_hat_k
                    x_all[fi]  = x_noisy[fi] + step_scale_eta * dt * delta

            return x_all

        # Select denoising function
        _do_denoise = (
            (lambda x, m, f: _denoise_staircase(x, m, f, steps_per_stage))
            if steps_per_stage > 0
            else _denoise
        )

        # ------------------------------------------------------------
        #  Prepare x_0  (initial structure, augmented for S samples)
        # ------------------------------------------------------------
        x_init = torch.zeros(1, N_atom, 3, device=device, dtype=dtype)
        x_init[0] = input_feature_dict["frame0_coordinate"]
        x_init = centre_random_augmentation(
            x_input_coords=x_init,
            N_sample=chunk_n_sample,
            mask=coordinate_mask,
            s_trans=0.,
        ).to(dtype)                                       # [1, S, A, 3]
        x_0 = x_init[0].clone()                           # [S, A, 3]
        x_0[:, coordinate_mask == 0, :] = 0.

        # ============================================================
        #  Stage 1 – Coarse-grained Forecasting
        # ============================================================
        print("=== Stage 1: Coarse Forecasting ===")
        coarse_feat_dict = {**input_feature_dict, "time_interval": coarse_interval}

        X_C = [x_0]                                        # list of [S, A, 3]
        idx = 0

        while idx < coarse_frame_num - 1:
            n_gen = min(W_G, coarse_frame_num - 1 - idx)
            print(f"  coarse: generating frames {idx + 1}..{idx + n_gen}")

            # ---- Select last W_H frames as history context ----
            h0 = max(0, len(X_C) - W_H)
            X_h = torch.stack(X_C[h0:], dim=0)            # [Nh, S, A, 3]
            Nh = X_h.shape[0]

            # Align all history frames to the first history frame
            with torch.no_grad():
                ref = X_h[0]                               # [S, A, 3]
                for fi in range(1, Nh):
                    for si in range(chunk_n_sample):
                        X_h[fi, si] = weighted_rigid_align(
                            x=X_h[fi, si],
                            x_target=ref[si],
                            atom_weight=coordinate_mask,
                            stop_gradient=True,
                        )

            # Initialise generation frames with full noise
            x_gen_noise = noise_schedule[0] * torch.randn(
                n_gen, chunk_n_sample, N_atom, 3,
                device=device, dtype=dtype,
            )

            # Concatenate: [history | generation]
            x_all = torch.cat([X_h, x_gen_noise], dim=0)  # [Nh+ng, S, A, 3]
            cond_mask = torch.zeros(Nh + n_gen, dtype=torch.bool, device=device)
            cond_mask[:Nh] = True

            # Run denoising
            x_all = _do_denoise(x_all, cond_mask, coarse_feat_dict)

            # Store newly generated coarse frames
            for b in range(n_gen):
                X_C.append(x_all[Nh + b].clone())
            idx += n_gen

        X_C_tensor = torch.stack(X_C, dim=0)              # [Lc, S, A, 3]
        print(f"  coarse trajectory: {X_C_tensor.shape[0]} frames")

        # ============================================================
        #  Stage 2 – Fine-grained Interpolation
        # ============================================================
        print("=== Stage 2: Fine Interpolation ===")
        fine_feat_dict = {**input_feature_dict, "time_interval": fine_interval}
        N_inner = fine_frame_num - 1  # intermediate frames to generate per interval

        X_F = [X_C_tensor[0]]                             # start with x_0

        for i in range(coarse_frame_num - 1):
            x_start = X_C_tensor[i].clone()                # [S, A, 3]
            x_end = X_C_tensor[i + 1].clone()              # [S, A, 3]

            if N_inner > 0:
                print(f"  fine: interpolating interval {i} -> {i + 1}")
                # Initialise intermediate frames with full noise
                x_gen_noise = noise_schedule[0] * torch.randn(
                    N_inner, chunk_n_sample, N_atom, 3,
                    device=device, dtype=dtype,
                )

                # Arrange: [x_start, gen_1, …, gen_K, x_end]
                x_all = torch.cat([
                    x_start.unsqueeze(0),
                    x_gen_noise,
                    x_end.unsqueeze(0),
                ], dim=0)                                  # [Ni+2, S, A, 3]

                cond_mask = torch.zeros(N_inner + 2, dtype=torch.bool, device=device)
                cond_mask[0] = True
                cond_mask[-1] = True

                # Run denoising
                x_all = _do_denoise(x_all, cond_mask, fine_feat_dict)

                # Collect interpolated inner frames
                for j in range(1, N_inner + 1):
                    X_F.append(x_all[j].clone())

            # Append the coarse end-frame
            X_F.append(x_end)

        X_F_tensor = torch.stack(X_F, dim=0)              # [Ntotal, S, A, 3]
        print(f"  full trajectory: {X_F_tensor.shape[0]} frames")
        return X_F_tensor

    # ================================================================
    #  Optional chunking over N_sample
    # ================================================================
    if diffusion_chunk_size is None:
        return _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)

    parts = []
    n_chunks = math.ceil(N_sample / diffusion_chunk_size)
    for i in tqdm(range(n_chunks)):
        cs = (
            diffusion_chunk_size
            if i < n_chunks - 1
            else N_sample - i * diffusion_chunk_size
        )
        parts.append(
            _chunk_sample_diffusion(cs, inplace_safe=inplace_safe)
        )
    return torch.cat(parts, dim=1)  # concatenate along sample dimension


def sample_diffusion_conformation(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Standard AF3 (Alg.18) conformational sampler — mirrors eba/protenix/model/generator.py.

    Generates N_sample INDEPENDENT conformations (no temporal/time-attention coupling),
    each with per-step centre_random_augmentation + predictor-corrector denoising. Intended
    for the EBA baseline: load a no-time-attention checkpoint with time_attn=false and sample
    conformations. Returns [..., N_sample, N_atom, 3].
    """
    # BioKinema's DiffusionConditioning requires 2 leading dims [N_frame, N_sample].
    # We map the N conformations to the FRAME dim (sample dim = 1); with time_attn off the
    # frames are processed independently -> N independent conformations [N_conf, 1, N_atom, 3].
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    device = s_inputs.device
    dtype = s_inputs.dtype
    feat_dict = {**input_feature_dict, "time_interval": 1.0}

    def _chunk(nc, inplace_safe):
        x_l = noise_schedule[0] * torch.randn(
            size=(nc, 1, N_atom, 3), device=device, dtype=dtype
        )  # [nc, 1, N_atom, 3]
        for c_tau_last, c_tau in zip(noise_schedule[:-1], noise_schedule[1:]):
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )  # [nc, 1, N_atom, 3]
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)
            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )
            t_hat_t = torch.full((nc, 1), float(t_hat), device=device, dtype=dtype)
            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat_t,
                input_feature_dict=feat_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
            )
            delta = (x_noisy - x_denoised) / t_hat_t[..., None, None]
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt * delta
        return x_l  # [nc, 1, N_atom, 3]

    if diffusion_chunk_size is None:
        x_l = _chunk(N_sample, inplace_safe=inplace_safe)
    else:
        chunks = []
        n_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in range(n_chunks):
            cs = diffusion_chunk_size if i < n_chunks - 1 else N_sample - i * diffusion_chunk_size
            chunks.append(_chunk(cs, inplace_safe=inplace_safe))
        x_l = torch.cat(chunks, dim=0)  # concat along conformation (frame) dim
    return x_l  # [N_conf, 1, N_atom, 3]


