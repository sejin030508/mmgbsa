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

import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_conformation,
    sample_diffusion_train_eval,
    sample_diffusion_training,
)
from protenix.model.utils import simple_merge_dict_list
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator
from protenix.metrics.rmsd import weighted_rigid_align
from .modules.confidence import ConfidenceHead
from .modules.diffusion import DiffusionModule
from .modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from .modules.head import DistogramHead
from .modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from .modules.primitives import LinearNoBias

logger = get_logger(__name__)


class Protenix(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs) -> None:

        super(Protenix, self).__init__()
        self.configs = configs

        # Some constants
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
            
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        # self.distogram_head = DistogramHead(**configs.model.distogram_head)
        # self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        # device = input_feature_dict["restype"].device
        # N_token = input_feature_dict["restype"].shape[0]

        # print("N_token", N_token)

        # s_inputs = torch.zeros([N_token, self.configs.c_token + 32 + 32 + 1]).to(device)
        # s = torch.zeros([N_token, self.c_s]).to(device)
        # z = torch.zeros([N_token, N_token, self.c_z]).to(device)

        # return s_inputs, s, z

        # Use precomputed embeddings if present, UNLESS train_input_embedder is on (in
        # which case we always recompute so gradients can flow into input_embedder /
        # msa_module / pairformer_stack).
        _use_precomputed = (
            (not self.configs.dump_embeddings)
            and "s_inputs" in input_feature_dict
            and not self.configs.model.get("train_input_embedder", False)
        )
        if _use_precomputed:
            s_inputs, s, z = input_feature_dict["s_inputs"], input_feature_dict["s"], input_feature_dict["z"]
            return s_inputs, s, z


        # assert not self.configs.dump_embeddings
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            # Deepspeed_evo_attention do not support token <= 16
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        s_init = self.linear_no_bias_sinit(s_inputs)  #  [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict)
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(input_feature_dict)
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                        use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                        and deepspeed_evo_attention_condition_satisfy,
                        use_lma=self.configs.use_lma,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                        use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                        and deepspeed_evo_attention_condition_satisfy,
                        use_lma=self.configs.use_lma,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=self.configs.use_deepspeed_evo_attention
                    and deepspeed_evo_attention_condition_satisfy,
                    use_lma=self.configs.use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()

        return s_inputs, s, z

    def sample_diffusion_train_eval(self, **kwargs) -> torch.Tensor:
        """
        Training/eval-time per-frame windowed diffusion sampler. Used only when a
        ``label_dict`` is available (train / val / test). Real inference uses
        :meth:`sample_diffusion` (hierarchical).
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion_train_eval
        )(**_configs, **kwargs)

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Inference-time hierarchical diffusion sampler (coarse forecasting + fine
        interpolation). Used when no ``label_dict`` is available.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args, **kwargs):
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        pred_dicts = []
        log_dicts = []
        time_trackers = []
        for _ in range(N_model_seed):
            pred_dict, log_dict, time_tracker = self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
            )
            pred_dicts.append(pred_dict)
            log_dicts.append(log_dict)
            time_trackers.append(time_tracker)

        # Combine outputs of multiple models
        def _cat(dict_list, key):
            return torch.cat([x[key] for x in dict_list], dim=0)

        def _list_join(dict_list, key):
            return sum([x[key] for x in dict_list], [])

        all_pred_dict = {
            "coordinate": _cat(pred_dicts, "coordinate"),
            "coordinate_gt": _cat(pred_dicts, "coordinate_gt") if "coordinate_gt" in pred_dicts[0] else None
            # "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
            # "full_data": _list_join(pred_dicts, "full_data"),
            # "plddt": _cat(pred_dicts, "plddt"),
            # "pae": _cat(pred_dicts, "pae"),
            # "pde": _cat(pred_dicts, "pde"),
            # "resolved": _cat(pred_dicts, "resolved"),
        }

        all_log_dict = simple_merge_dict_list(log_dicts)
        all_time_dict = simple_merge_dict_list(time_trackers)
        return all_pred_dict, all_log_dict, all_time_dict

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        with torch.no_grad():
            s_inputs, s, z = self.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=N_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        if mode == "inference":
            keys_to_delete = []
            for key in input_feature_dict.keys():
                if "template_" in key or key in [
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "profile",
                    "deletion_mean",
                    "token_bonds",
                ]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del input_feature_dict[key]
            torch.cuda.empty_cache()
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        # Eval path (label_dict given) compares pred to GT via the assert N_sample==1
        # invariant below. Override to 1 to keep eval semantics independent of the
        # training-time --sample_diffusion.N_sample setting.
        N_sample = 1 if label_dict is not None else self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]
        no_history = not self.configs.model.diffusion_module["causal_mask"] # ignore history when causal_mask is False

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )

        if label_dict is not None:
            N_frame = label_dict["traj_len"]
            input_feature_dict["frame0_coordinate"] = label_dict["coordinate_0"]
            input_feature_dict["frame0_coordinate_mask"] = label_dict["coordinate_mask"]
        else:
            N_frame = self.configs.infer_setting.sample_N_frame

        if label_dict is not None:
            # Training / validation / test eval: per-frame windowed sampler.
            pred_coordinate = self.sample_diffusion_train_eval(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                N_sample=N_sample,
                N_frame=N_frame,
                no_history=no_history,
                window_size=self.configs.sample_diffusion["window_size"],
                noise_schedule=noise_schedule,
                inplace_safe=inplace_safe,
            )
        else:
            # Real inference: hierarchical coarse forecasting + fine interpolation,
            # or single-shot conformation sampling.
            coarse_frame_num = self.configs.coarse_frame_num
            coarse_interval = self.configs.coarse_interval
            fine_frame_num = self.configs.fine_frame_num
            W_H = self.configs.W_H
            W_G = self.configs.W_G
            steps_per_stage = getattr(self.configs, "steps_per_stage", 0)
            history_noise = getattr(self.configs, "history_noise", 0.0)
            history_t = getattr(self.configs, "history_t", 0.0)
            conformation_sampling = getattr(self.configs, "conformation_sampling", False)
            if conformation_sampling:
                n_conf = max(1, coarse_frame_num - 1)
                pred_coordinate = sample_diffusion_conformation(
                    denoise_net=self.diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s,
                    z_trunk=z,
                    N_sample=n_conf,
                    noise_schedule=noise_schedule,
                    gamma0=self.configs.sample_diffusion["gamma0"],
                    gamma_min=self.configs.sample_diffusion["gamma_min"],
                    noise_scale_lambda=self.configs.sample_diffusion["noise_scale_lambda"],
                    step_scale_eta=self.configs.sample_diffusion["step_scale_eta"],
                    diffusion_chunk_size=getattr(self.configs, "conformation_chunk_size", 25),
                    inplace_safe=inplace_safe,
                    attn_chunk_size=chunk_size,
                )
            else:
                pred_coordinate = self.sample_diffusion(
                    denoise_net=self.diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s,
                    z_trunk=z,
                    N_sample=N_sample,
                    W_H=W_H,
                    W_G=W_G,
                    coarse_frame_num=coarse_frame_num,
                    coarse_interval=coarse_interval,
                    fine_frame_num=fine_frame_num,
                    noise_schedule=noise_schedule,
                    inplace_safe=inplace_safe,
                    steps_per_stage=steps_per_stage,
                    history_noise=history_noise,
                    history_t=history_t,
                )

        def align_to_first(x_original, ref_frame, coordinate_mask):
            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    # Align all frames to the reference frame
                    x_aligned = torch.zeros_like(x_original).float()
                    for i in range(0, x_original.shape[0]):
                        # print(x_original[i].dtype, ref_frame.dtype, label_dict["coordinate_mask"].float().dtype)
                        x_aligned[i] = weighted_rigid_align(
                            x=x_original[i].float(),
                            x_target=ref_frame,
                            atom_weight=coordinate_mask.float(),
                            stop_gradient=True
                        )
                    return x_aligned

        # align predictions and gt coordinate to the first frame 

        if label_dict is not None:
            pred_dict["coordinate"] = align_to_first(
                x_original=pred_coordinate,
                ref_frame=label_dict["coordinate_0"].unsqueeze(-3),
                coordinate_mask=input_feature_dict["frame0_coordinate_mask"]
            )

            traj_len = label_dict["traj_len"]
            traj = []
            for i in range(traj_len):
                traj.append(label_dict[f"coordinate_{i}"])
            coordinate_gt = torch.stack(traj, dim=0).unsqueeze(-3)
            assert pred_coordinate.shape[-3] == 1 # N_sample = 1
            pred_dict["coordinate_gt"] = align_to_first(
                x_original=coordinate_gt,
                ref_frame=label_dict["coordinate_0"].unsqueeze(-3),
                coordinate_mask=input_feature_dict["frame0_coordinate_mask"]
            )

            pred_dict["noise_level"] = None
        else:
            pred_dict["coordinate"] = align_to_first(
                x_original=pred_coordinate,
                ref_frame=pred_coordinate[0],
                coordinate_mask=input_feature_dict["frame0_coordinate_mask"]
            )

        
        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        if mode == "inference" or N_token > 500:
            torch.cuda.empty_cache()

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict,
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """
        N_token = input_feature_dict["residue_index"].shape[-1]
        if N_token <= 16:
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        # Default: freeze input_embedder + pairformer (use precomputed embeddings, no grad).
        # When --model.train_input_embedder true: enable grad so these modules fine-tune.
        # The recycling loop inside get_pairformer_output already restricts pairformer-stack
        # gradients to the last cycle (set_grad_enabled on cycle_no == N_cycle-1); input_embedder
        # runs once outside that loop and gets gradients whenever the outer context allows it.
        _train_embedder = self.configs.model.get("train_input_embedder", False)
        if _train_embedder and self.training:
            s_inputs, s, z = self.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=N_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            with torch.no_grad():
                s_inputs, s, z = self.get_pairformer_output(
                    input_feature_dict=input_feature_dict,
                    N_cycle=N_cycle,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        log_dict = {}
        pred_dict = {}

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]
        N_sample = self.diffusion_batch_size
        x_target_for_loss, x_denoised, x_noise_level, gamma_bar, x_gt_augment = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=N_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
        )
        pred_dict.update(
            {
                # "distogram": self.distogram_head(z),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
                "coordinate_gt": x_target_for_loss,
                "coordinate_gt_original": x_gt_augment,
                "gamma_bar": gamma_bar
            }
        )

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "inference", "eval"]
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            # nc_rng = np.random.RandomState(current_step)
            # N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            # pred_dict, label_dict, log_dict = self.main_train_loop(
            #     input_feature_dict=input_feature_dict,
            #     label_full_dict=label_full_dict,
            #     label_dict=label_dict,
            #     N_cycle=N_cycle,
            #     symmetric_permutation=symmetric_permutation,
            #     inplace_safe=inplace_safe,
            #     chunk_size=chunk_size,
            # )
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=symmetric_permutation,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
