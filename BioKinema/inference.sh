#!/bin/bash
# Copyright 2026 International Digital Economy Academy (IDEA).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUTLASS_PATH=/path/to/cutlass
export CUDA_HOME=/path/to/cuda
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true

N_step=20
N_cycle=10
seed=101
use_deepspeed_evo_attention=true
lambda=1.75
eta=1.5
test_set=inference
# beta: ALiBi power exponent matching the checkpoint. MUST match the model:
#   0.5  -> "sqrt" checkpoint (complexes + short-time MD)   [default]
#   0.25 -> "beta" checkpoint (long single-chain MD)
beta=0.5

# ================== default =====================
# N_sample (int): Number of generated samples
N_sample=1
# coarse_frame_num (int): Number of coarse frames including x_0.
coarse_frame_num=50
# coarse_interval (float): Temporal spacing between coarse frames, unit = ns.
coarse_interval=2
# fine_frame_num (int): Number of sub-intervals per coarse interval. fine_frame_num=1 means no interpolation
fine_frame_num=1
# W_H (int): History window size  (coarse forecasting).
W_H=1
# W_G (int): Generation window size (coarse forecasting).
W_G=50

# parse arguments
while [[ $# -gt 0 ]]; do
    [[ $1 == --* ]] && declare "${1#--}=$2" && shift 2 || { echo "Unknown argument: $1"; exit 1; }
done

# ================= required =====================
: ${checkpoint_path:?required} ${dump_dir:?required} ${input_file:?required}

python3 runner/inference.py \
    --seeds ${seed} \
    --load_checkpoint_path ${checkpoint_path} \
    --dump_dir ${dump_dir} \
    --model.N_cycle ${N_cycle} \
    --model.diffusion_module.causal_mask false \
    --model.diffusion_module.beta ${beta} \
    --data.train_sets ${test_set} \
    --data.test_sets ${test_set} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --sample_diffusion.noise_scale_lambda ${lambda} \
    --sample_diffusion.step_scale_eta ${eta} \
    --infer_setting.sample_diffusion_chunk_size 1 \
    --coarse_frame_num ${coarse_frame_num} \
    --coarse_interval ${coarse_interval} \
    --fine_frame_num ${fine_frame_num} \
    --W_H ${W_H} \
    --W_G ${W_G} \
    --input_file ${input_file} \
    --data.num_dl_workers 1 \
    --data.msa.enable true \
    --load_strict false