#!/bin/bash
# Copyright 2024 ByteDance and/or its affiliates.
# Licensed under the Apache License, Version 2.0.
#
# Regenerate precomputed Pairformer embeddings (precomputed_emb_dir) for a dataset.
# Every training config reads embeddings from <dataset>.precomputed_emb_dir; instead
# of shipping multi-TB embeddings, regenerate them here from the released checkpoint.
#
# The embedder is run via runner/train.py in --dump_embeddings mode, which writes one
# <traj_name>.pt per system directly into the dataset's precomputed_emb_dir (resolved from
# configs/configs_data.py) — the same path training reads from. Shard across GPUs with
# split_id / total_split.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/encode_embeddings.sh <dataset> <split_id> <total_split>
# e.g. to encode the Atlas training set across 8 GPUs:
#   for i in $(seq 0 7); do
#     CUDA_VISIBLE_DEVICES=$i bash scripts/encode_embeddings.sh atlas_train $i 8 &
#   done; wait
set -euo pipefail

export LAYERNORM_TYPE=fast_layernorm
export CUTLASS_PATH=${CUTLASS_PATH:?set CUTLASS_PATH}
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME}
export USE_DEEPSPEED_EVO_ATTTENTION=true
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

checkpoint_path=${BIOKINEMA_INIT_CKPT:?set BIOKINEMA_INIT_CKPT (release checkpoint used for embedding)}

dataset=${1:?usage: encode_embeddings.sh <dataset> <split_id> <total_split>}
split_id=${2:-0}
total_split=${3:-1}

python3 ./runner/train.py \
    --run_name encode_${dataset} \
    --seed 42 \
    --base_dir ./output_encode \
    --dtype bf16 \
    --project protenix \
    --use_wandb false \
    --diffusion_batch_size 48 \
    --eval_interval 400 \
    --log_interval 50 \
    --checkpoint_interval 400 \
    --ema_decay 0.999 \
    --train_crop_size 1000 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --lr 0.001 \
    --sample_diffusion.N_step 20 \
    --load_checkpoint_path ${checkpoint_path} \
    --load_ema_checkpoint_path ${checkpoint_path} \
    --model.N_cycle 10 \
    --data.train_sets ${dataset} \
    --data.test_sets ${dataset} \
    --data.num_dl_workers 1 \
    --dump_embeddings True \
    --total_split ${total_split} \
    --split_id ${split_id}
