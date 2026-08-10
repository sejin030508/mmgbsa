#!/bin/bash
# End-to-end bioemu domainmotion benchmark for MSM-trained BioKinema ckpts.
#
# Per ckpt:
#   1) inference (batch_structure_trajectories.py protenix mode) — for each of 20
#      Protenix init CIFs per system × 22 domainmotion systems, generate 50
#      x_at_1us samples via one W_H=1 W_G=1 coarse_frame_num=2 coarse_interval=1000ns AR step.
#   2) merge (scripts/merge_bioemu_xtc_msm.py) — concatenate 20 init structures +
#      1000 generated frames into <UID>.xtc + <UID>.pdb per system, naming chosen
#      so bioemu_benchmarks.samples.find_samples_in_dir picks them up.
#   3) benchmark (bioemu-benchmarks/get_metric_msm.py) — runs the domainmotion
#      evaluator under the bioemu conda env, saves plots and result tables.
#
# Customise:
#   CKPTS         – array of "<run_subdir>:<step>:<run_tag>" triples
#   PROTENIX_RES  – Protenix init dir (must contain domainmotion/<UID>/<UID>/seed_101/predictions/*.cif)
#   GPUS, CHUNK   – tune for the available cards (large proteins OOM with chunk≥50 on 40 GiB cards)
#   METRIC        – currently fixed to domainmotion; the script also supports localunfolding / crypticpocket
#
# Outputs:
#   <OUT_BASE>/biokinema_<run_tag>/domainmotion/<UID>/<UID>_seed_101_sample_*/predictions/*.cif
#   <OUT_BASE>/biokinema_<run_tag>/domainmotion/<UID>.xtc + <UID>.pdb (after merge — flat, matching names)
#   /cto_studio/xtalpi_lab/fengbin/datas/bioemu_predicted_plot/<run_tag>/domainmotion/*.png
#   /cto_studio/xtalpi_lab/fengbin/datas/bioemu_predicted_metric/<run_tag>/domainmotion/*
set -e

BIOKINEMA_ROOT="/cto_studio/xtalpi_lab/fengbin/Protenix_v0.2.0/BioKinema"
PROTENIX_RES="/cto_studio/xtalpi_lab/fengbin/Protenix_v1.0.0/protenix_results"
OUT_BASE="${BIOKINEMA_ROOT}/output_bioemu_bench"
PLOT_ROOT="/cto_studio/xtalpi_lab/fengbin/datas/bioemu_predicted_plot"
SAVE_ROOT="/cto_studio/xtalpi_lab/fengbin/datas/bioemu_predicted_metric"

PROTENIX_PY="/cto_studio/xtalpi_lab/fengbin/anaconda3/envs/protenix/bin/python"
BIOEMU_PY="/cto_studio/xtalpi_lab/fengbin/anaconda3/envs/bioemu/bin/python"

# === Tunables ===
CKPTS=(
  "BioKinema_MSM_paired_training_20260521_044101:4999:msm_paired_4999ema"
  "BioKinema_MSM_anchored_training_20260521_182624:3999:msm_anchored_3999ema"
)
CKPT_ROOT="/cto_studio/xtalpi_lab/fengbin/Protenix_v0.2.0/Protenix/output_msm"
METRIC="domainmotion"
GPUS="${GPUS:-1,2,3,4}"          # avoid GPU0/5/6/7 if they are contested
CHUNK="${CHUNK:-20}"             # N_sample chunk size; lower if OOM on large proteins
N_SAMPLE="${N_SAMPLE:-50}"
COARSE_FRAME_NUM="${COARSE_FRAME_NUM:-2}"
COARSE_INTERVAL="${COARSE_INTERVAL:-1000.0}"

# === Env (matches inference.sh defaults) ===
export PATH="/cto_studio/xtalpi_lab/fengbin/anaconda3/envs/protenix/bin:${PATH}"
export CUTLASS_PATH=/cto_studio/xtalpi_lab/fengbin/cutlass
export CUDA_HOME=/cto_studio/xtalpi_lab/softwares/cuda-11.8
export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export PYTHONPATH="${BIOKINEMA_ROOT}:${PYTHONPATH}"
export BIOKINEMA_MSA_CACHE_DIR="${BIOKINEMA_ROOT}/msa"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${BIOKINEMA_ROOT}"

for triple in "${CKPTS[@]}"; do
    IFS=':' read -r RUN STEP TAG <<< "${triple}"
    CHECKPOINT="${CKPT_ROOT}/${RUN}/checkpoints/${STEP}_ema_0.999.pt"
    INFER_OUT="${OUT_BASE}/biokinema_${TAG}"
    XTC_OUT="${OUT_BASE}/biokinema_${TAG}_xtc"

    [ -f "${CHECKPOINT}" ] || { echo "[FATAL] missing ${CHECKPOINT}"; exit 1; }

    echo "============================================================"
    echo " ${TAG}   $(date '+%H:%M:%S')"
    echo "   ckpt: ${CHECKPOINT}"
    echo "============================================================"

    echo "[1/3] inference -> ${INFER_OUT}"
    ${PROTENIX_PY} scripts/batch_structure_trajectories.py protenix \
        --protenix-results-root "${PROTENIX_RES}" \
        --output-root "${INFER_OUT}" \
        --checkpoint-path "${CHECKPOINT}" \
        --categories "${METRIC}" \
        --coarse-frame-num "${COARSE_FRAME_NUM}" \
        --coarse-interval "${COARSE_INTERVAL}" \
        --fine-frame-num 1 \
        --W_H 1 --W_G 1 \
        --N_sample "${N_SAMPLE}" --N_step 20 --N_cycle 10 \
        --sample-diffusion-chunk-size "${CHUNK}" \
        --gpus "${GPUS}"

    echo "[2/3] merge -> ${XTC_OUT}/${METRIC}/<UID>.xtc + .pdb"
    ${BIOEMU_PY} scripts/merge_bioemu_xtc_msm.py \
        --inference-root "${INFER_OUT}" \
        --output-root "${XTC_OUT}" \
        --categories "${METRIC}"

    echo "[3/3] benchmark -> ${PLOT_ROOT}/${TAG}/${METRIC}"
    ${BIOEMU_PY} /cto_studio/xtalpi_lab/fengbin/Protenix_v0.2.0/bioemu-benchmarks/get_metric_msm.py \
        "${XTC_OUT}" \
        "${PLOT_ROOT}/${TAG}" \
        "${SAVE_ROOT}/${TAG}" \
        "${METRIC}"

    echo "${TAG} done $(date '+%H:%M:%S')"
done

echo "ALL DONE $(date '+%H:%M:%S')"
