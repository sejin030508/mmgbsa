#!/bin/bash
# =============================================================================
# Reproduce the BioKinema Atlas conformational-ensemble benchmark.
#
# Target configuration (the result this package reproduces):
#   checkpoint = sqrt-pretrain 5999 EMA   |  coarse interval = 1 ns
#   100 ns trajectory (101 frames @ 1 ns) per init structure, MD frame-0 init, MSA on.
#   Expected metrics are in expected_metrics.txt (RMWD 2.51, PC-sim 44.4 %, ...).
#
# Two stages:
#   (1) inference  : roll out one 100 ns trajectory from each of the 243 init structures
#                    (81 Atlas targets x 3 MD replicas), sharded across GPUs.
#   (2) analysis   : compare the generated ensembles to the reference MD with the
#                    10-metric AlphaFlow evaluation suite -> metrics.txt + out.pkl.
#
# REQUIRED inputs (pass as args):
#   --checkpoint_path <PATH>   BioKinema model checkpoint (.pt)
#   --md_dir          <PATH>   Atlas MD dataset root. One sub-dir per target, each containing
#                              <name>.pdb and <name>_prod_R{1,2,3}_fit.xtc  (see README.md).
#
# OPTIONAL:
#   --output_dir      <PATH>   where predictions + metrics are written  (default: ./reproduce_output)
#   --init_frames_dir <PATH>   init structures                          (default: <pkg>/init_frames)
#   --gpus            "0 1 .."  GPU ids to shard across                  (default: "0")
#   --msa_cache_dir   <PATH>   MSA cache (hashed by sequence)           (default: <repo>/msa)
#   --num_workers     <N>      CPU workers for the analysis stage       (default: 64)
#   --stage           all|inference|analysis                           (default: all)
#
# Usage:
#   bash run_reproduce.sh \
#       --checkpoint_path /path/to/5999_ema_0.999.pt \
#       --md_dir          /path/to/atlas_sims \
#       --output_dir      ./reproduce_output \
#       --gpus            "0 1 2 3"
# =============================================================================
set -euo pipefail

# ---- locate this package and the BioKinema repo root ----
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PKG_DIR}/../.." && pwd)"          # .../BioKinema

# ---- defaults ----
CHECKPOINT_PATH=""
MD_DIR=""
OUTPUT_DIR="./reproduce_output"
INIT_FRAMES_DIR="${PKG_DIR}/init_frames"
GPUS="0"
MSA_CACHE_DIR="${REPO_ROOT}/msa"
NUM_WORKERS=64
STAGE="all"
# Python interpreter for the BioKinema (protenix) conda env. Override with $BIOKINEMA_PY if needed.
PY="${BIOKINEMA_PY:-/cto_studio/xtalpi_lab/fengbin/anaconda3/envs/protenix/bin/python}"

# ---- parse args ----
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2;;
    --md_dir)          MD_DIR="$2"; shift 2;;
    --output_dir)      OUTPUT_DIR="$2"; shift 2;;
    --init_frames_dir) INIT_FRAMES_DIR="$2"; shift 2;;
    --gpus)            GPUS="$2"; shift 2;;
    --msa_cache_dir)   MSA_CACHE_DIR="$2"; shift 2;;
    --num_workers)     NUM_WORKERS="$2"; shift 2;;
    --stage)           STAGE="$2"; shift 2;;
    -h|--help)         sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

[ -z "${CHECKPOINT_PATH}" ] && { echo "ERROR: --checkpoint_path is required"; exit 1; }
[ -z "${MD_DIR}" ]          && { echo "ERROR: --md_dir is required"; exit 1; }
[ -f "${CHECKPOINT_PATH}" ] || { echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}"; exit 1; }
[ -d "${MD_DIR}" ]          || { echo "ERROR: md_dir not found: ${MD_DIR}"; exit 1; }
[ -d "${INIT_FRAMES_DIR}" ] || { echo "ERROR: init_frames_dir not found: ${INIT_FRAMES_DIR}"; exit 1; }

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"; mkdir -p "${LOG_DIR}"

# ---- runtime env for the BioKinema model kernels (edit for your machine) ----
export PATH="$(dirname "${PY}"):${PATH}"
export CUTLASS_PATH="${CUTLASS_PATH:-/cto_studio/xtalpi_lab/fengbin/cutlass}"
export CUDA_HOME="${CUDA_HOME:-/cto_studio/xtalpi_lab/softwares/cuda-11.8}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"
export USE_DEEPSPEED_EVO_ATTTENTION="${USE_DEEPSPEED_EVO_ATTTENTION:-true}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export BIOKINEMA_MSA_CACHE_DIR="${MSA_CACHE_DIR}"
export BIOKINEMA_QUIET_CCD_MSG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- fixed generation hyper-parameters for the reproduce target (do not change) ----
COARSE_FRAME_NUM=101   # 100 ns trajectory ...
COARSE_INTERVAL=1      # ... at 1 ns / frame (this is the "1 ns interval" setting)
FINE_FRAME_NUM=1       # no fine interpolation
W_H=1                  # 1 history frame of autoregressive context
W_G=100                # generate the whole trajectory jointly (OOM fallback: 50/25/10)
N_SAMPLE=1
N_STEP=20
N_CYCLE=10
LAMBDA=1.75            # noise_scale_lambda
ETA=1.5               # step_scale_eta
BETA=0.5   # sqrt time scaling (matches the sqrt-pretrain checkpoint)
HISTORY_T=4e-4
HISTORY_NOISE=0.0
SEED=101

count_done() { find "$1" -maxdepth 1 -type d -name '*_R*_0' 2>/dev/null \
               | while read d; do ls "$d/predictions/"*.cif >/dev/null 2>&1 && echo x; done | wc -l; }

# ======================================================================
# Stage 1: inference (trajectory rollout)
# ======================================================================
run_inference() {
  local gpus=(${GPUS}); local ngpu=${#gpus[@]}
  local jobs=(); for f in "${INIT_FRAMES_DIR}"/*_R[123]_0.cif; do [ -f "$f" ] && jobs+=("$f"); done
  echo "[infer] ${#jobs[@]} init structures -> ${OUTPUT_DIR}  (GPUs: ${GPUS}, ckpt: $(basename ${CHECKPOINT_PATH}))"

  run_one() {  # <init_cif> <gpu> <w_g>
    local cif="$1" gpu="$2" wg="$3"; local stem; stem=$(basename "${cif}" .cif)
    ls "${OUTPUT_DIR}/${stem}/predictions/"*.cif >/dev/null 2>&1 && { echo "[gpu${gpu}] ${stem} skip"; return; }
    local plist="${OUTPUT_DIR}/_list_${stem}.txt"
    printf "%s\t%s\n" "${cif}" "${OUTPUT_DIR}" > "${plist}"
    CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${REPO_ROOT}/runner/inference_multi.py" \
        --pdb_list_file "${plist}" --seeds ${SEED} \
        --load_checkpoint_path "${CHECKPOINT_PATH}" \
        --model.N_cycle ${N_CYCLE} --model.diffusion_module.causal_mask false \
        --model.diffusion_module.beta ${BETA} \
        --history_noise ${HISTORY_NOISE} --history_t ${HISTORY_T} \
        --data.train_sets inference --data.test_sets inference \
        --sample_diffusion.N_sample ${N_SAMPLE} --sample_diffusion.N_step ${N_STEP} \
        --sample_diffusion.noise_scale_lambda ${LAMBDA} --sample_diffusion.step_scale_eta ${ETA} \
        --infer_setting.sample_diffusion_chunk_size 1 \
        --coarse_frame_num ${COARSE_FRAME_NUM} --coarse_interval ${COARSE_INTERVAL} \
        --fine_frame_num ${FINE_FRAME_NUM} --W_H ${W_H} --W_G ${wg} \
        --steps_per_stage 0 \
        --data.num_dl_workers 1 --data.msa.enable true --load_strict false \
        > "${LOG_DIR}/${stem}.log" 2>&1 || echo "[gpu${gpu}] ${stem} FAILED (see ${LOG_DIR}/${stem}.log)"
  }

  # W_G fallback passes: full joint generation first, then progressively smaller batches for
  # any target that hit CUDA OOM (only un-finished targets are retried; finished ones skip).
  for wg in ${W_G} 50 25 10; do
    local pids=()
    for w in $(seq 0 $((ngpu-1))); do
      ( idx=${w}
        while [ ${idx} -lt ${#jobs[@]} ]; do
          run_one "${jobs[$idx]}" "${gpus[$w]}" "${wg}"
          idx=$((idx + ngpu))
        done ) &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait ${p} || true; done
    echo "[infer] pass W_G=${wg} done: $(count_done ${OUTPUT_DIR})/${#jobs[@]} structures complete"
  done
}

# ======================================================================
# Stage 2: analysis (10-metric AlphaFlow suite)
# ======================================================================
run_analysis() {
  echo "[analyze] ${OUTPUT_DIR}  vs  MD in ${MD_DIR}  (num_workers=${NUM_WORKERS})"
  "${PY}" "${PKG_DIR}/analysis/analyze_ensembles.py" \
      --pdbdir "${OUTPUT_DIR}" --atlas_dir "${MD_DIR}" --num_workers "${NUM_WORKERS}"
  echo "==== metrics ===="
  "${PY}" "${PKG_DIR}/analysis/print_analysis.py" "${OUTPUT_DIR}/out.pkl" | tee "${OUTPUT_DIR}/metrics.txt"
  echo "Saved -> ${OUTPUT_DIR}/metrics.txt   (compare against ${PKG_DIR}/expected_metrics.txt)"
}

echo "==== BioKinema Atlas reproduce  ($(date '+%F %T')) ===="
echo "  repo=${REPO_ROOT}  output=${OUTPUT_DIR}  stage=${STAGE}"
case "${STAGE}" in
  all)       run_inference; run_analysis;;
  inference) run_inference;;
  analysis)  run_analysis;;
  *) echo "ERROR: --stage must be all|inference|analysis"; exit 1;;
esac
echo "==== DONE ($(date '+%F %T')) ===="
