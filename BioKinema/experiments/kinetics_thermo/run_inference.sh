#!/bin/bash
# ============================================================================
# BioKinema kinetics/thermodynamics benchmark — STEP 1: generate trajectories.
#
# Generates, for every bundled MD initial frame (data/init_pdbs/<sys>/start_NN.pdb),
# N_SAMPLE independent BioKinema trajectories of 1 us at 10 ns/frame, sharded across
# GPUs, with per-system W_G OOM-halving (W_G -> W_G/2 on out-of-memory / missing output).
#
# REQUIRED: --checkpoint <model.pt>   --output_dir <dir>
# The model checkpoint is the only mandatory user input.
#
# Requires the BioKinema repository (for runner/inference_multi.py + the protenix
# package) and the `protenix` conda environment. See README.md.
# ============================================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- defaults ----
CHECKPOINT=""
OUTPUT_DIR=""
INIT_PDB_DIR="${HERE}/data/init_pdbs"
# Repo root that contains runner/inference_multi.py (default: two levels up from here).
BIOKINEMA_ROOT="${BIOKINEMA_ROOT:-$(cd "${HERE}/../.." && pwd)}"
PYTHON="${PYTHON:-/cto_studio/xtalpi_lab/fengbin/anaconda3/envs/protenix/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

BETA=0.25    # beta exponent in the temporal attention bias (released model: 0.25)
N_SAMPLE=5
COARSE_FRAME_NUM=101     # 1 cond + 100 generated -> 1 us
COARSE_INTERVAL=10       # ns/frame
FINE_FRAME_NUM=1
W_H=1
WG_START=100
WG_FLOOR=5
N_STEP=20
N_CYCLE=10
LAMBDA=1.75
ETA=1.5
HISTORY_NOISE=0.0
HISTORY_T=0.0            # conditioning-frame noise time (released model: 0.0)
SEED=101

usage() {
    grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "Options (defaults in brackets):"
    echo "  --checkpoint PATH        (required) model checkpoint .pt"
    echo "  --output_dir DIR         (required) where trajectories are written"
    echo "  --init_pdb_dir DIR       [${INIT_PDB_DIR}]"
    echo "  --beta F     [${BETA}]  beta exponent in the temporal bias (released model: 0.25)"
    echo "  --n_sample N             [${N_SAMPLE}]   trajectories per init frame"
    echo "  --coarse_frame_num N     [${COARSE_FRAME_NUM}]"
    echo "  --coarse_interval N      [${COARSE_INTERVAL}] (ns)"
    echo "  --w_g N                  [${WG_START}]  joint-generation window (halved on OOM)"
    echo "  --history_t F            [${HISTORY_T}]  conditioning-frame noise time (released model: 0.0)"
    echo "  --n_step / --n_cycle / --lambda / --eta / --seed"
    echo "  --gpus LIST              [${GPUS}]  comma-separated GPU ids"
    echo "  --biokinema_root DIR     [${BIOKINEMA_ROOT}]"
    echo "  --python PATH            [${PYTHON}]"
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --init_pdb_dir) INIT_PDB_DIR="$2"; shift 2;;
        --beta) BETA="$2"; shift 2;;
        --n_sample) N_SAMPLE="$2"; shift 2;;
        --coarse_frame_num) COARSE_FRAME_NUM="$2"; shift 2;;
        --coarse_interval) COARSE_INTERVAL="$2"; shift 2;;
        --w_g) WG_START="$2"; shift 2;;
        --history_t) HISTORY_T="$2"; shift 2;;
        --n_step) N_STEP="$2"; shift 2;;
        --n_cycle) N_CYCLE="$2"; shift 2;;
        --lambda) LAMBDA="$2"; shift 2;;
        --eta) ETA="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --biokinema_root) BIOKINEMA_ROOT="$2"; shift 2;;
        --python) PYTHON="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done
[ -z "${CHECKPOINT}" ] && { echo "ERROR: --checkpoint is required"; usage; }
[ -z "${OUTPUT_DIR}" ] && { echo "ERROR: --output_dir is required"; usage; }
[ -f "${CHECKPOINT}" ] || { echo "ERROR: checkpoint not found: ${CHECKPOINT}"; exit 1; }
[ -f "${BIOKINEMA_ROOT}/runner/inference_multi.py" ] || {
    echo "ERROR: runner/inference_multi.py not found under --biokinema_root ${BIOKINEMA_ROOT}"; exit 1; }

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
TRAJ_DIR="${OUTPUT_DIR}/biokinema_trajs"
LOG_DIR="${OUTPUT_DIR}/logs"
LIST_DIR="${OUTPUT_DIR}/_pdb_lists"
mkdir -p "${TRAJ_DIR}" "${LOG_DIR}" "${LIST_DIR}"

# ---- environment (override via env vars to match your system) ----
export PATH="$(dirname "${PYTHON}"):${PATH}"
export CUTLASS_PATH="${CUTLASS_PATH:-/cto_studio/xtalpi_lab/fengbin/cutlass}"
export CUDA_HOME="${CUDA_HOME:-/cto_studio/xtalpi_lab/softwares/cuda-11.8}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"
export USE_DEEPSPEED_EVO_ATTTENTION="${USE_DEEPSPEED_EVO_ATTTENTION:-true}"
export PYTHONPATH="${BIOKINEMA_ROOT}:${PYTHONPATH:-}"
export BIOKINEMA_QUIET_CCD_MSG=1
export BIOKINEMA_MSA_CACHE_DIR="${BIOKINEMA_MSA_CACHE_DIR:-${BIOKINEMA_ROOT}/msa}"

SYSTEMS=($(ls -d "${INIT_PDB_DIR}"/*/ 2>/dev/null | xargs -n1 basename))
[ ${#SYSTEMS[@]} -eq 0 ] && { echo "ERROR: no systems under ${INIT_PDB_DIR}"; exit 1; }
echo "Checkpoint     : ${CHECKPOINT}"
echo "Output         : ${OUTPUT_DIR}"
echo "beta=${BETA}  N_sample=${N_SAMPLE}  W_G=${WG_START}  history_t=${HISTORY_T}"
echo "Systems (${#SYSTEMS[@]}): ${SYSTEMS[*]}"
echo "GPUs (${NUM_GPUS}): ${GPUS}"
cd "${BIOKINEMA_ROOT}"

build_missing_list() {  # sys, plist -> echo #missing
    local sys="$1" plist="$2"; : > "${plist}"; local n=0
    for pdb in "${INIT_PDB_DIR}/${sys}"/start_*.pdb; do
        [ -e "${pdb}" ] || continue
        local sid; sid=$(basename "${pdb}" .pdb | sed 's/start_//')
        local dump_dir="${TRAJ_DIR}/${sys}/seed_${sid}"
        if [ ! -f "${dump_dir}/start_${sid}/start_${sid}_pred_coordinates.npy" ]; then
            printf "%s\t%s\n" "${pdb}" "${dump_dir}" >> "${plist}"; n=$((n+1))
        fi
    done
    echo "${n}"
}

run_system() {  # gpu_id, sys
    local gpu="$1" sys="$2"
    local syslog="${LOG_DIR}/${sys}.log" plist="${LIST_DIR}/${sys}.txt" wg=${WG_START}
    while [ "${wg}" -ge "${WG_FLOOR}" ]; do
        local nmiss; nmiss=$(build_missing_list "${sys}" "${plist}")
        [ "${nmiss}" -eq 0 ] && { echo "[gpu${gpu}] ${sys} complete"; return 0; }
        echo "[gpu${gpu}] ${sys} W_G=${wg}: ${nmiss} inits $(date '+%H:%M:%S')" | tee -a "${syslog}"
        CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} runner/inference_multi.py \
            --pdb_list_file "${plist}" --seeds ${SEED} \
            --load_checkpoint_path "${CHECKPOINT}" \
            --model.N_cycle ${N_CYCLE} --model.diffusion_module.causal_mask false \
            --model.diffusion_module.beta ${BETA} \
            --data.train_sets inference --data.test_sets inference \
            --sample_diffusion.N_sample ${N_SAMPLE} --sample_diffusion.N_step ${N_STEP} \
            --sample_diffusion.noise_scale_lambda ${LAMBDA} --sample_diffusion.step_scale_eta ${ETA} \
            --history_noise ${HISTORY_NOISE} --history_t ${HISTORY_T} \
            --infer_setting.sample_diffusion_chunk_size 1 \
            --coarse_frame_num ${COARSE_FRAME_NUM} --coarse_interval ${COARSE_INTERVAL} \
            --fine_frame_num ${FINE_FRAME_NUM} --W_H ${W_H} --W_G ${wg} --steps_per_stage 0 \
            --data.num_dl_workers 1 --data.msa.enable true --load_strict false \
            >> "${syslog}" 2>&1
        local still; still=$(build_missing_list "${sys}" "${plist}")
        [ "${still}" -eq 0 ] && { echo "[gpu${gpu}] ${sys} DONE at W_G=${wg}"; return 0; }
        echo "[gpu${gpu}] ${sys} ${still} left at W_G=${wg} -> halving" | tee -a "${syslog}"
        wg=$((wg/2))
    done
    echo "[gpu${gpu}] ${sys} GAVE UP below W_G floor"; return 1
}

worker() { local gpu="$1"; shift; for s in "$@"; do run_system "${gpu}" "${s}"; done; echo "[gpu${gpu}] worker done"; }

declare -A G
for i in "${!SYSTEMS[@]}"; do gi=${GPU_ARR[$((i % NUM_GPUS))]}; G[$gi]="${G[$gi]:-} ${SYSTEMS[$i]}"; done
pids=()
for gpu in "${GPU_ARR[@]}"; do
    list=(${G[$gpu]:-}); [ ${#list[@]} -eq 0 ] && continue
    worker "${gpu}" "${list[@]}" > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
    pids+=($!); echo "GPU ${gpu}: ${list[*]}"
done
echo "Launched ${#pids[@]} workers; waiting..."
for pid in "${pids[@]}"; do wait ${pid} || echo "WARN: worker ${pid} non-zero exit"; done
N=$(find "${TRAJ_DIR}" -name '*_pred_coordinates.npy' | wc -l)
E=$(find "${INIT_PDB_DIR}" -name '*.pdb' | wc -l)
echo "ALL DONE $(date '+%F %H:%M:%S') — pred npys: ${N} / ${E} inits"
