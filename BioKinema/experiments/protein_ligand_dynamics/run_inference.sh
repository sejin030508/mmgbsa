#!/bin/bash
# =============================================================================
# BioKinema paper-result reproduction  —  STEP 1: generate trajectories
# (experiments/protein_ligand_dynamics/run_inference.sh)
# -----------------------------------------------------------------------------
# Reproduces the all-atom trajectories behind the manuscript figures for the
# three protein systems below, using the released sqrt-time-conditioning
# checkpoint (5999 EMA):
#
#     pin1        -> Fig 4c  (allosteric loop motion)
#     adk         -> Fig 4b  (induced-fit open<->closed transition)
#     misato_ood  -> Fig 2a  (ligand physical stability)
#
# Initial-frame structures are bundled in data/init_structures/<system>/.
# The model checkpoint and the output directory are REQUIRED arguments; nothing
# is hardcoded and trajectories are NEVER written inside this folder.
#
# Usage
# -----
#   bash run_inference.sh --system {pin1|adk|misato_ood} \
#        --checkpoint_path /abs/path/to/5999_ema_0.999.pt \
#        --output_dir /abs/path/for/trajectories \
#        [--input_file <single .cif>]   # default: every init structure of the system
#        [--gpu <id>]                   # default: 0
#
# Output layout (one folder per input structure):
#   <output_dir>/<system>/<stem>/<stem>_pred_coordinates.npy   # [frame, sample, atom, 3]
#   <output_dir>/<system>/<stem>/predictions/<stem>_s{S}_f{F}_wounresol.cif
# Frame f0 is the input (initial) structure; frames f1.. are the generated path.
# Then plot with:  run_analysis.sh --trajs_root <output_dir>
# =============================================================================
set -euo pipefail

# ---- locate self + repo root (this script lives in experiments/protein_ligand_dynamics) ----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"   # .../BioKinema (provides runner/ + protenix package)

# ---- machine-specific env (override by exporting before calling) ------------
# DeepSpeed EvoformerAttention needs CUTLASS + CUDA. Edit these for your host.
export CUTLASS_PATH="${CUTLASS_PATH:-/cto_studio/xtalpi_lab/fengbin/cutlass}"
export CUDA_HOME="${CUDA_HOME:-/cto_studio/xtalpi_lab/softwares/cuda-11.8}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"
export USE_DEEPSPEED_EVO_ATTTENTION="${USE_DEEPSPEED_EVO_ATTTENTION:-true}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export BIOKINEMA_MSA_CACHE_DIR="${BIOKINEMA_MSA_CACHE_DIR:-${REPO_ROOT}/msa}"
# Reference ligand bond geometry (only needed for misato_ood featurization):
export BIOKINEMA_LIGAND_BONDS_DIR="${BIOKINEMA_LIGAND_BONDS_DIR:-/cto_studio/xtalpi_lab/fengbin/datas/misato_bio_noref}"
export BIOKINEMA_QUIET_CCD_MSG=1
export PYTHONUNBUFFERED=1

# ---- shared sampling hyper-parameters (identical to the paper run) ----------
# These exactly match example_runs/_figrun.py (sqrt-5999 figure inference).
SEED=101
N_CYCLE=10
N_STEP=20
LAMBDA=1.75            # sample_diffusion.noise_scale_lambda
ETA=1.5               # sample_diffusion.step_scale_eta
BETA=0.5   # sqrt(t) ALiBi temporal scaling for this checkpoint
COARSE_INTERVAL=10     # ns between coarse frames (training resolution)
FINE_FRAME_NUM=1       # no fine interpolation
W_H=1                  # history window
HISTORY_NOISE=0.0
HISTORY_T=1.6e-1       # advertised noise level of the history/conditioning frame

# ---- args -------------------------------------------------------------------
SYSTEM=""
CHECKPOINT_PATH=""
OUTPUT_DIR=""
INPUT_FILE=""
GPU=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system)          SYSTEM="$2"; shift 2;;
        --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2;;
        --output_dir)      OUTPUT_DIR="$2"; shift 2;;
        --input_file)      INPUT_FILE="$2"; shift 2;;
        --gpu)             GPU="$2"; shift 2;;
        *) echo "Unknown argument: $1" >&2; exit 1;;
    esac
done

[[ -n "$SYSTEM" ]]          || { echo "ERROR: --system {pin1|adk|misato_ood} is required" >&2; exit 1; }
[[ -n "$CHECKPOINT_PATH" ]] || { echo "ERROR: --checkpoint_path is required (model is not hardcoded)" >&2; exit 1; }
[[ -f "$CHECKPOINT_PATH" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }
[[ -n "$OUTPUT_DIR" ]]      || { echo "ERROR: --output_dir is required (trajectories are written there, NOT inside this folder)" >&2; exit 1; }

# ---- per-system trajectory settings (length / #samples) ---------------------
# CFN = coarse_frame_num (includes f0); total time = (CFN-1) * COARSE_INTERVAL ns
INIT_DIR="${HERE}/data/init_structures/${SYSTEM}"
case "$SYSTEM" in
    pin1)        CFN=151; N_SAMPLE=10; W_G=50 ;;  # ~1.5 us, 10 replicas
    adk)         CFN=501; N_SAMPLE=10; W_G=10 ;;  # ~5.0 us, 10 replicas
    misato_ood)  CFN=101; N_SAMPLE=1;  W_G=100 ;; # ~1.0 us, 1 replica
    *) echo "ERROR: unknown system '$SYSTEM' (use pin1|adk|misato_ood)" >&2; exit 1;;
esac
[[ -d "$INIT_DIR" ]] || { echo "ERROR: bundled init structures not found: $INIT_DIR" >&2; exit 1; }

DUMP_DIR="${OUTPUT_DIR}/${SYSTEM}"
mkdir -p "$DUMP_DIR"

# ---- collect input cif(s) from the bundled init structures ------------------
if [[ -n "$INPUT_FILE" ]]; then
    INPUTS=("$INPUT_FILE")
else
    INPUTS=()
    while IFS= read -r f; do INPUTS+=("$f"); done < <(ls "${INIT_DIR}"/*.cif | sort)
fi
[[ ${#INPUTS[@]} -gt 0 ]] || { echo "ERROR: no input .cif found for system '$SYSTEM' in ${INIT_DIR}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$REPO_ROOT"

echo "[reproduce] system=$SYSTEM  cfn=$CFN  N_sample=$N_SAMPLE  W_G=$W_G  gpu=$GPU"
echo "[reproduce] checkpoint=$CHECKPOINT_PATH"
echo "[reproduce] init_structures=$INIT_DIR  ($((${#INPUTS[@]})) structure(s))"
echo "[reproduce] output=$DUMP_DIR"

for cif in "${INPUTS[@]}"; do
    stem="$(basename "$cif" .cif)"
    pl="$(mktemp)"
    printf '%s\t%s\n' "$cif" "$DUMP_DIR" > "$pl"
    echo "[reproduce] >>> $stem"
    python -u runner/inference_multi.py \
        --pdb_list_file "$pl" \
        --seeds "$SEED" \
        --load_checkpoint_path "$CHECKPOINT_PATH" \
        --model.N_cycle "$N_CYCLE" \
        --model.diffusion_module.causal_mask false \
        --model.diffusion_module.beta "$BETA" \
        --data.train_sets inference --data.test_sets inference \
        --sample_diffusion.N_sample "$N_SAMPLE" \
        --sample_diffusion.N_step "$N_STEP" \
        --sample_diffusion.noise_scale_lambda "$LAMBDA" \
        --sample_diffusion.step_scale_eta "$ETA" \
        --infer_setting.sample_diffusion_chunk_size 1 \
        --coarse_frame_num "$CFN" \
        --coarse_interval "$COARSE_INTERVAL" \
        --fine_frame_num "$FINE_FRAME_NUM" \
        --W_H "$W_H" --W_G "$W_G" \
        --steps_per_stage 0 \
        --history_noise "$HISTORY_NOISE" --history_t "$HISTORY_T" \
        --data.num_dl_workers 1 --data.msa.enable true --load_strict false
    rm -f "$pl"
done

echo "[reproduce] DONE system=$SYSTEM"
