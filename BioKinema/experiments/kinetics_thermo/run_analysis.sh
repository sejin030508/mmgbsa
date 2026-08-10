#!/bin/bash
# ============================================================================
# BioKinema kinetics/thermodynamics benchmark — STEP 2: analyze trajectories.
#
# Builds the analysis cache from the generated trajectories + the bundled MD reference
# (data/msm_cache: pre-extracted TICA basis + K=10 MSM per system), then produces the
# manuscript artifacts:
#   1. mfpt_jointplot.{pdf,png}            — main-text MFPT figure (MD vs BioKinema, pooled)
#   2. figure_kinetics_sup_page{1,2,3}.pdf — per-system supplement figure (20 systems)
#   3. per_system_table.csv                — aggregated per-system result table (+ MEAN row)
#
# Self-contained: needs only python (numpy, scipy, matplotlib, seaborn, mdtraj, PIL) +
# the bundled data/ + the trajectories from run_inference.sh. No raw MD required.
#
# REQUIRED: --traj_dir <output_dir/biokinema_trajs>
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAJ_DIR=""
OUT_DIR=""
MSM_DIR="${HERE}/data/msm_cache"
STRUCT_DIR="${HERE}/data/structures"
PYTHON="${PYTHON:-/cto_studio/xtalpi_lab/fengbin/anaconda3/envs/protenix/bin/python}"

usage() {
    echo "Usage: $0 --traj_dir DIR [--out_dir DIR] [--msm_dir DIR] [--python PATH]"
    echo "  --traj_dir DIR   (required) the biokinema_trajs/ dir produced by run_inference.sh"
    echo "  --out_dir DIR    [<traj_dir>/../analysis]"
    echo "  --msm_dir DIR    [${MSM_DIR}]  bundled MD reference (TICA + K=10 MSM)"
    echo "  --python PATH    [${PYTHON}]"
    exit 1
}
while [ $# -gt 0 ]; do
    case "$1" in
        --traj_dir) TRAJ_DIR="$2"; shift 2;;
        --out_dir) OUT_DIR="$2"; shift 2;;
        --msm_dir) MSM_DIR="$2"; shift 2;;
        --python) PYTHON="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done
[ -z "${TRAJ_DIR}" ] && { echo "ERROR: --traj_dir is required"; usage; }
[ -d "${TRAJ_DIR}" ] || { echo "ERROR: traj_dir not found: ${TRAJ_DIR}"; exit 1; }
[ -z "${OUT_DIR}" ] && OUT_DIR="$(cd "${TRAJ_DIR}/.." && pwd)/analysis"
mkdir -p "${OUT_DIR}"
S="${HERE}/scripts"

# Paths consumed by the python scripts (see scripts/*.py).
export KT_MSM_DIR="${MSM_DIR}"
export KT_TRAJ_DIR="${TRAJ_DIR}"
export KT_OUT_DIR="${OUT_DIR}"
export KT_CACHE="${OUT_DIR}/data_cache.pkl"
export KT_STRUCT_DIR="${STRUCT_DIR}"

echo "############ build analysis cache ############"
${PYTHON} "${S}/prepare_data.py" 2>&1 | tee "${OUT_DIR}/log_prepare.txt"

echo "############ 1. MFPT figure (main text) ############"
${PYTHON} "${S}/plot_mfpt.py" 2>&1 | tee "${OUT_DIR}/log_mfpt.txt"

echo "############ 2. per-system supplement figure ############"
${PYTHON} "${S}/plot_supplement.py" 2>&1 | tee "${OUT_DIR}/log_supplement.txt"

echo "############ 3. aggregated per-system table ############"
${PYTHON} "${S}/make_table.py" 2>&1 | tee "${OUT_DIR}/log_table.txt"

echo "DONE. Results in ${OUT_DIR}:"
echo "  mfpt_jointplot.pdf/png  |  figure_kinetics_sup_page{1,2,3}.pdf  |  per_system_table.csv"
