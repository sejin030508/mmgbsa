#!/bin/bash
# =============================================================================
# BioKinema paper-result reproduction  —  STEP 2: plot figures from trajectories
#
# Usage:
#   bash run_analysis.sh --trajs_root <dir> \      # REQUIRED: the --output_dir from run_inference.sh
#        [--system {pin1|adk|misato_ood|all}] \    # default: all
#        [--out_dir <dir>]                         # default: <trajs_root>/figures
#
# Requires the `biokinema` conda env active. For misato you must also set
# BIOKINEMA_LIGAND_BONDS_DIR (reference ligand bond geometry).
#
# Reference outputs for comparison are bundled in example_results/.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="${HERE}/scripts"

SYSTEM="all"
TRAJS_ROOT=""
OUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system)     SYSTEM="$2"; shift 2;;
        --trajs_root) TRAJS_ROOT="$2"; shift 2;;
        --out_dir)    OUT_DIR="$2"; shift 2;;
        *) echo "Unknown argument: $1" >&2; exit 1;;
    esac
done
[[ -n "$TRAJS_ROOT" ]] || { echo "ERROR: --trajs_root is required (the --output_dir used by run_inference.sh)" >&2; exit 1; }
[[ -d "$TRAJS_ROOT" ]] || { echo "ERROR: trajs_root not found: $TRAJS_ROOT" >&2; exit 1; }
[[ -n "$OUT_DIR" ]] || OUT_DIR="${TRAJS_ROOT}/figures"
mkdir -p "$OUT_DIR"

run_pin1()   { echo "### Fig 4c (Pin1)";   python -u "${S}/plot_pin1.py"   --traj_dir "${TRAJS_ROOT}/pin1"       --out_dir "$OUT_DIR"; }
run_adk()    { echo "### Fig 4b (ADK)";    python -u "${S}/plot_adk.py"    --traj_dir "${TRAJS_ROOT}/adk"        --out_dir "$OUT_DIR"; }
run_misato() { echo "### Fig 2a (MISATO)"; python -u "${S}/plot_misato.py" --traj_dir "${TRAJS_ROOT}/misato_ood" --out_dir "$OUT_DIR"; }

case "$SYSTEM" in
    pin1)       run_pin1 ;;
    adk)        run_adk ;;
    misato_ood) run_misato ;;
    all)        run_pin1; run_adk; run_misato ;;
    *) echo "ERROR: unknown system '$SYSTEM'" >&2; exit 1;;
esac
echo "[analysis] figures written to $OUT_DIR"
