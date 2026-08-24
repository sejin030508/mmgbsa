#!/usr/bin/env bash

# Run 14-3-3/ERalpha molecular-glue MM/GBSA jobs from the A6000 dataset.
#
# Usage:
#   ./run_mmgbsa_1433_eralpha_pipeline.sh                 # all PDB IDs
#   ./run_mmgbsa_1433_eralpha_pipeline.sh 8BXQ 8BYY       # selected PDB IDs
#
# Optional environment overrides:
#   DATA_DIR=/path/to/14-3-3_ERalpha
#   BASE_OUT_DIR=/path/to/MD_RESULTS

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/dataset/14-3-3_ERalpha}"
BASE_OUT_DIR="${BASE_OUT_DIR:-$DATA_DIR/MD_RESULTS}"
SINGLE_RUN_SCRIPT="$SCRIPT_DIR/run_mmgbsa_moleculeglue_single.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [PDB_ID ...]

Run all PDB IDs in $DATA_DIR when no IDs are given, or only the supplied IDs.
Expected input files for each ID:
  $DATA_DIR/<PDB_ID>/<PDB_ID>_14-3-3.pdb
  $DATA_DIR/<PDB_ID>/<PDB_ID>_ER.pdb
  $DATA_DIR/<PDB_ID>/<PDB_ID>_ligand.sdf

Environment variables:
  DATA_DIR      Dataset root (default: $HOME/dataset/14-3-3_ERalpha)
  BASE_OUT_DIR  Result root   (default: $DATA_DIR/MD_RESULTS)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Dataset directory not found: $DATA_DIR" >&2
    exit 1
fi

if [[ ! -f "$SINGLE_RUN_SCRIPT" ]]; then
    echo "ERROR: Single-run script not found: $SINGLE_RUN_SCRIPT" >&2
    exit 1
fi

declare -a TARGET_IDS=()
if [[ "$#" -gt 0 ]]; then
    for pdb_id in "$@"; do
        TARGET_IDS+=("$(printf '%s' "$pdb_id" | tr '[:lower:]' '[:upper:]')")
    done
else
    while IFS= read -r pdb_id; do
        TARGET_IDS+=("$pdb_id")
    done < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d ! -name "$(basename "$BASE_OUT_DIR")" -exec basename {} \; | sort)
fi

if [[ "${#TARGET_IDS[@]}" -eq 0 ]]; then
    echo "ERROR: No PDB ID directories found in $DATA_DIR" >&2
    exit 1
fi

mkdir -p "$BASE_OUT_DIR"

completed=0
skipped=0
failed=0

echo "=========================================================="
echo "14-3-3/ERalpha molecular-glue MM/GBSA pipeline"
echo "Dataset: $DATA_DIR"
echo "Results: $BASE_OUT_DIR"
echo "PDB IDs: ${TARGET_IDS[*]}"
echo "Model: [14-3-3 + ligand] vs ERalpha"
echo "=========================================================="

for pdb_id in "${TARGET_IDS[@]}"; do
    input_dir="$DATA_DIR/$pdb_id"
    protein_1433="$input_dir/${pdb_id}_14-3-3.pdb"
    protein_er="$input_dir/${pdb_id}_ER.pdb"
    ligand="$input_dir/${pdb_id}_ligand.sdf"
    out_dir="$BASE_OUT_DIR/$pdb_id/MMGBSA_Run"
    result_file="$out_dir/${pdb_id}_14-3-3_${pdb_id}_ligand_[A+Glue]_vs_${pdb_id}_ER.dat"

    echo "----------------------------------------------------------"
    echo "PDB ID: $pdb_id"

    missing=0
    for input_file in "$protein_1433" "$protein_er" "$ligand"; do
        if [[ ! -f "$input_file" ]]; then
            echo "SKIP: Required input is missing: $input_file" >&2
            missing=1
        fi
    done
    if [[ "$missing" -ne 0 ]]; then
        ((failed += 1))
        continue
    fi

    if [[ -f "$result_file" ]]; then
        echo "SKIP: Result already exists: $result_file"
        ((skipped += 1))
        continue
    fi

    mkdir -p "$(dirname "$out_dir")"
    if bash "$SINGLE_RUN_SCRIPT" "$protein_1433" "$protein_er" "$ligand" "$out_dir"; then
        echo "DONE: $pdb_id"
        ((completed += 1))
    else
        echo "FAIL: MM/GBSA job failed for $pdb_id" >&2
        ((failed += 1))
    fi
done

echo "=========================================================="
echo "Pipeline finished: completed=$completed skipped=$skipped failed=$failed"
echo "=========================================================="

[[ "$failed" -eq 0 ]]
