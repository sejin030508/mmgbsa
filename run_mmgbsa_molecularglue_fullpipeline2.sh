#!/bin/bash

# Run one WT representative for every candidate in candidate_manifest.csv.
# A candidate is skipped when its MMGBSA_Run directory already exists.
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/Nurr1/md_handoff}"
CANDIDATE_CSV="$DATA_DIR/candidate_manifest.csv"
STRUCTURE_CSV="$DATA_DIR/structure_manifest.csv"
RECOMMENDED_CSV="$DATA_DIR/recommended_starting_structures.csv"
BASE_OUT_DIR="$DATA_DIR/MD_RESULTS"

# Optional: process only the listed data rows (header excluded).
TARGET_ROWS=("$@")

for target_row in "${TARGET_ROWS[@]}"; do
    if ! [[ "$target_row" =~ ^[1-9][0-9]*$ ]]; then
        echo "Usage: $0 [candidate_manifest_row ...]"
        exit 1
    fi
done

is_selected_row() {
    local current_row=$1
    local target_row

    if [ "${#TARGET_ROWS[@]}" -eq 0 ]; then
        return 0
    fi

    for target_row in "${TARGET_ROWS[@]}"; do
        if [ "$current_row" -eq "$target_row" ]; then
            return 0
        fi
    done
    return 1
}

for required_file in "$CANDIDATE_CSV" "$STRUCTURE_CSV"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required file not found: $required_file"
        exit 1
    fi
done

echo "=========================================================="
echo "Nurr1 Molecular Glue full pipeline (candidate manifest)"
if [ "${#TARGET_ROWS[@]}" -gt 0 ]; then
    echo "Mode: candidate_manifest.csv rows ${TARGET_ROWS[*]} only"
else
    echo "Mode: all candidates in candidate_manifest.csv"
fi
echo "Structure rule: recommended WT representative, otherwise WT model_0"
echo "=========================================================="

mkdir -p "$BASE_OUT_DIR"
cd "$DATA_DIR" || exit 1

ROW_COUNT=1

tail -n +2 "$CANDIDATE_CSV" | while IFS=',' read -r \
    phase candidate_id internal_id priority evidence_tier target smiles \
    route_found route_steps qvina_source_or_worst qvina_mutant_delta \
    boltz_wt_minus_mutant_iptm recommendation; do

    if ! is_selected_row "$ROW_COUNT"; then
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    phase=$(printf '%s' "$phase" | tr -d '\r')
    candidate_id=$(printf '%s' "$candidate_id" | tr -d '\r')
    smiles=$(printf '%s' "$smiles" | tr -d '\r')

    echo "----------------------------------------------------------"
    echo "[$ROW_COUNT] Phase: $phase | Candidate: $candidate_id"

    if [ -z "$candidate_id" ] || [ -z "$smiles" ]; then
        echo "SKIP: candidate_id or SMILES is empty."
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    WORK_DIR="$BASE_OUT_DIR/${candidate_id}_wt"
    if [ -d "$WORK_DIR/MMGBSA_Run" ]; then
        echo "SKIP: MMGBSA_Run directory already exists: $WORK_DIR/MMGBSA_Run"
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    # Reuse the curated representative when available.
    filepath=""
    if [ -f "$RECOMMENDED_CSV" ]; then
        filepath=$(awk -F',' -v id="$candidate_id" \
            'NR > 1 && $3 == id && $5 == "wt" {gsub(/\r/, "", $6); print $6; exit}' \
            "$RECOMMENDED_CSV")
    fi

    # Other manifest candidates use WT model_0 from structure_manifest.csv.
    if [ -z "$filepath" ]; then
        filepath=$(awk -F',' -v id="$candidate_id" \
            'NR > 1 {gsub(/\r/, "", $9)} $1 == id && $6 == "0" && $9 == "wt" {print $7; exit}' \
            "$STRUCTURE_CSV")
    fi

    # Fall back to the first WT structure if model_0 is unavailable.
    if [ -z "$filepath" ]; then
        filepath=$(awk -F',' -v id="$candidate_id" \
            'NR > 1 {gsub(/\r/, "", $9)} $1 == id && $9 == "wt" {print $7; exit}' \
            "$STRUCTURE_CSV")
    fi

    filepath=${filepath#md_handoff/}
    if [ -z "$filepath" ] || [ ! -f "$filepath" ]; then
        echo "SKIP: no WT PDB is listed and present for this candidate."
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    echo "PDB: $filepath"
    echo "SMILES: $smiles"

    mkdir -p "$WORK_DIR"

    echo "[1/3] Extracting chains A, B, and C..."
    python "$SCRIPT_DIR/extract_chains.py" "$filepath" \
        --out_dir "$WORK_DIR" -A A -B B -L C
    if [ $? -ne 0 ] || [ ! -s "$WORK_DIR/Prot_A.pdb" ] || \
       [ ! -s "$WORK_DIR/Prot_B.pdb" ] || [ ! -s "$WORK_DIR/Ligand_raw.pdb" ]; then
        echo "FAIL: chain extraction failed or produced an empty component."
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    echo "[2/3] Restoring ligand bond orders from SMILES..."
    python "$SCRIPT_DIR/fix_ligand.py" \
        --pdb "$WORK_DIR/Ligand_raw.pdb" \
        --smiles "$smiles" \
        --out "$WORK_DIR/Ligand_fixed.sdf"
    if [ $? -ne 0 ] || [ ! -s "$WORK_DIR/Ligand_fixed.sdf" ]; then
        echo "FAIL: Ligand_fixed.sdf was not created."
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    echo "[3/3] Running MD and MM/GBSA..."
    "$SCRIPT_DIR/run_mmgbsa_moleculeglue_single.sh" \
        "$WORK_DIR/Prot_A.pdb" \
        "$WORK_DIR/Prot_B.pdb" \
        "$WORK_DIR/Ligand_fixed.sdf" \
        "$WORK_DIR/MMGBSA_Run"

    if [ $? -eq 0 ]; then
        echo "DONE: $candidate_id"
    else
        echo "FAIL: MD/MMGBSA returned a non-zero exit status for $candidate_id."
    fi

    ROW_COUNT=$((ROW_COUNT + 1))
done

echo "=========================================================="
echo "Pipeline finished."
echo "=========================================================="
