#!/bin/bash

# CSV-driven molecular-glue pipeline using BioKinema instead of MD.
set +e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_DIR="${MOLECULAR_GLUE_DATA_DIR:-$HOME/Nurr1/md_handoff}"
RECOMMENDED_CSV="$DATA_DIR/recommended_starting_structures.csv"
CANDIDATE_CSV="$DATA_DIR/candidate_manifest.csv"
BASE_OUT_DIR="${MOLECULAR_GLUE_BIOKINEMA_OUT_DIR:-$DATA_DIR/BIOKINEMA_RESULTS}"
TARGET_ROW="${1:-}"

for required_file in "$RECOMMENDED_CSV" "$CANDIDATE_CSV"; do
    if [ ! -f "$required_file" ]; then
        echo "에러: 필수 CSV를 찾을 수 없습니다: $required_file"
        exit 1
    fi
done

echo "=========================================================="
echo "Nurr1 molecular-glue BioKinema 파이프라인 시작"
if [ -n "$TARGET_ROW" ]; then
    echo "CSV target row: $TARGET_ROW"
else
    echo "CSV 전체 target 실행"
fi
echo "=========================================================="

mkdir -p "$BASE_OUT_DIR"
cd "$DATA_DIR" || exit 1
ROW_COUNT=1

tail -n +2 "$RECOMMENDED_CSV" | while IFS=',' read -r priority phase candidate_id role variant filepath; do
    if [ -n "$TARGET_ROW" ] && [ "$ROW_COUNT" -ne "$TARGET_ROW" ]; then
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    candidate_id=$(echo "$candidate_id" | tr -d '\r')
    phase=$(echo "$phase" | tr -d '\r')
    variant=$(echo "$variant" | tr -d '\r')
    filepath=$(echo "$filepath" | sed 's|^md_handoff/||' | tr -d '\r')
    echo "[$ROW_COUNT] $candidate_id ($phase)"

    if [ ! -f "$filepath" ]; then
        echo "PDB 파일 없음: $filepath (건너뜀)"
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    SMILES=$(awk -F',' -v id="$candidate_id" '$2==id {print $7}' \
        "$CANDIDATE_CSV" | head -n 1 | tr -d '\r')
    if [ -z "$SMILES" ]; then
        echo "SMILES 없음 (건너뜀)"
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    CHAIN_A="A"
    CHAIN_B="B"
    CHAIN_L="C"
    WORK_DIR="$BASE_OUT_DIR/${candidate_id}_${variant}"
    mkdir -p "$WORK_DIR"

    if ! python "$SCRIPT_DIR/extract_chains.py" "$filepath" \
        --out_dir "$WORK_DIR" -A "$CHAIN_A" -B "$CHAIN_B" -L "$CHAIN_L"; then
        echo "구조 분리 실패 (건너뜀)"
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    if ! python "$SCRIPT_DIR/fix_ligand.py" \
        --pdb "$WORK_DIR/Ligand_raw.pdb" \
        --smiles "$SMILES" \
        --out "$WORK_DIR/Ligand_fixed.sdf"; then
        echo "ligand 보정 실패 (건너뜀)"
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    if "$SCRIPT_DIR/run_mmgbsa_moleculeglue_biokinema_single.sh" \
        "$WORK_DIR/Prot_A.pdb" \
        "$WORK_DIR/Prot_B.pdb" \
        "$WORK_DIR/Ligand_fixed.sdf" \
        "$WORK_DIR/BioKinema_MMGBSA_Run"; then
        echo "[$ROW_COUNT] $candidate_id 완료"
    else
        echo "[$ROW_COUNT] $candidate_id 실패 (다음 target 진행)"
    fi

    ROW_COUNT=$((ROW_COUNT + 1))
done

echo "BioKinema molecular-glue 파이프라인 종료"
