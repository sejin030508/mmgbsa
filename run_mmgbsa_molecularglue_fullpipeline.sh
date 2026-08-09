#!/bin/bash

# 에러 발생 시 즉시 종료하지 않고 다음 타겟으로 넘어감
set +e

# 스크립트 및 데이터 경로 명시적 지정
SCRIPT_DIR="$HOME/mmgbsa"
DATA_DIR="$HOME/Nurr1/md_handoff"
RECOMMENDED_CSV="$DATA_DIR/recommended_starting_structures.csv"
CANDIDATE_CSV="$DATA_DIR/candidate_manifest.csv"
BASE_OUT_DIR="$DATA_DIR/MD_RESULTS"

# 사용자가 입력한 특정 행 번호 받기 (입력 없으면 전체 실행)
TARGET_ROW=$1

echo "=========================================================="
echo "🎯 Nurr1 Molecular Glue 파이프라인 자동화 런너 시작"
if [ -n "$TARGET_ROW" ]; then
    echo "📌 단일 실행 모드: CSV의 ${TARGET_ROW}번째 타겟만 처리합니다."
else
    echo "📌 연속 실행 모드: CSV의 모든 타겟을 순차적으로 처리합니다."
fi
echo "=========================================================="

mkdir -p "$BASE_OUT_DIR"
cd "$DATA_DIR"

# 데이터 행 카운터 초기화 (헤더 제외)
ROW_COUNT=1

# CSV 헤더를 건너뛰고 한 줄씩 읽기
tail -n +2 "$RECOMMENDED_CSV" | while IFS=',' read -r priority phase candidate_id role variant filepath; do
    
    # 타겟 행 번호가 주어졌는데 현재 행 번호와 일치하지 않으면 패스
    if [ -n "$TARGET_ROW" ] && [ "$ROW_COUNT" -ne "$TARGET_ROW" ]; then
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi
    
    # 개행문자 제거 및 CSV 안의 'md_handoff/' 경로 앞부분을 현재 폴더에 맞게 잘라내기
    candidate_id=$(echo "$candidate_id" | tr -d '\r')
    phase=$(echo "$phase" | tr -d '\r')
    filepath=$(echo "$filepath" | sed 's|^md_handoff/||' | tr -d '\r')

    echo "----------------------------------------------------------"
    echo "▶ 현재 처리 중 [타겟 번호: $ROW_COUNT]: [Phase: $phase] Candidate: $candidate_id"
    
    # 원본 PDB 파일 존재 확인
    if [ ! -f "$filepath" ]; then
        echo "⚠️ PDB 파일을 찾을 수 없습니다: $filepath (스킵)"
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    # 1. SMILES 문자열 추출 (candidate_manifest.csv에서 candidate_id로 검색)
    SMILES=$(awk -F',' -v id="$candidate_id" '$2==id {print $7}' "$CANDIDATE_CSV" | head -n 1 | tr -d '\r')    
    if [ -z "$SMILES" ]; then
        echo "⚠️ SMILES를 찾을 수 없습니다. (스킵)"
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi
    echo "  - SMILES: $SMILES"

    # 2. Phase에 따른 단백질 Chain ID 설정
    if [[ "$phase" == "phase1_af1_pasb" ]]; then
        CHAIN_A="A" # SRC1 PAS-B
        CHAIN_B="B" # Nurr1 AF-1
    else
        CHAIN_A="A" # Nurr1 LBD
        CHAIN_B="B" # SRC1 NR-box
    fi
    CHAIN_L="C"

    # 작업용 임시 폴더
    WORK_DIR="${BASE_OUT_DIR}/${candidate_id}_${variant}"
    mkdir -p "$WORK_DIR"

    # Step 1 실행 (추출)
    echo "  - [Step 1] PDB 구조 분리 중..."
    python "$SCRIPT_DIR/extract_chains.py" "$filepath" --out_dir "$WORK_DIR" -A "$CHAIN_A" -B "$CHAIN_B" -L "$CHAIN_L"
    
    # Step 2 실행 (SMILES 보정)
    echo "  - [Step 2] 리간드 화학 정보 보정 (RDKit)..."
    python "$SCRIPT_DIR/fix_ligand.py" \
        --pdb "$WORK_DIR/Ligand_raw.pdb" \
        --smiles "$SMILES" \
        --out "$WORK_DIR/Ligand_fixed.sdf"
    
    # 에러 체크 (SDF 생성이 안 되었으면 스킵)
    if [ ! -f "$WORK_DIR/Ligand_fixed.sdf" ]; then
        echo "⚠️ SDF 생성 실패 (스킵)"
        ROW_COUNT=$((ROW_COUNT + 1))
        continue
    fi

    # Step 3 실행 (MD / MMGBSA)
    echo "  - [Step 3] MD 및 MM/GBSA 시뮬레이션 돌입..."
    "$SCRIPT_DIR/run_mmgbsa_moleculeglue_single.sh" \
        "$WORK_DIR/Prot_A.pdb" \
        "$WORK_DIR/Prot_B.pdb" \
        "$WORK_DIR/Ligand_fixed.sdf" \
        "$WORK_DIR/MMGBSA_Run"

    echo "✅ [타겟 번호: $ROW_COUNT] $candidate_id 처리 완료!"
    ROW_COUNT=$((ROW_COUNT + 1))

done

echo "=========================================================="
echo "🎉 파이프라인 작업이 종료되었습니다!"