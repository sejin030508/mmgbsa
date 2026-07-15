#!/bin/bash

# 에러 발생 시 즉시 종료 설정
set -e

# [매우 중요] 쉘 스크립트 내부에서 conda activate를 사용하기 위한 초기화 세팅
source ~/miniconda3/etc/profile.d/conda.sh

# 인자가 제공되었는지 확인
if [ -z "$1" ]; then
    echo "사용법: ./run_biokinema_batch.sh [타겟_그룹_번호] (예: ./run_biokinema_batch.sh 3)"
    exit 1
fi

TARGET_GROUP=$1
DATA_BASE_DIR="/home/sejin/data/coreset_classified/${TARGET_GROUP}"
SCRIPT_DIR="/home/sejin/mmgbsa"
BIOKINEMA_DIR="/home/sejin/BioKinema"

# 타겟 그룹 폴더가 존재하는지 확인
if [ ! -d "$DATA_BASE_DIR" ]; then
    echo "에러: $DATA_BASE_DIR 디렉터리를 찾을 수 없습니다."
    exit 1
fi

echo "=========================================================="
echo "🚀 Target Group [ $TARGET_GROUP ] BioKinema 자동화 파이프라인 시작"
echo "=========================================================="

# 대상 폴더 내부의 시스템(PDB ID) 폴더들을 순회
for SYSTEM_DIR in "$DATA_BASE_DIR"/*/; do
    SYSTEM_DIR=${SYSTEM_DIR%/}
    PDB_ID=$(basename "$SYSTEM_DIR")
    RESULT_FILE="${PDB_ID}_BioKinema_MMPBSA_results.dat"
    
    echo "----------------------------------------------------------"
    echo ">>> 진행 중인 타겟: $PDB_ID"
    
    # ====================================================================
    # 🌟 스킵(Skip) 로직 🌟
    if [ -f "$SYSTEM_DIR/$RESULT_FILE" ]; then
        echo "✅ [ $PDB_ID ] 이미 BioKinema 결과 파일이 존재하여 작업을 건너뜁니다."
        continue
    fi
    # ====================================================================
    
    # 0. 작업용 폴더(biokinema_run) 생성
    WORK_DIR="$SYSTEM_DIR/biokinema_run"
    if [ -d "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    fi
    mkdir -p "$WORK_DIR"
    
    # 원본 파일 복사
    cp "$SYSTEM_DIR/${PDB_ID}_protein.pdb" "$WORK_DIR/"
    cp "$SYSTEM_DIR/${PDB_ID}_ligand_opt.mol2" "$WORK_DIR/"
    
    cd "$WORK_DIR"
    
    # ------------------------------------------------------------------
    conda activate md_mmgbsa
    # ------------------------------------------------------------------
    
    # 1. 단백질 전처리 (수소 제거)
    echo "[$PDB_ID] 1/6 단백질 수소 제거 중..."
    pdb4amber -i ${PDB_ID}_protein.pdb -o protein_noh.pdb --nohyd > pdb4amber.log 2>&1
    
    # 2. 리간드 파라미터화 (Antechamber)
    echo "[$PDB_ID] 2/6 리간드 파라미터화 중..."
    antechamber -i ${PDB_ID}_ligand_opt.mol2 -fi mol2 -o ligand_bcc.mol2 -fo mol2 -c bcc -s 2 > antechamber.log 2>&1
    parmchk2 -i ligand_bcc.mol2 -f mol2 -o ligand.frcmod > parmchk2.log 2>&1
    
    # 3. 위상 파일 및 PDB 생성 (Tleap)
    echo "[$PDB_ID] 3/6 Tleap 위상 파일 생성 및 PDB 교정 중..."
    cat <<EOF > tleap.in
source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
source leaprc.phosaa10
rec = loadpdb protein_noh.pdb
lig = loadmol2 ligand_bcc.mol2
loadamberparams ligand.frcmod
comp = combine {rec lig}
saveamberparm comp complex.prmtop complex.inpcrd
saveamberparm rec receptor.prmtop receptor.inpcrd
saveamberparm lig ligand.prmtop ligand.inpcrd
quit
EOF
    tleap -f tleap.in > tleap.log 2>&1
    ambpdb -p complex.prmtop -c complex.inpcrd > complex_reference_raw.pdb

    # [핵심] 회원님 환경에서 가장 안정적으로 작동하는 PDB 패치 코드 유지
    cat << 'EOF' > fix_pdb_for_bk.py
import sys
with open('complex_reference_raw.pdb', 'r') as fin, open('complex_reference.pdb', 'w') as fout:
    for line in fin:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            res_name = line[17:20].strip()
            if res_name == "MOL":
                line = "HETATM" + line[6:]
                line = line[:17] + "LIG" + line[20:]
                line = line[:21] + "Z" + line[22:]
            else:
                if line[21] == " ":
                    line = line[:21] + "A" + line[22:]
        fout.write(line)
EOF
    python fix_pdb_for_bk.py

    # ------------------------------------------------------------------
    conda activate biokinema
    # ------------------------------------------------------------------
    
    # 4. BioKinema AI 궤적 추론
    echo "[$PDB_ID] 4/6 BioKinema AI 궤적 추론 중..."
    cd "$BIOKINEMA_DIR"
    
    # 🌟 [MSA 캐시 충돌 방지] dump_dir에 타겟 ID를 포함시켜 고유 MSA 캐시 폴더 생성 🌟
    UNIQUE_DUMP="$WORK_DIR/bk_output_${PDB_ID}"
    
    bash inference.sh \
        --input_file "$WORK_DIR/complex_reference.pdb" \
        --dump_dir "$UNIQUE_DUMP" \
        --checkpoint_path ./biokinema.pt > "$WORK_DIR/biokinema.log" 2>&1
    cd "$WORK_DIR"

    # ------------------------------------------------------------------
    conda activate md_mmgbsa
    # ------------------------------------------------------------------
    
    # 5. DCD 궤적 변환 및 OpenMM 수소 복원/충돌 해소
    echo "[$PDB_ID] 5/6 OpenMM DCD 변환 및 수소 복원/충돌 해소 중..."
    
    # 🌟 [폴더/파일명 복구] npy_to_dcd.py가 하드코딩된 경로를 찾을 수 있도록 이름을 강제로 되돌림 🌟
    rm -rf "$WORK_DIR/bk_output" 2>/dev/null || true
    mv "$UNIQUE_DUMP" "$WORK_DIR/bk_output"
    
    # 내부의 폴더명(bk_output_PDBID_0 -> bk_output_0)과 파일명 변경
    mv "$WORK_DIR/bk_output/bk_output_${PDB_ID}_0" "$WORK_DIR/bk_output/bk_output_0" 2>/dev/null || true
    mv "$WORK_DIR/bk_output/bk_output_0/bk_output_${PDB_ID}_0_pred_coordinates.npy" "$WORK_DIR/bk_output/bk_output_0/bk_output_0_pred_coordinates.npy" 2>/dev/null || true
    
    python "$SCRIPT_DIR/npy_to_dcd.py" "$WORK_DIR" > npy_to_dcd.log 2>&1
    
    # 6. MM/GBSA 결합 에너지 계산
    echo "[$PDB_ID] 6/6 MMPBSA 결합 에너지 계산 중..."
    MMPBSA.py -O -i "$SCRIPT_DIR/mmpbsa.in" \
        -sp complex.prmtop \
        -cp complex.prmtop \
        -rp receptor.prmtop \
        -lp ligand.prmtop \
        -y biokinema_trajectory.dcd > mmpbsa.log 2>&1
    
    # 마무리 및 결과 추출
    echo "[$PDB_ID] 마무리 및 파일 정리 중..."
    if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
        cp FINAL_RESULTS_MMPBSA.dat "$SYSTEM_DIR/$RESULT_FILE"
    fi
    
    echo ">>> $PDB_ID 완료! 결과: $RESULT_FILE"
done

echo "=========================================================="
echo "🎉 Target Group [ $TARGET_GROUP ] 전체 처리가 완료되었습니다!"