#!/bin/bash

# 에러 발생 시 즉시 종료 설정 (화면 출력은 없어도 에러 나면 안전하게 멈춤)
set -e

# 인자가 제공되었는지 확인
if [ -z "$1" ]; then
    echo "사용법: ./run_batch.sh [타겟_그룹_번호] (예: ./run_batch.sh 3)"
    exit 1
fi

TARGET_GROUP=$1
DATA_BASE_DIR="/home/sejin/data/coreset_classified/${TARGET_GROUP}"
SCRIPT_DIR="/home/sejin/mmgbsa"

# 타겟 그룹 폴더가 존재하는지 확인
if [ ! -d "$DATA_BASE_DIR" ]; then
    echo "에러: $DATA_BASE_DIR 디렉터리를 찾을 수 없습니다."
    exit 1
fi

echo "=========================================================="
echo "🚀 Target Group [ $TARGET_GROUP ] 자동화 파이프라인 시작"
echo "=========================================================="

# 대상 폴더 내부의 시스템(PDB ID) 폴더들을 순회
for SYSTEM_DIR in "$DATA_BASE_DIR"/*/; do
    PDB_ID=$(basename "$SYSTEM_DIR")
    
    echo "----------------------------------------------------------"
    echo ">>> 진행 중인 타겟: $PDB_ID"
    
    # ====================================================================
    # 🌟 스킵(Skip) 로직 추가 🌟
    # 해당 시스템 폴더 안에 최종 결과 파일이 이미 존재하면 작업을 건너뜁니다.
    if [ -f "$SYSTEM_DIR/${PDB_ID}_MMPBSA_results.dat" ]; then
        echo "✅ [ $PDB_ID ] 이미 결과 파일이 존재하여 작업을 건너뜁니다."
        continue
    fi
    # ====================================================================
    
    cd "$SYSTEM_DIR"
    
    # 0. 작업용 폴더(md_run) 생성 및 이동 (이전 흔적 깔끔히 지우고 새로 시작)
    WORK_DIR="md_run"
    if [ -d "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    fi
    mkdir -p $WORK_DIR
    
    # 원본 파일들 중 필요한 것만 작업 폴더로 복사
    cp ${PDB_ID}_protein.pdb $WORK_DIR/
    cp ${PDB_ID}_ligand_opt.mol2 $WORK_DIR/
    
    cd $WORK_DIR
    
    # 1. 단백질 전처리 (수소 제거)
    echo "[$PDB_ID] 1/5 단백질 수소 제거 중..."
    pdb4amber -i ${PDB_ID}_protein.pdb -o protein_noh.pdb --nohyd > pdb4amber.log 2>&1
    
    # 2. 리간드 파라미터화 (Antechamber)
    echo "[$PDB_ID] 2/5 리간드 전하 계산 및 파라미터화 중..."
    antechamber -i ${PDB_ID}_ligand_opt.mol2 -fi mol2 -o ligand_bcc.mol2 -fo mol2 -c bcc -s 2 > antechamber.log 2>&1
    parmchk2 -i ligand_bcc.mol2 -f mol2 -o ligand.frcmod > parmchk2.log 2>&1
    
    # 3. 위상 파일 생성 (Tleap)
    echo "[$PDB_ID] 3/5 Tleap 위상 파일 생성 중..."
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
solvatebox comp TIP3PBOX 10.0
addions comp Na+ 0
addions comp Cl- 0
addions comp Na+ 40
addions comp Cl- 40
saveamberparm comp complex_solv.prmtop complex_solv.inpcrd
quit
EOF
    tleap -f tleap.in > tleap.log 2>&1
    
    # 4. OpenMM 시뮬레이션
    echo "[$PDB_ID] 4/5 OpenMM 시뮬레이션(3ns) 실행 중..."
    python $SCRIPT_DIR/run_md.py $(pwd) > openmm.log 2>&1
    
    # 5. MM/GBSA 에너지 계산
    echo "[$PDB_ID] 5/5 MMPBSA 결합 에너지 계산 중..."
    MMPBSA.py -O -i $SCRIPT_DIR/mmpbsa.in -sp complex_solv.prmtop -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y production.dcd > mmpbsa.log 2>&1
    
    # 6. 결과 파일 추출 및 청소
    echo "[$PDB_ID] 마무리 및 파일 정리 중..."
    if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
        cp FINAL_RESULTS_MMPBSA.dat ../${PDB_ID}_MMPBSA_results.dat
    fi
    
    cd ../
    
    echo ">>> $PDB_ID 완료! 결과: ${PDB_ID}_MMPBSA_results.dat"
done

echo "=========================================================="
echo "🎉 Target Group [ $TARGET_GROUP ] 전체 처리가 완료되었습니다!"