#!/bin/bash

# 에러 발생 시 즉시 종료
set -e

# 인자 확인
if [ -z "$1" ]; then
    echo "사용법: ./run_ternary_full.sh [PDB_폴더_경로]"
    echo "예시: ./run_ternary_full.sh data/1ABC"
    exit 1
fi

TARGET_DIR=$1
# 틸드(~) 대신 $HOME 변수를 사용하여 파이썬 파일 인식 오류 원천 차단
SCRIPT_DIR="$HOME/sejin_workspace/mmgbsa"

if [ ! -d "$TARGET_DIR" ]; then
    echo "에러: '$TARGET_DIR' 디렉터리를 찾을 수 없습니다."
    exit 1
fi

PDB_ID=$(basename "$TARGET_DIR")
cd "$TARGET_DIR"

# 단백질 파일 확인
if [ ! -f "${PDB_ID}_chainA.pdb" ] || [ ! -f "${PDB_ID}_chainB.pdb" ]; then
    echo "에러: ${PDB_ID}_chainA.pdb 또는 ${PDB_ID}_chainB.pdb 파일이 없습니다!"
    exit 1
fi

echo "=========================================================="
echo "🚀 [ $PDB_ID ] 다중 리간드 + 양방향 Ternary 파이프라인 시작"
echo "=========================================================="

# 폴더 내의 모든 .sdf 파일을 찾아서 반복 실행
shopt -s nullglob
SDF_FILES=(*.sdf)

if [ ${#SDF_FILES[@]} -eq 0 ]; then
    echo "에러: 현재 폴더에 .sdf 파일이 하나도 없습니다!"
    exit 1
fi

for LIGAND_SDF in "${SDF_FILES[@]}"; do
    LIGAND_NAME=$(basename "$LIGAND_SDF" .sdf)
    
    # 도출될 2개의 결과 파일명 정의
    RES_DIR1="${PDB_ID}_${LIGAND_NAME}_[A+Glue]_vs_B.dat"
    RES_DIR2="${PDB_ID}_${LIGAND_NAME}_[B+Glue]_vs_A.dat"
    
    echo "----------------------------------------------------------"
    echo ">>> 진행 중인 리간드: $LIGAND_NAME"
    
    # 스킵 로직 (두 결과 파일이 모두 존재하면 패스)
    if [ -f "$RES_DIR1" ] && [ -f "$RES_DIR2" ]; then
        echo "✅ [ $LIGAND_NAME ] 이미 두 방향의 결과 파일이 존재하여 건너뜁니다."
        continue
    fi
    
    # 작업 폴더 생성
    WORK_DIR="md_run_${LIGAND_NAME}"
    if [ -d "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    fi
    mkdir -p "$WORK_DIR"
    
    # 단백질과 리간드를 작업 폴더로 복사
    cp ${PDB_ID}_chainA.pdb "$WORK_DIR/"
    cp ${PDB_ID}_chainB.pdb "$WORK_DIR/"
    cp "$LIGAND_SDF" "$WORK_DIR/ligand_original.sdf"
    
    cd "$WORK_DIR"
    
    # ========================================================
    # [공통 단계] 단백질 전처리 및 리간드 파라미터화 (1회만 수행)
    # ========================================================
    echo "[$LIGAND_NAME] 1/7 단백질 체인 수소 제거 중..."
    pdb4amber -i ${PDB_ID}_chainA.pdb -o protA_noh.pdb --nohyd > pdb4amber_A.log 2>&1
    pdb4amber -i ${PDB_ID}_chainB.pdb -o protB_noh.pdb --nohyd > pdb4amber_B.log 2>&1
    
    echo "[$LIGAND_NAME] 2/7 리간드 수소 추가 및 3D 변환 중 (Open Babel)..."
    obabel -isdf ligand_original.sdf -osdf -O ligand_h.sdf -h > obabel.log 2>&1

    echo "[$LIGAND_NAME] 3/7 리간드 전하 계산 및 파라미터화 중..."
    echo "  -> 1차 시도: AM1-BCC (구조 최적화 생략)"

    # if문을 쓰면 에러가 나도 스크립트가 뻗지 않고 else 블록으로 넘어갑니다.
    if antechamber -i ligand_h.sdf -fi sdf -o lig_bcc.mol2 -fo mol2 -c bcc -s 2 -ek "maxcyc=0" > antechamber.log 2>&1; then
        echo "  ✅ AM1-BCC 계산 성공!"
    else
        echo "  ⚠️ AM1-BCC 계산 실패 (로그: antechamber.log 참조)"
        echo "  -> 2차 시도: Gasteiger 방식으로 자동 우회(Fallback) 합니다..."
        antechamber -i ligand_h.sdf -fi sdf -o lig_bcc.mol2 -fo mol2 -c gas -s 2 > antechamber_gas.log 2>&1
        echo "  ✅ Gasteiger 계산 성공!"
    fi

    parmchk2 -i lig_bcc.mol2 -f mol2 -o lig.frcmod > parmchk2.log 2>&1
    # ========================================================
    # [방향 1] Receptor: [ProtA + Glue], Ligand: [ProtB]
    # ========================================================
    echo "[$LIGAND_NAME] 4/7 (방향1) [Prot A + Glue] vs [Prot B] 모델 구축 및 계산 중..."
    mkdir -p dir1_A_Glue_vs_B
    cd dir1_A_Glue_vs_B
    
    # 이전 단계의 파일들을 소프트 링크로 연결
    ln -s ../protA_noh.pdb .
    ln -s ../protB_noh.pdb .
    ln -s ../lig_bcc.mol2 .
    ln -s ../lig.frcmod .
    
    cat <<EOF > tleap_dir1.in
source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.phosaa10
source leaprc.gaff2

protA = loadpdb protA_noh.pdb
glue  = loadmol2 lig_bcc.mol2
protB = loadpdb protB_noh.pdb
loadamberparams lig.frcmod

# 그룹화 1: ProtA와 Glue가 먼저 하나로 묶임
rec  = combine {protA glue}
lig  = combine {protB}
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
    tleap -f tleap_dir1.in > tleap.log 2>&1
    python "$SCRIPT_DIR/run_md.py" $(pwd) > openmm.log 2>&1
    MMPBSA.py -O -i "$SCRIPT_DIR/mmpbsa.in" -sp complex_solv.prmtop -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y production.dcd > mmpbsa.log 2>&1
    
    if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
        cp FINAL_RESULTS_MMPBSA.dat "../../$RES_DIR1"
    fi
    cd ..

    # ========================================================
    # [방향 2] Receptor: [ProtB + Glue], Ligand: [ProtA]
    # ========================================================
    echo "[$LIGAND_NAME] 5/7 (방향2) [Prot B + Glue] vs [Prot A] 모델 구축 및 계산 중..."
    mkdir -p dir2_B_Glue_vs_A
    cd dir2_B_Glue_vs_A
    
    ln -s ../protA_noh.pdb .
    ln -s ../protB_noh.pdb .
    ln -s ../lig_bcc.mol2 .
    ln -s ../lig.frcmod .
    
    cat <<EOF > tleap_dir2.in
source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.phosaa10
source leaprc.gaff2

protB = loadpdb protB_noh.pdb
glue  = loadmol2 lig_bcc.mol2
protA = loadpdb protA_noh.pdb
loadamberparams lig.frcmod

# 그룹화 2: ProtB와 Glue가 먼저 하나로 묶임
rec  = combine {protB glue}
lig  = combine {protA}
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
    tleap -f tleap_dir2.in > tleap.log 2>&1
    python "$SCRIPT_DIR/run_md.py" $(pwd) > openmm.log 2>&1
    MMPBSA.py -O -i "$SCRIPT_DIR/mmpbsa.in" -sp complex_solv.prmtop -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y production.dcd > mmpbsa.log 2>&1
    
    if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
        cp FINAL_RESULTS_MMPBSA.dat "../../$RES_DIR2"
    fi
    cd ..
    
    # ========================================================
    echo "[$LIGAND_NAME] 7/7 계산 완료 및 파일 추출 완료!"
    cd ../
done

echo "=========================================================="
echo "🎉 [ $PDB_ID ] 모든 리간드의 양방향 계산이 완료되었습니다!"