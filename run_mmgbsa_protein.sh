#!/bin/bash

# 에러 발생 시 즉시 종료
set -e

# 인자가 제공되었는지 확인 (폴더 경로 입력받기)
if [ -z "$1" ]; then
    echo "사용법: ./run_pp_mmgbsa.sh [PDB_폴더_경로]"
    echo "예시: ./run_pp_mmgbsa.sh data/1ABC"
    exit 1
fi

TARGET_DIR=$1
SCRIPT_DIR="$HOME/sejin_workspace/mmgbsa"

# 입력한 폴더가 존재하는지 확인
if [ ! -d "$TARGET_DIR" ]; then
    echo "에러: '$TARGET_DIR' 디렉터리를 찾을 수 없습니다."
    exit 1
fi

# 폴더 이름에서 PDB_ID 추출 (예: data/1ABC -> 1ABC)
PDB_ID=$(basename "$TARGET_DIR")

echo "=========================================================="
echo "🚀 [ $PDB_ID ] 단백질-단백질 파이프라인 시작"
echo "=========================================================="

cd "$TARGET_DIR"

# 스킵(Skip) 로직
if [ -f "${PDB_ID}_PP_MMPBSA_results.dat" ]; then
    echo "✅ 이미 결과 파일(${PDB_ID}_PP_MMPBSA_results.dat)이 존재하여 작업을 건너뜁니다."
    exit 0
fi

# 0. 작업용 폴더 생성
WORK_DIR="md_run"
if [ -d "$WORK_DIR" ]; then
    rm -rf "$WORK_DIR"
fi
mkdir -p $WORK_DIR

# 입력 파일 존재 여부 확인 후 복사
if [ ! -f "${PDB_ID}_chainA.pdb" ] || [ ! -f "${PDB_ID}_chainB.pdb" ]; then
    echo "에러: ${PDB_ID}_chainA.pdb 또는 ${PDB_ID}_chainB.pdb 파일을 찾을 수 없습니다!"
    exit 1
fi

cp ${PDB_ID}_chainA.pdb $WORK_DIR/
cp ${PDB_ID}_chainB.pdb $WORK_DIR/

cd $WORK_DIR

# 1. 단백질 전처리 (수소 제거 - 두 체인 모두 수행)
echo "[$PDB_ID] 1/4 두 단백질 체인 수소 제거 중..."
pdb4amber -i ${PDB_ID}_chainA.pdb -o receptor_noh.pdb --nohyd > pdb4amber_rec.log 2>&1
pdb4amber -i ${PDB_ID}_chainB.pdb -o ligand_noh.pdb --nohyd > pdb4amber_lig.log 2>&1

# 2. 위상 파일 생성 (Tleap)
echo "[$PDB_ID] 2/4 Tleap 위상 파일 생성 중..."
cat <<EOF > tleap.in
source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.phosaa10

# 수용체와 타겟 단백질 로드
rec = loadpdb receptor_noh.pdb
lig = loadpdb ligand_noh.pdb

# 복합체 결합
comp = combine {rec lig}

# 진공 상태의 위상 파일 저장
saveamberparm comp complex.prmtop complex.inpcrd
saveamberparm rec receptor.prmtop receptor.inpcrd
saveamberparm lig ligand.prmtop ligand.inpcrd

# 솔벤트 박스 및 이온 추가
solvatebox comp TIP3PBOX 10.0
addions comp Na+ 0
addions comp Cl- 0
addions comp Na+ 40
addions comp Cl- 40

# 솔벤트 상태의 위상 파일 저장
saveamberparm comp complex_solv.prmtop complex_solv.inpcrd
quit
EOF
tleap -f tleap.in > tleap.log 2>&1

# 3. OpenMM 시뮬레이션
echo "[$PDB_ID] 3/4 OpenMM 시뮬레이션 실행 중..."
python $SCRIPT_DIR/run_md.py $(pwd) > openmm.log 2>&1

# 4. MM/GBSA 에너지 계산
echo "[$PDB_ID] 4/4 단백질-단백질 MMPBSA 결합 에너지 계산 중..."
MMPBSA.py -O -i $SCRIPT_DIR/mmpbsa.in -sp complex_solv.prmtop -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y production.dcd > mmpbsa.log 2>&1

# 5. 결과 파일 추출
echo "[$PDB_ID] 마무리 및 파일 정리 중..."
if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
    cp FINAL_RESULTS_MMPBSA.dat ../${PDB_ID}_PP_MMPBSA_results.dat
fi

cd ../

echo ">>> $PDB_ID 완료! 최종 결과: ${PDB_ID}_PP_MMPBSA_results.dat"