#!/bin/bash

# 에러 발생 시 즉시 종료
set -e

# ========================================================
# 1. 인자 확인 및 변수 할당
# ========================================================
if [ "$#" -lt 2 ]; then
    echo "=========================================================="
    echo "사용법: ./run_mmgbsa_apo_single.sh [Prot_A.pdb] [Prot_B.pdb] [출력_폴더_경로(선택)]"
    echo "예시 1: ./run_mmgbsa_apo_single.sh chainA.pdb chainB.pdb"
    echo "예시 2: ./run_mmgbsa_apo_single.sh chainA.pdb chainB.pdb /경로/내맘대로/Apo_Result"
    echo "=========================================================="
    exit 1
fi

SCRIPT_DIR="$HOME/mmgbsa"

# 절대 경로 추출
PROT_A=$(readlink -f "$1")
PROT_B=$(readlink -f "$2")

# 원본 파일명 추출
PROT_A_NAME=$(basename "$PROT_A" .pdb)
PROT_B_NAME=$(basename "$PROT_B" .pdb)

for file in "$PROT_A" "$PROT_B"; do
    if [ ! -f "$file" ]; then
        echo "에러: 파일 '$file'을(를) 찾을 수 없습니다."
        exit 1
    fi
done

# ========================================================
# 2. 작업 폴더 설정
# ========================================================
if [ -n "$3" ]; then
    OUT_DIR=$(readlink -m "$3")
else
    OUT_DIR="$PWD/Apo_${PROT_A_NAME}_vs_${PROT_B_NAME}_MMGBSA"
fi

RES_DAT="Apo_${PROT_A_NAME}_vs_${PROT_B_NAME}_results.dat"

if [ -f "$OUT_DIR/$RES_DAT" ]; then
    echo "✅ 이미 결과 파일이 존재하여 작업을 건너뜁니다."
    exit 0
fi

echo "=========================================================="
echo "🚀 Apo (단백질-단백질) 파이프라인 시작"
echo " - Protein A : $PROT_A_NAME"
echo " - Protein B : $PROT_B_NAME"
echo " - 작업 폴더 : $OUT_DIR"
echo "=========================================================="

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

cp "$PROT_A" protA_raw.pdb
cp "$PROT_B" protB_raw.pdb

# ========================================================
# 3. 단백질 전처리
# ========================================================
echo "[1/4] 두 단백질 체인 수소 제거 중..."
pdb4amber -i protA_raw.pdb -o receptor_noh.pdb --nohyd > pdb4amber_rec.log 2>&1
pdb4amber -i protB_raw.pdb -o ligand_noh.pdb --nohyd > pdb4amber_lig.log 2>&1

# ========================================================
# 4. Tleap 위상 파일 생성
# ========================================================
echo "[2/4] Tleap 위상 파일 및 물 상자 생성 중..."
cat <<EOF > tleap.in
source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.phosaa10

rec = loadpdb receptor_noh.pdb
lig = loadpdb ligand_noh.pdb
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

# ========================================================
# 5. MD 및 MM/GBSA 시뮬레이션
# ========================================================
echo "[3/4] OpenMM 시뮬레이션 실행 중 (100 ns)..."
python "$SCRIPT_DIR/run_md_refined.py" $(pwd) > openmm.log 2>&1

echo "[4/4] 단백질-단백질 MM/GBSA 결합 에너지 계산 중..."
MMPBSA.py -O -i "$SCRIPT_DIR/mmpbsa.in" -sp complex_solv.prmtop -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y production.dcd > mmpbsa.log 2>&1

if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
    cp FINAL_RESULTS_MMPBSA.dat "../$RES_DAT"
fi
cd ..

echo "=========================================================="
echo "🎉 Apo 계산 완료! 최종 결과: $OUT_DIR/$RES_DAT"