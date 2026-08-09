#!/bin/bash

# 에러 발생 시 즉시 종료
set -e

# ========================================================
# 1. 인자 확인 및 변수 할당
# ========================================================
if [ "$#" -lt 3 ]; then
    echo "=========================================================="
    echo "사용법: ./run_ternary_explicit.sh [Prot_A.pdb] [Prot_B.pdb] [Ligand.sdf] [출력_폴더_경로(선택)]"
    echo "예시 1 (출력 자동 지정): ./run_ternary_explicit.sh 1ABC_A.pdb 1ABC_B.pdb ligand.sdf"
    echo "예시 2 (출력 수동 지정): ./run_ternary_explicit.sh 1ABC_A.pdb 1ABC_B.pdb ligand.sdf /경로/내맘대로/폴더"
    echo "=========================================================="
    exit 1
fi

# 파이썬 및 입력 파일(mmpbsa.in) 경로 고정
SCRIPT_DIR="$HOME/mmgbsa"

# 입력 파일들의 절대 경로 추출 (실행 위치가 달라져도 찾을 수 있도록)
PROT_A=$(readlink -f "$1")
PROT_B=$(readlink -f "$2")
LIGAND=$(readlink -f "$3")

# 원본 파일명 추출 (확장자 제외)
PROT_A_NAME=$(basename "$PROT_A" .pdb)
PROT_B_NAME=$(basename "$PROT_B" .pdb)
LIG_NAME=$(basename "$LIGAND" .sdf)

# 입력 파일 존재 여부 확인
for file in "$PROT_A" "$PROT_B" "$LIGAND"; do
    if [ ! -f "$file" ]; then
        echo "에러: 파일 '$file'을(를) 찾을 수 없습니다."
        exit 1
    fi
done

# ========================================================
# 2. 작업 출력 폴더(Output Directory) 설정
# ========================================================
# 4번째 인자(출력 폴더)가 주어졌으면 그것을 사용하고, 없으면 조합해서 고유 이름 생성
if [ -n "$4" ]; then
    OUT_DIR=$(readlink -m "$4")
else
    # 조합 예: 1ABC_A_1ABC_B_ligand_MMGBSA_Run
    OUT_DIR="$PWD/${PROT_A_NAME}_${PROT_B_NAME}_${LIG_NAME}_MMGBSA_Run"
fi

# 도출될 결과 파일명 정의
RES_DAT="${PROT_A_NAME}_${LIG_NAME}_[A+Glue]_vs_${PROT_B_NAME}.dat"

echo "=========================================================="
echo "🚀 단방향 Ternary 파이프라인 시작 ([Prot A + Glue] vs Prot B)"
echo " - Protein A : $PROT_A_NAME"
echo " - Protein B : $PROT_B_NAME"
echo " - Ligand    : $LIG_NAME"
echo " - 작업 폴더 : $OUT_DIR"
echo "=========================================================="

# 스킵 로직 (결과 파일이 해당 폴더에 존재하면 패스)
if [ -f "$OUT_DIR/$RES_DAT" ]; then
    echo "✅ 이미 결과 파일이 존재하여 작업을 건너뜁니다."
    exit 0
fi

# 폴더 생성 및 이동
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# 원본 파일들을 내부 작업용 고정 이름으로 복사 (스크립트 꼬임 방지)
cp "$PROT_A" protA_raw.pdb
cp "$PROT_B" protB_raw.pdb
cp "$LIGAND" ligand_original.sdf

# ========================================================
# 3. [공통 단계] 단백질 전처리 및 리간드 파라미터화
# ========================================================
echo "[1/5] 단백질 체인 수소 제거 중..."
pdb4amber -i protA_raw.pdb -o protA_noh.pdb --nohyd > pdb4amber_A.log 2>&1
pdb4amber -i protB_raw.pdb -o protB_noh.pdb --nohyd > pdb4amber_B.log 2>&1

echo "[2/5] 리간드 수소 추가 및 3D 변환 중 (Open Babel)..."
obabel -isdf ligand_original.sdf -osdf -O ligand_h.sdf -h > obabel.log 2>&1

echo "[3/5] 리간드 전하 계산 및 파라미터화 중..."
echo "  -> 1차 시도: AM1-BCC (구조 최적화 생략)"
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
# 4. [방향 1] Receptor: [ProtA + Glue], Ligand: [ProtB]
# ========================================================
echo "[4/5] [Prot A + Glue] vs [Prot B] 모델 구축 및 계산 중..."
mkdir -p dir1_A_Glue_vs_B
cd dir1_A_Glue_vs_B

ln -sf ../protA_noh.pdb .
ln -sf ../protB_noh.pdb .
ln -sf ../lig_bcc.mol2 .
ln -sf ../lig.frcmod .

cat <<EOF > tleap_dir1.in
source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.phosaa10
source leaprc.gaff2

protA = loadpdb protA_noh.pdb
glue  = loadmol2 lig_bcc.mol2
protB = loadpdb protB_noh.pdb
loadamberparams lig.frcmod

# 그룹화: ProtA와 Glue가 먼저 하나로 묶임
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
python "$SCRIPT_DIR/run_md_refined.py" $(pwd) > openmm.log 2>&1
MMPBSA.py -O -i "$SCRIPT_DIR/mmpbsa.in" -sp complex_solv.prmtop -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y production.dcd > mmpbsa.log 2>&1

if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
    cp FINAL_RESULTS_MMPBSA.dat "../$RES_DAT"
fi
cd ..

echo "[5/5] 계산 완료 및 파일 추출 완료!"
echo "=========================================================="
echo "🎉 단방향 계산 작업이 성공적으로 완료되었습니다!"
echo "결과 폴더: $OUT_DIR"