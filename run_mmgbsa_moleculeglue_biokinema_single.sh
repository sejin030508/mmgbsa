#!/bin/bash

# Molecular-glue BioKinema trajectory + MM/GBSA pipeline.
# The existing run_mmgbsa_moleculeglue_single.sh remains the MD implementation.
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "사용법: $0 Prot_A.pdb Prot_B.pdb Ligand.sdf [출력_폴더]"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIOKINEMA_DIR="${BIOKINEMA_DIR:-$SCRIPT_DIR/BioKinema}"
BIOKINEMA_CHECKPOINT="${BIOKINEMA_CHECKPOINT:-$BIOKINEMA_DIR/checkpoints/BioKinema_atlas+misato+mdposit_sqrt.pt}"
BIOKINEMA_MSA_CACHE_DIR="${BIOKINEMA_MSA_CACHE_DIR:-$BIOKINEMA_DIR/msa_cache}"
MMGBSA_CONDA_ENV="${MMGBSA_CONDA_ENV:-md_mmgbsa}"
BIOKINEMA_CONDA_ENV="${BIOKINEMA_CONDA_ENV:-biokinema_a6000}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

for required_path in "$BIOKINEMA_DIR" "$BIOKINEMA_CHECKPOINT" "$CONDA_SH"; do
    if [ ! -e "$required_path" ]; then
        echo "에러: 필수 경로를 찾을 수 없습니다: $required_path"
        exit 1
    fi
done

source "$CONDA_SH"
conda activate "$MMGBSA_CONDA_ENV"

PROT_A=$(readlink -f "$1")
PROT_B=$(readlink -f "$2")
LIGAND=$(readlink -f "$3")
for input_file in "$PROT_A" "$PROT_B" "$LIGAND"; do
    if [ ! -f "$input_file" ]; then
        echo "에러: 입력 파일을 찾을 수 없습니다: $input_file"
        exit 1
    fi
done

PROT_A_NAME=$(basename "$PROT_A" .pdb)
PROT_B_NAME=$(basename "$PROT_B" .pdb)
LIG_NAME=$(basename "$LIGAND" .sdf)
if [ -n "${4:-}" ]; then
    OUT_DIR=$(readlink -m "$4")
else
    OUT_DIR="$PWD/${PROT_A_NAME}_${PROT_B_NAME}_${LIG_NAME}_BioKinema_MMGBSA_Run"
fi
RESULT_NAME="${PROT_A_NAME}_${LIG_NAME}_BioKinema_[A+Glue]_vs_${PROT_B_NAME}.dat"

if [ -s "$OUT_DIR/$RESULT_NAME" ]; then
    echo "기존 결과를 재사용합니다: $OUT_DIR/$RESULT_NAME"
    exit 0
fi

echo "=========================================================="
echo "BioKinema molecular-glue MM/GBSA 시작"
echo " - Protein A : $PROT_A_NAME"
echo " - Protein B : $PROT_B_NAME"
echo " - Glue      : $LIG_NAME"
echo " - 출력      : $OUT_DIR"
echo "=========================================================="

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"
cp "$PROT_A" protA_raw.pdb
cp "$PROT_B" protB_raw.pdb
cp "$LIGAND" ligand_original.sdf

echo "[1/6] 단백질 전처리"
pdb4amber -i protA_raw.pdb -o protA_noh.pdb --nohyd > pdb4amber_A.log 2>&1
pdb4amber -i protB_raw.pdb -o protB_noh.pdb --nohyd > pdb4amber_B.log 2>&1

echo "[2/6] glue 수소 추가 및 3D 변환"
obabel -isdf ligand_original.sdf -osdf -O ligand_h.sdf -h > obabel.log 2>&1

echo "[3/6] glue 파라미터화"
if ! antechamber -i ligand_h.sdf -fi sdf -o lig_bcc.mol2 -fo mol2 \
    -c bcc -s 2 -ek "maxcyc=0" > antechamber.log 2>&1; then
    echo "AM1-BCC 실패: Gasteiger 전하로 재시도합니다."
    antechamber -i ligand_h.sdf -fi sdf -o lig_bcc.mol2 -fo mol2 \
        -c gas -s 2 > antechamber_gas.log 2>&1
fi
parmchk2 -i lig_bcc.mol2 -f mol2 -o lig.frcmod > parmchk2.log 2>&1

echo "[4/6] [Protein A + glue] / [Protein B] topology 생성"
RUN_DIR="$OUT_DIR/dir1_A_Glue_vs_B"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
ln -sf ../protA_noh.pdb .
ln -sf ../protB_noh.pdb .
ln -sf ../lig_bcc.mol2 .
ln -sf ../lig.frcmod .

cat <<'EOF' > tleap_biokinema.in
source leaprc.protein.ff14SB
source leaprc.phosaa10
source leaprc.gaff2
protA = loadpdb protA_noh.pdb
glue = loadmol2 lig_bcc.mol2
protB = loadpdb protB_noh.pdb
loadamberparams lig.frcmod
rec = combine {protA glue}
lig = combine {protB}
comp = combine {rec lig}
saveamberparm comp complex.prmtop complex.inpcrd
saveamberparm rec receptor.prmtop receptor.inpcrd
saveamberparm lig ligand.prmtop ligand.inpcrd
quit
EOF

tleap -f tleap_biokinema.in > tleap.log 2>&1
ambpdb -p complex.prmtop -c complex.inpcrd > complex_reference_raw.pdb

# AMBER order is Protein A -> glue -> Protein B. Preserve these as A/Z/B.
python - <<'PY'
from pathlib import Path

source = Path("complex_reference_raw.pdb")
target = Path("complex_reference.pdb")
seen_glue = False
atom_counts = {"A": 0, "B": 0, "Z": 0}

with source.open() as fin, target.open("w") as fout:
    for line in fin:
        if line.startswith(("ATOM", "HETATM")):
            residue_name = line[17:20].strip()
            if residue_name in {"MOL", "LIG"}:
                seen_glue = True
                line = "HETATM" + line[6:]
                line = line[:17] + "LIG" + line[20:]
                chain_id = "Z"
            else:
                chain_id = "B" if seen_glue else "A"
            line = line[:21] + chain_id + line[22:]
            atom_counts[chain_id] += 1
        fout.write(line)

missing = [chain_id for chain_id, count in atom_counts.items() if count == 0]
if missing:
    target.unlink(missing_ok=True)
    raise RuntimeError(f"BioKinema A/Z/B chain 구성 실패: {missing}")
PY

echo "[5/6] BioKinema 추론 및 all-atom DCD 변환"
BK_DUMP_DIR="$RUN_DIR/bk_output"
PREDICTION_FILE=$(find "$BK_DUMP_DIR" -type f -name '*_pred_coordinates.npy' -print -quit 2>/dev/null || true)
if [ -n "$PREDICTION_FILE" ]; then
    echo "기존 BioKinema 예측을 재사용합니다: $PREDICTION_FILE"
else
    rm -rf "$BK_DUMP_DIR"
    conda activate "$BIOKINEMA_CONDA_ENV"
    export BIOKINEMA_MSA_CACHE_DIR
    (
        cd "$BIOKINEMA_DIR"
        bash inference.sh \
            --input_file "$RUN_DIR/complex_reference.pdb" \
            --dump_dir "$BK_DUMP_DIR" \
            --checkpoint_path "$BIOKINEMA_CHECKPOINT"
    ) > biokinema.log 2>&1
fi

conda activate "$MMGBSA_CONDA_ENV"
python "$SCRIPT_DIR/npy_to_dcd_moleculeglue_biokinema.py" \
    "$RUN_DIR" "$BK_DUMP_DIR" > npy_to_dcd_biokinema.log 2>&1

echo "[6/6] MM/GBSA 계산"
MMPBSA.py -O -i "$SCRIPT_DIR/mmpbsa.in" \
    -sp complex.prmtop \
    -cp complex.prmtop \
    -rp receptor.prmtop \
    -lp ligand.prmtop \
    -y biokinema_trajectory.dcd > mmpbsa.log 2>&1

if [ ! -s FINAL_RESULTS_MMPBSA.dat ]; then
    echo "에러: FINAL_RESULTS_MMPBSA.dat가 생성되지 않았습니다."
    exit 1
fi
cp FINAL_RESULTS_MMPBSA.dat "$OUT_DIR/$RESULT_NAME"
echo "완료: $OUT_DIR/$RESULT_NAME"
