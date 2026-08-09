#!/bin/bash

SCRIPT_DIR="$HOME/mmgbsa"
DATA_DIR="$HOME/Nurr1/md_handoff"
BASE_OUT_DIR="$DATA_DIR/MD_RESULTS_APO"

echo "=========================================================="
echo "🎯 Apo (Ligand-free) MM/GBSA 자동화 런너 시작"
echo "=========================================================="

mkdir -p "$BASE_OUT_DIR"

# ---------------------------------------------------------
# [1] Phase 1 (AF-1 / PAS-B) Apo 실행
# ---------------------------------------------------------
echo "▶ Phase 1 Apo 시뮬레이션 시작..."
P1_OUT="$BASE_OUT_DIR/phase1_apo_wt"
mkdir -p "$P1_OUT"

"$SCRIPT_DIR/run_mmgbsa_protein_single.sh" \
    "$DATA_DIR/phase1_af1_pasb/components/wt_r7_chainA.pdb" \
    "$DATA_DIR/phase1_af1_pasb/components/wt_r7_chainB.pdb" \
    "$P1_OUT"

# ---------------------------------------------------------
# [2] Phase 2 (LBD / SRC1) Apo 실행
# ---------------------------------------------------------
# (주의: Phase 2 파일명은 잠시 후 확인해서 정확히 맞출 예정입니다. 일단 주석 처리!)
# echo "▶ Phase 2 Apo 시뮬레이션 시작..."
# P2_OUT="$BASE_OUT_DIR/phase2_apo_wt"
# mkdir -p "$P2_OUT"
#
# "$SCRIPT_DIR/run_mmgbsa_protein_single.sh" \
#     "$DATA_DIR/phase2_af2_src1/components/[ChainA_이름].pdb" \
#     "$DATA_DIR/phase2_af2_src1/components/[ChainB_이름].pdb" \
#     "$P2_OUT"

echo "=========================================================="
echo "🎉 모든 Apo 기준점(Baseline) 파이프라인 배치가 종료되었습니다!"