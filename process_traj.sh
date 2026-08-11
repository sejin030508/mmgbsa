#!/bin/bash

# 에러 발생 시 종료
set -e

if [ -z "$1" ]; then
    echo "=========================================================="
    echo "사용법: ./process_traj.sh [결과_폴더_경로]"
    echo "예시: ./process_traj.sh ~/Nurr1/md_handoff/MD_RESULTS/EC_wt_r7_P07_S20_123_wt"
    echo "=========================================================="
    exit 1
fi

TARGET_DIR=$(readlink -f "$1")

# 타겟 폴더 내에서 production.dcd 파일 찾기 (Apo, Ternary 구조 상관없이 적용하기 위함)
DCD_FILE=$(find "$TARGET_DIR" -name "production.dcd" | head -n 1)

if [ -z "$DCD_FILE" ]; then
    echo "⚠️ 에러: '$TARGET_DIR' 내에 production.dcd 파일이 존재하지 않습니다."
    exit 1
fi

# 작업할 디렉터리 경로 추출
WORK_DIR=$(dirname "$DCD_FILE")
cd "$WORK_DIR"

if [ ! -f "complex_solv.prmtop" ]; then
    echo "⚠️ 에러: 토폴로지 파일(complex_solv.prmtop)이 존재하지 않습니다."
    exit 1
fi

echo "=========================================================="
echo "🚀 MD 궤적 전처리 시작"
echo "작업 폴더: $WORK_DIR"
echo "=========================================================="


# cpptraj 입력 파일 동적 생성
cat <<EOF > cpptraj.in
# 1. 뼈대와 궤적 파일 로드
parm complex_solv.prmtop
trajin production.dcd

# 2. 주기적 경계(PBC) 문제 해결
autoimage

# 3. 물 분자(WAT) 및 이온(Na+, Cl-) 삭제 + 삭제된 상태의 뼈대 파일 동시 생성!
strip :WAT,Na+,Cl- outprefix dry

# 4. 단백질 흔들림 방지 (정렬)
rms fit @CA,C,N,O

# 5. 가벼워진 궤적 파일 출력
trajout dry_aligned.dcd

run
quit
EOF

# cpptraj 실행
cpptraj -i cpptraj.in > cpptraj.log 2>&1

# cpptraj가 만들어준 뼈대 파일 이름을 깔끔하게 'dry.prmtop'으로 변경
if [ -f "dry.complex_solv.prmtop" ]; then
    mv dry.complex_solv.prmtop dry.prmtop
fi

echo "✅ 전처리 완료! 아래 두 파일만 로컬 컴퓨터로 다운로드하세요."
echo " - 뼈대 파일: $WORK_DIR/dry.prmtop"
echo " - 궤적 파일: $WORK_DIR/dry_aligned.dcd"
echo "=========================================================="
