#!/bin/bash

# [하드코딩] 사용할 GPU 번호를 지정하세요. (예: "0" 또는 "1" 또는 "0,1")
# export CUDA_VISIBLE_DEVICES="0"

# 입력받은 파라미터가 없는지 확인
if [ $# -eq 0 ]; then
    echo "오류: 실행할 파라미터를 입력해주세요. (예: $0 3 4 5)"
    exit 1
fi

echo "========================================"
echo "지정된 GPU 번호: $CUDA_VISIBLE_DEVICES"
echo "총 $# 개의 작업을 시작합니다: $@"
echo "========================================"

# 사용자가 입력한 파라미터를 하나씩 꺼내서 반복 실행
for param in "$@"
do
    echo "[시작] 파라미터 $param 작업 중... (GPU: $CUDA_VISIBLE_DEVICES)"
    
    # 기존 mmgbsa 스크립트 실행
    ./run_mmgbsa_target_error_report.sh "$param"
    
    echo "[완료] 파라미터 $param 작업이 끝났습니다."
    echo "----------------------------------------"
done

echo "모든 작업이 성공적으로 완료되었습니다!"