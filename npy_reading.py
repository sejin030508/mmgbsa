import numpy as np
import os
import sys  # sys 모듈 추가

def read_npy(file_path):
    """
    주어진 경로의 .npy 파일을 읽어서 NumPy 배열로 반환합니다.
    """
    # 💡 '~' 기호를 실제 홈 디렉토리 절대 경로로 변환합니다.
    expanded_path = os.path.expanduser(file_path)
    
    # 파일 존재 여부 확인 (변환된 경로 사용)
    if not os.path.exists(expanded_path):
        print(f"오류: '{expanded_path}' 파일을 찾을 수 없습니다.")
        return None

    try:
        # allow_pickle=True는 객체 배열이 포함된 파일을 읽을 때 필요할 수 있습니다.
        data = np.load(expanded_path, allow_pickle=True)
        
        print("✅ 파일 로드 성공!")
        print(f"읽어온 실제 경로: {expanded_path}")
        print(f"데이터 타입: {type(data)}")
        print(f"데이터 형태(Shape): {data.shape}")
        
        return data
        
    except Exception as e:
        print(f"❌ 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

if __name__ == "__main__":
    # 문제가 발생했던 파일 경로
    target_file = "~/data/coreset_classified/3/4m0y/biokinema_run/bk_output/bk_output_0/bk_output_0_is_ligand.npy"
    #target_file = "~/output_6h76_ligz/output_6h76_ligz_0/output_6h76_ligz_0_is_ligand.npy"
    # 함수 실행
    my_array = read_npy(target_file)
    
    # 데이터가 정상적으로 로드된 경우 전체 출력
    if my_array is not None:
        # 💡 핵심 추가: 배열 출력 시 생략(truncation) 제한을 없앰
        np.set_printoptions(threshold=sys.maxsize)
        
        print("\n[전체 데이터 출력]")
        print(my_array)