import os
import cv2
import numpy as np

# 마스크 파일들이 저장된 디렉토리 경로를 설정하세요.
mask_directory_path = '/Users/hwany/Documents/인공지능프로젝트 작업파일/PLSU/Mask'

# 모든 픽셀이 검정색인 마스크의 경로를 저장할 리스트를 생성합니다.
black_masks_paths = []

# 디렉토리 내의 모든 파일을 순회합니다.
for mask_filename in os.listdir(mask_directory_path):
    # 파일 경로를 완성합니다.
    mask_path = os.path.join(mask_directory_path, mask_filename)
    # 이미지를 그레이스케일로 읽어옵니다.
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 모든 픽셀이 검정색인지 확인합니다.
    if np.all(mask == 0):
        # 해당하는 경우 리스트에 파일 경로를 추가합니다.
        black_masks_paths.append(mask_path)

# 모든 픽셀이 검정색인 마스크 파일의 경로를 출력합니다.
print(black_masks_paths)
