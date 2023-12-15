import cv2
import numpy as np
import os

# 마스크 파일이 있는 디렉토리 경로
mask_directory = '/Users/hwany/Documents/인공지능프로젝트 작업파일/PLSU/Mask'

# 마스크 파일 목록을 가져옵니다
mask_files = os.listdir(mask_directory)

# 0과 255가 아닌 값을 가지는 마스크를 찾습니다
for mask_file in mask_files:
    # 마스크 이미지를 그레이스케일로 읽어옵니다
    mask_path = os.path.join(mask_directory, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 마스크 값이 0과 255 이외의 값을 가지는지 확인합니다
    if np.any((mask != 0) & (mask != 255)) & (mask_file != '.DS_Store'):
        print(f"Mask '{mask_file}' has values other than 0 and 255.")
