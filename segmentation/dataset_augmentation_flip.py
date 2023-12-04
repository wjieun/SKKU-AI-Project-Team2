import albumentations as A
import cv2
import os

# 이미지와 마스크가 저장된 디렉토리 경로
image_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_1016/img'
mask_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_1016/Mask'

# 증강된 이미지와 마스크를 저장할 디렉토리
augmented_image_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_2032/img'
augmented_mask_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_2032/Mask'

# 이미지와 마스크 모두에 적용할 데이터 증강 설정
shared_augmentations = A.Compose([
    A.HorizontalFlip(p=1.0),

])

# os.listdir을 사용하여 디렉토리 내의 파일 이름을 가져옴
for i, filename in enumerate(os.listdir(image_directory)):
    print(i)
    print(filename)
    if filename.endswith('.jpg'):
        img_path = os.path.join(image_directory, filename)
        # 이미지 파일명과 마스크 파일명이 다르므로, 파일명을 적절히 변환
        mask_filename = filename.replace('image', 'mask')
        mask_path = os.path.join(mask_directory, mask_filename)

        # 이미지와 마스크를 읽음
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 이미지나 마스크가 None이면 건너뜀
        if image is None or mask is None:
            print(f"Cannot read image or mask for {filename}")
            continue

        # 공통 데이터 증강을 수행
        augmented_shared = shared_augmentations(image=image, mask=mask)
        aug_image = augmented_shared['image']
        aug_mask = augmented_shared['mask']

        aug_img_path = os.path.join(
            augmented_image_directory, f'{i}.jpg')
        aug_mask_path = os.path.join(
            augmented_mask_directory, f'{i}.jpg')

        cv2.imwrite(aug_img_path, aug_image)
        cv2.imwrite(aug_mask_path, aug_mask)

