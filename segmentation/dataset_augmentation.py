# import albumentations as A
# import cv2
# import os

# # 이미지와 마스크가 저장된 디렉토리 경로
# image_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU/img'
# mask_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU/Mask'

# # 증강된 이미지와 마스크를 저장할 디렉토리
# augmented_image_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_NEW/img'
# augmented_mask_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_NEW/Mask'

# # 이미지와 마스크 모두에 적용할 데이터 증강 설정
# shared_augmentations = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0, rotate_limit=15,
#                        scale_limit=(-0.1, 0.3), p=0.5)
# ])

# # 이미지에만 적용할 데이터 증강 설정
# image_only_augmentations = A.Compose([
#     A.RandomBrightnessContrast(
#         brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#     A.CLAHE(p=0.5)
# ])


# # os.listdir을 사용하여 디렉토리 내의 파일 이름을 가져옴
# for i, filename in enumerate(os.listdir(image_directory)):
#     print(i)
#     print(filename)
#     if filename.endswith('.jpg'):
#         img_path = os.path.join(image_directory, filename)
#         # 이미지 파일명과 마스크 파일명이 다르므로, 파일명을 적절히 변환
#         mask_filename = filename.replace('image', 'mask')
#         mask_path = os.path.join(mask_directory, mask_filename)

#         # 이미지와 마스크를 읽음
#         image = cv2.imread(img_path)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         # 이미지나 마스크가 None이면 건너뜀
#         if image is None or mask is None:
#             print(f"Cannot read image or mask for {filename}")
#             continue

#         # 데이터 증강을 수행하고 저장
#         for j in range(4):
#             # 공통 데이터 증강을 수행
#             augmented_shared = shared_augmentations(image=image, mask=mask)
#             aug_image = augmented_shared['image']
#             aug_mask = augmented_shared['mask']

#             # 이미지에만 추가적인 데이터 증강을 수행
#             augmented_image = image_only_augmentations(image=aug_image)[
#                 'image']

#             aug_img_path = os.path.join(
#                 augmented_image_directory, f'{i}_{j}.jpg')
#             aug_mask_path = os.path.join(
#                 augmented_mask_directory, f'{i}_{j}.jpg')

#             cv2.imwrite(aug_img_path, augmented_image)
#             cv2.imwrite(aug_mask_path, aug_mask)


import os
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split

# 이미지와 마스크가 저장된 디렉토리 경로
image_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU/img'
mask_directory = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU/Mask'

# 각 데이터셋에 대한 저장 디렉토리
train_image_dir = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_NEW/train/image'
train_mask_dir = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_NEW/train/Mask'
val_image_dir = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_NEW/val/image'
val_mask_dir = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_NEW/val/Mask'
test_image_dir = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_NEW/test/image'
test_mask_dir = '/Users/hwany/Documents/인공지능프로젝트_작업파일/PLSU_NEW/test/Mask'

# 이미지와 마스크 모두에 적용할 데이터 증강 설정
shared_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0, rotate_limit=15,
                       scale_limit=(-0.1, 0.3), p=0.5)
])

# 이미지에만 적용할 데이터 증강 설정
image_only_augmentations = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    A.CLAHE(p=0.5)
])


# 데이터 증강 및 저장 함수
def augment_and_save(image_path, mask_path, save_image_dir, save_mask_dir, augment=False, num_augments=1):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(num_augments):
        aug_image, aug_mask = image, mask
        if augment:
            # 공통 데이터 증강 적용
            augmented_shared = shared_augmentations(
                image=aug_image, mask=aug_mask)
            aug_image = augmented_shared['image']
            aug_mask = augmented_shared['mask']

            # 이미지에만 추가적인 데이터 증강 적용
            aug_image = image_only_augmentations(image=aug_image)['image']

        # 파일명에 증강 인덱스 추가
        augmented_filename = f"{base_filename}_{i}.jpg"
        cv2.imwrite(os.path.join(save_image_dir,
                    augmented_filename), aug_image)
        cv2.imwrite(os.path.join(save_mask_dir, augmented_filename), aug_mask)


# 이미지와 마스크 파일 리스트 생성 및 데이터셋 분할
image_files = [os.path.join(image_directory, f)
               for f in os.listdir(image_directory) if f.endswith('.jpg')]
mask_files = [os.path.join(mask_directory, f.replace(
    'image', 'mask')) for f in image_files]

# 데이터셋 분할
train_files, test_files, train_masks, test_masks = train_test_split(
    image_files, mask_files, test_size=0.3, random_state=42)
val_files, test_files, val_masks, test_masks = train_test_split(
    test_files, test_masks, test_size=0.5, random_state=42)

# Train 데이터셋 증강 및 저장
for img_path, mask_path in zip(train_files, train_masks):
    augment_and_save(img_path, mask_path, train_image_dir,
                     train_mask_dir, augment=True)

# Validation 및 Test 데이터셋에 대해 원본 저장
for img_path, mask_path, save_image_dir, save_mask_dir in zip(val_files + test_files, val_masks + test_masks, [val_image_dir] * len(val_files) + [test_image_dir] * len(test_files), [val_mask_dir] * len(val_masks) + [test_mask_dir] * len(test_masks)):
    augment_and_save(img_path, mask_path, save_image_dir,
                     save_mask_dir, augment=False, num_augments=1)
