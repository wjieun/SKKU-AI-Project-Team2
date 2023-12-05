import pyheif
import torch
import cv2
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from PIL import Image

def heic_to_jpeg(heic_dir, jpeg_dir):
    heif_file = pyheif.read(heic_dir)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride
        )
    image.save(jpeg_dir,"JPEG")

def preprocess_image(img_path, size=(256, 256)):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(*size),
        A.Normalize(),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    return transformed['image']

# 단일 이미지를 검출하는 함수
def detect(jpeg_dir, output_dir, model, device, save=True):
    image = preprocess_image(jpeg_dir)

    with torch.no_grad():
        image = image.float().unsqueeze(0).to(device)
        outputs = model(image)

    pil_img = Image.open(jpeg_dir).convert('RGB').resize((256, 256), resample=Image.NEAREST)
    img_cpu = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred = torch.sigmoid(outputs).squeeze(0).cpu().numpy().squeeze()
    Image.fromarray((pred * 255).astype(np.uint8)).save(output_dir)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap='gray')
    plt.title('Predicted Mask')
    plt.show()
    plt.savefig('image_with_mask.png')

# Use as below
# heic_to_jpeg("./inputs/IMG_6732.HEIC", "./inputs/IMG_6732.jpg")
# 밑의 모델은 seg_model을 통해 정의합니다.
# detect("./inputs/IMG_6732.jpg", model , device, './outputs/IMG_6732_mask.jpg')
