import pyheif
import torch
import cv2
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from rembg import remove 
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

def remove_background(jpg_dir, output_dir):
    fill_color = (255,255,255)
    img = Image.open(jpg_dir) 
    out = remove(img) 
    im = out.convert('RGBA')
    if im.mode in ('RGBA', 'LA'):
        background = Image.new(im.mode[:-1], im.size, fill_color)
        background.paste(im, im.split()[-1]) # omit transparency
    im = background
    im.convert("RGB").save(output_dir)

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
# heic_to_jpeg("./image/IMG_6732.HEIC", "./image/IMG_6732.jpg")
# remove_background('IMG_1726.jpg', 'IMG_1726_bg.jpg')
# 밑의 모델은 seg_model을 통해 정의합니다.
# detect("./image/IMG_6732.jpg", model , device, './mask/IMG_6732_mask.jpg')