import numpy as np
import mediapipe as mp
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from CLSmodule import classify
from tensorflow.keras.applications import MobileNetV2

hand_img = cv2.imread('../PLSU/img/image2.jpg')
mask_img = cv2.imread('../PLSU/Mask/image2.png')
image_height, image_width, _ = hand_img.shape

mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
low_intensity_pixels = np.where(np.all(mask_img < 200, axis=-1))

for y, x in zip(low_intensity_pixels[0], low_intensity_pixels[1]):
    mask_img[y, x] = [0, 0, 0]

# MediaPipe Hands 초기화
with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    results = hands.process(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

    if results.multi_handedness:
        for hand_landmarks in results.multi_hand_landmarks:
            WRIST_point = [hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x * image_width, hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y * image_height]
            INDEX_FINGER_MCP_Point = [hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y * image_height]
            MIDDLE_FINGER_MCP_Point = [hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height]
            RING_FINGER_MCP_Point = [hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP].y * image_height]
            
            point_life = [WRIST_point[1]*0.7 + RING_FINGER_MCP_Point[1]*0.3, WRIST_point[0]*0.7 + RING_FINGER_MCP_Point[0]*0.3]
            point_head = [WRIST_point[1]*0.375 + RING_FINGER_MCP_Point[1]*0.625, WRIST_point[0]*0.375 + RING_FINGER_MCP_Point[0]*0.625]
            point_heart = [WRIST_point[1]*0.1875 + RING_FINGER_MCP_Point[1]*0.8125, WRIST_point[0]*0.1875 + RING_FINGER_MCP_Point[0]*0.8125]

            point_total = [point_life, point_head, point_heart]
            life_img, head_img, heart_img = classify(mask_img, point_total)
    else:
        print("오류: 손을 감지할 수 없습니다.")

img_size = (128, 128)
resized_mask_img = cv2.resize(mask_img, img_size)
resized_life_img = cv2.resize(life_img, img_size)
resized_heart_img = cv2.resize(heart_img, img_size)

# head line prediction
mask_array = np.expand_dims(resized_mask_img, axis=0)
mask_array = mask_array.astype(np.float32) / 255.0

head_loaded_model = load_model('head_model.h5')
head_pred = head_loaded_model.predict(mask_array)
head_pred_class = 1 if head_pred[0][0] > 0.5 else 0

# life line prediction
mask_array = np.expand_dims(resized_life_img, axis=0)
mask_array = mask_array.astype(np.float32) / 255.0

# Initialize MobileNetV2 model with pretrained ImageNet weights
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base model

life_loaded_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 1 unit for binary classification
])
life_loaded_model.load_weights('life_model.h5')
life_pred = life_loaded_model.predict(mask_array)
life_pred_class = 1 if life_pred[0][0] > 0.5 else 0

# heart line prediction
mask_array = np.expand_dims(resized_heart_img, axis=0)
mask_array = mask_array.astype(np.float32) / 255.0

heart_loaded_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 1 unit for binary classification
])

heart_loaded_model.load_weights('heart_model.h5')
heart_pred = heart_loaded_model.predict(mask_array)
heart_pred_class = 1 if heart_pred[0][0] > 0.5 else 0


print("head:", head_pred_class)
print("life:", life_pred_class)
print("heart:", heart_pred_class)
