import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "sign_cnn_model.h5"
DATASET_PATH = "dataset"
IMG_SIZE = 64

labels = sorted(os.listdir(DATASET_PATH))

# =========================
# REBUILD MODEL (FIXES ERROR)
# =========================
def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model(len(labels))
model.load_weights(MODEL_PATH)

print("✅ Model loaded successfully")
print("Classes:", labels)

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# =========================
# REALTIME LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                hand_img = hand_img / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                prediction = model.predict(hand_img)
                class_id = np.argmax(prediction)
                label = labels[class_id]

                cv2.putText(frame, label, (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Alphabet Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
