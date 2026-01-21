import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATASET_PATH = "dataset"
IMG_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 32

# =========================
# LOAD DATA
# =========================
X = []
y = []

labels = sorted(os.listdir(DATASET_PATH))
label_map = {label: i for i, label in enumerate(labels)}

print("Classes:", labels)

for label in labels:
    folder_path = os.path.join(DATASET_PATH, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_map[label])

X = np.array(X) / 255.0
y = np.array(y)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# CNN MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# TRAIN
# =========================
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    batch_size=BATCH_SIZE
)

# =========================
# SAVE MODEL
# =========================
model.save("sign_cnn_model.h5")
print("✅ Model saved as sign_cnn_model.h5")
