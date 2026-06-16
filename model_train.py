import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# -----------------------
# PATH
# -----------------------
BASE_PATH = r"D:\Brain tumor detection\archive"

TRAIN_PATH = os.path.join(BASE_PATH, "Training")
TEST_PATH  = os.path.join(BASE_PATH, "Testing")

# -----------------------
# SETTINGS
# -----------------------
img_size = 224
categories = ['glioma', 'meningioma', 'pituitary', 'notumor']

# -----------------------
# LOAD DATA
# -----------------------
def load_data(data_path):
    data, labels = [], []

    for category in categories:
        path = os.path.join(data_path, category)

        if not os.path.exists(path):
            print("Missing:", path)
            continue

        label = categories.index(category)

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)

                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0

                data.append(img)
                labels.append(label)

            except:
                pass

    return np.array(data), np.array(labels)

# -----------------------
# LOAD DATA
# -----------------------
print("Loading data...")
X_train, y_train = load_data(TRAIN_PATH)
X_test, y_test = load_data(TEST_PATH)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

if len(X_train) == 0:
    raise Exception("Dataset not loaded properly!")

# -----------------------
# MODEL
# -----------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

# -----------------------
# COMPILE
# -----------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# TRAIN
# -----------------------
print("Training started...")

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# -----------------------
# SAVE MODEL
# -----------------------
model.save("brain_tumor_model.h5")

print("Model saved successfully!")