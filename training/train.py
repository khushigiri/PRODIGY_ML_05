# =========================
# IMPORTS
# =========================
from utils.data_loader import load_data
from utils.label_encoder import encode_labels

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import numpy as np
import os
import pickle

# =========================
# FIX oneDNN
# =========================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =========================
# PATHS
# =========================
TRAIN_JSON = "dataset/food-101/meta/train.json"
IMAGE_DIR = "dataset/food-101/images"

# =========================
# LOAD DATA
# =========================
X, y = load_data(TRAIN_JSON, IMAGE_DIR)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# PREPROCESS
# =========================
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# =========================
# ENCODE LABELS
# =========================
y_train, y_test = encode_labels(y_train, y_test)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

num_classes = len(le.classes_)

print("Classes:", num_classes)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# =========================
# BASE MODEL
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3)
)

# 🔥 Phase 1: Freeze ALL layers
base_model.trainable = False

# =========================
# STRONG DATA AUGMENTATION 🔥
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# =========================
# MODEL
# =========================
model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),   # 🔥 improves stability
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# =========================
# COMPILE (PHASE 1)
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("models/best_model.keras", save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=3, factor=0.3)  # 🔥 VERY IMPORTANT
]

# =========================
# TRAIN PHASE 1
# =========================
print("\n🔵 Phase 1: Training top layers...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weights
)

# =========================
# PHASE 2: FINE-TUNING 🔥
# =========================
print("\n🟢 Phase 2: Fine-tuning...\n")

# Unfreeze last 40 layers
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00001),  # 🔥 lower LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weights
)

# =========================
# EVALUATE
# =========================
loss, acc = model.evaluate(X_test, y_test)
print(f"\n🔥 Final Test Accuracy: {acc*100:.2f}%")

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
model.save("models/model.keras")

print("Model saved successfully!")