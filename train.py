import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import pickle
import os

from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 📁 Paths
# -------------------------------
TRAIN_DIR = "dataset/training"
VAL_DIR = "dataset/validation"
EVAL_DIR = "dataset/evaluation"
MODEL_DIR = "model"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 📊 Data Generators
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
eval_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

eval_data = eval_datagen.flow_from_directory(
    EVAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -------------------------------
# 🏷 Save Class Labels
# -------------------------------
class_labels = list(train_data.class_indices.keys())

with open(os.path.join(MODEL_DIR, "class_labels.pkl"), "wb") as f:
    pickle.dump(class_labels, f)

print("✅ Class labels saved!")

# -------------------------------
# 🧠 Load Base Model
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# -------------------------------
# 🏗 Build Model
# -------------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_labels), activation="softmax")
])

# -------------------------------
# ⚙ Compile Model
# -------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 🚀 Train Model
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------------------
# 💾 Save Model
# -------------------------------
model.save(os.path.join(MODEL_DIR, "food_model.keras"))
print("\n✅ Model saved successfully!")

# -------------------------------
# 📊 Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(eval_data)
print(f"\n📊 Evaluation Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# 📈 Predictions
# -------------------------------
predictions = model.predict(eval_data)
y_pred = np.argmax(predictions, axis=1)
y_true = eval_data.classes

labels = list(eval_data.class_indices.keys())

# -------------------------------
# 📋 Classification Report
# -------------------------------
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# -------------------------------
# 🔲 Confusion Matrix
# -------------------------------
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))