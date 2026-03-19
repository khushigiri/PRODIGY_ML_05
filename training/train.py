from utils.data_loader import load_data
from utils.label_encoder import encode_labels
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# Paths
TRAIN_JSON = "dataset/food-101/meta/train.json"
TEST_JSON = "dataset/food-101/meta/test.json"
IMAGE_DIR = "dataset/food-101/images"

# Load data
X_train, y_train = load_data(TRAIN_JSON, IMAGE_DIR)
X_test, y_test = load_data(TEST_JSON, IMAGE_DIR)

# Encode labels
y_train, y_test = encode_labels(y_train, y_test)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(set(y_train)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          batch_size=32)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/model.h5")

print("Model saved!")