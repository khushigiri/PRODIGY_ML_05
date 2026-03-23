import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from utils.calorie_map import calorie_map
import os

# -------------------------------
# Load model and label encoder
# -------------------------------
model = load_model("models/best_model.keras")

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------------------
# Constants
# -------------------------------
IMG_SIZE = 160
CONFIDENCE_THRESHOLD = 0.7

# -------------------------------
# Prediction Function
# -------------------------------
def predict_food(image_path):
    try:
        # Fix path (handles Windows backslashes)
        image_path = image_path.strip().replace("\\", "/")

        # Check file exists
        if not os.path.exists(image_path):
            return "Error", "Image not found", 0

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return "Error", "Invalid image", 0

        # Preprocess
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img, verbose=0)
        confidence = float(np.max(pred))
        class_idx = int(np.argmax(pred))

        # Decode label
        food_name = le.inverse_transform([class_idx])[0]

        # Low confidence handling
        if confidence < CONFIDENCE_THRESHOLD:
            return "Not sure", "Try another image", confidence

        # Get calories
        calories = calorie_map.get(food_name, "Unknown")

        return food_name, calories, confidence

    except Exception as e:
        return "Error", str(e), 0

# -------------------------------
# CLI Interface
# -------------------------------
if __name__ == "__main__":
    print("\n🍔 Food Calorie Predictor")
    print("-" * 35)

    while True:
        image_path = input("\nEnter image path (or 'exit'): ")

        if image_path.lower() == "exit":
            print("👋 Exiting...")
            break

        food, calories, confidence = predict_food(image_path)

        print("\n📊 Result:")
        print(f"Prediction : {food}")
        print(f"Calories   : {calories} kcal")
        print(f"Confidence : {confidence:.2f}")

        print("-" * 35)