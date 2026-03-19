import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from utils.calorie_map import calorie_map

# Load model + encoder
model = load_model("models/model.h5")

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

IMG_SIZE = 224

def predict_food(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)

    food_name = le.inverse_transform([class_idx])[0]
    calories = calorie_map.get(food_name, "Unknown")

    return food_name, calories