import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import os

# Load trained model
model = tf.keras.models.load_model(
    "model/food_model.keras",
    compile=False
)

# Load class labels
with open("model/class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

# Calorie mapping
calorie_map = {
    "bread": 265,
    "dairy_product": 150,
    "dessert": 350,
    "egg": 155,
    "fried_food": 320,
    "meat": 250,
    "noodles_pasta": 300,
    "rice": 200,
    "sea_food": 220,
    "soup": 120,
    "vegetable_fruit": 80
}

# Get image path
img_path = input("Enter image path: ").strip()

if not os.path.exists(img_path):
    print("Image path not found!")
    exit()

# Preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array, verbose=0)
predicted_class = class_labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100
calories = calorie_map.get(predicted_class, "Unknown")

# Output
print("\nPrediction Result")
print(f"Food Item      : {predicted_class}")
print(f"Calories       : {calories} kcal")
print(f"Confidence     : {confidence:.2f}%")