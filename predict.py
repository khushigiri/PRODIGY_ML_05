import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle

# Load model
model = tf.keras.models.load_model(
    "model/food_model.keras",
    compile=False
)

# Load labels
with open("model/class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

# Calorie mapping
calorie_dict = {
    "bread": 265,
    "dairy_product": 150,
    "dessert": 300,
    "egg": 155,
    "fried_food": 320,
    "meat": 250,
    "noodles_pasta": 158,
    "rice": 130,
    "seafood": 206,
    "soup": 80,
    "vegetable_fruit": 50
}

def predict_food(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    calories = calorie_dict.get(predicted_class, "Unknown")

    return predicted_class, calories


if __name__ == "__main__":
    img_path = input("Enter image path: ")
    food, calories = predict_food(img_path)

    print(f"\n🍽 Predicted Food: {food}")
    print(f"🔥 Estimated Calories: {calories} kcal")