import numpy as np
import pickle
from keras.models import load_model   
from utils.preprocess import preprocess_image
from utils.calorie_data import calories_dict

model = load_model("model/food_model.h5", compile=False)

with open("model/class_labels.pkl", "rb") as f:
    class_indices = pickle.load(f)

class_labels = list(class_indices.keys())

def predict_food(img_path):
    img_array = preprocess_image(img_path)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    food = class_labels[class_index]
    calories = calories_dict.get(food, "Unknown")

    return food, calories, prediction   