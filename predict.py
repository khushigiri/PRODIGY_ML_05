import numpy as np
import pickle
import tflite_runtime.interpreter as tflite
from utils.preprocess import preprocess_image
from utils.calorie_data import calories_dict

interpreter = tflite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("model/class_labels.pkl", "rb") as f:
    class_indices = pickle.load(f)

class_labels = list(class_indices.keys())

def predict_food(img_path):
    img_array = preprocess_image(img_path).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(prediction)

    food = class_labels[class_index]
    calories = calories_dict.get(food, "Unknown")

    return food, calories, prediction