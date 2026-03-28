from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import os

app = Flask(__name__)

# -------------------------------
# Load trained model
# -------------------------------
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

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_food(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    calories = calorie_map[predicted_class]

    return predicted_class, calories


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    calories = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            prediction, calories = predict_food(filepath)
            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        calories=calories,
        image_path=image_path
    )




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)