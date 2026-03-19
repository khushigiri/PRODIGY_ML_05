from flask import Flask, request, jsonify
import os
from inference.predict import predict_food

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Food Calorie API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    food, calories = predict_food(path)

    return jsonify({
        "food": food,
        "calories": calories
    })

if __name__ == "__main__":
    app.run(debug=True)