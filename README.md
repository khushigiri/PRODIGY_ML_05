# Food Calorie Estimator

An AI-powered application that detects food items from images and estimates their calorie content using deep learning.
This project focuses on **image classification and inference using TensorFlow** and runs via the terminal.

---

## Features

* Predict food items from input images
* Deep learning model using MobileNetV2
* Estimate calories based on predicted class
* Fast and lightweight terminal-based execution
* Displays prediction confidence score

---

## How It Works

1. User provides an image path through terminal
2. The trained model processes the image
3. Model predicts the food category
4. The predicted class is mapped to calorie values
5. Result is displayed with confidence score

---

## Tech Stack

* **TensorFlow (MobileNetV2)** – Image classification
* **Python** – Core logic & scripting
* **NumPy & Pillow** – Image preprocessing

---

## Limitations

* Model can only predict trained food categories
* Calorie values are approximate (static mapping)
* Accuracy depends on dataset quality and diversity
* Works only with clear food images

---

## Project Highlights

* Built using **Transfer Learning (MobileNetV2)**
* Trained on **10K+ food images across multiple categories**
* Achieved **~85% accuracy on evaluation dataset**
* End-to-end ML pipeline (training → prediction)

---