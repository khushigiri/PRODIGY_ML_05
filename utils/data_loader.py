import os
import json
import cv2
import numpy as np

IMG_SIZE = 224

def load_data(json_path, image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = []
    labels = []

    for item in data:
        label = item.split('/')[0]  
        img_path = os.path.join(image_dir, item + ".jpg")

        if not os.path.exists(img_path):
            print("Missing:", img_path)
            continue

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)