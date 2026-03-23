import os
import json
import cv2
import numpy as np

IMG_SIZE = 160
MAX_PER_CLASS = 200   # 🔥 limit per class (NOT global)

def load_data(json_path, image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = []
    labels = []

    for label, items in data.items():

        count = 0   # ✅ reset per class

        for item in items:

            if count >= MAX_PER_CLASS:
                break

            img_path = os.path.join(image_dir, item + ".jpg")

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label)

            count += 1

    print("Loaded images:", len(images))

    X = np.array(images, dtype=np.float32)
    y = np.array(labels)

    print("X shape:", X.shape)

    return X, y