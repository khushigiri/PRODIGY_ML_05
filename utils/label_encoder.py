from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os

def encode_labels(y_train, y_test):
    le = LabelEncoder()

    # 🔥 Combine train + test labels (FIX for unseen labels)
    all_labels = np.concatenate((y_train, y_test))

    # Fit on ALL labels
    le.fit(all_labels)

    # Transform separately
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # ✅ Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Save encoder
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Label encoding done")
    print("Classes:", len(le.classes_))

    return y_train_enc, y_test_enc

    