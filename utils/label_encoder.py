from sklearn.preprocessing import LabelEncoder
import pickle

def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Save encoder
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return y_train_enc, y_test_enc