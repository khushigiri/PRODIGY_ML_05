import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import os

# Paths
train_dir = "dataset/training"
val_dir = "dataset/validation"
test_dir = "dataset/evaluation"

# Data Generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Model (Better than basic CNN)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(train_data.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/food_model.h5")

# Save class labels
with open("model/class_labels.pkl", "wb") as f:
    pickle.dump(train_data.class_indices, f)

# Evaluate
loss, acc = model.evaluate(test_data)
print("Test Accuracy:", acc)