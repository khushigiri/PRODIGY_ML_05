import tensorflow as tf

# Load old model
model = tf.keras.models.load_model("model/food_model.h5")

# Save in new format
model.save("model/food_model.keras")

print("✅ Converted successfully!")