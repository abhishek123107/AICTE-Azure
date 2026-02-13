"""Create and save a tiny Keras demo model to `models/mask_detector.h5`.
This model is only for UI/demo purposes (predicts deterministic probabilities).
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, Dense
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mask_detector.h5")

# simple model that mirrors expected input shape (224,224,3)
model = Sequential([
    InputLayer(input_shape=(224, 224, 3)),
    GlobalAveragePooling2D(),
    Dense(64, activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# set deterministic weights so predictions are stable
for layer in model.layers:
    for w in layer.weights:
        w.assign(tf.random.stateless_uniform(w.shape, seed=(1, 2)))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"Demo model saved to: {MODEL_PATH}")
