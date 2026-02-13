import os
import numpy as np
from PIL import Image

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


IMG_SIZE = 224
# use a repository-relative absolute path so Streamlit finds the model regardless of cwd
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mask_detector.h5")


@st.cache_resource
def load_mask_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please train the model first by running 'python src/train_mask_detector.py'."
        )
    return load_model(MODEL_PATH)


def predict_mask(image_pil, model):
    """
    Takes a PIL image, preprocesses it, and returns prediction probabilities.
    """
    image = image_pil.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = img_to_array(image)
    image_preprocessed = preprocess_input(image_array)
    image_batch = np.expand_dims(image_preprocessed, axis=0)

    (mask, without_mask) = model.predict(image_batch)[0]
    return mask, without_mask


def main():
    st.title("Face Mask Detection")
    st.write(
        "Upload an image and the model will detect whether the person is "
        "wearing a face mask or not."
    )

    try:
        model = load_mask_model()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            mask_prob, without_mask_prob = predict_mask(image, model)
            label = "Mask" if mask_prob > without_mask_prob else "No Mask"
            confidence = max(mask_prob, without_mask_prob) * 100

            st.subheader("Result")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")


if __name__ == "__main__":
    main()

