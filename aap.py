# --------------------------------------------
# Streamlit App for Hand Gesture Recognition
# --------------------------------------------

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------
# Load the trained model
# -----------------------
model = load_model("hand_gesture_cnn_model.h5")  # Must be in the same folder
IMG_SIZE = 64

# Define the gesture class labels (based on your dataset folders)
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']

# -----------------------
# Streamlit App Layout
# -----------------------
st.set_page_config(page_title="Hand Gesture Recognition", layout="centered")
st.title("🖐️ Hand Gesture Recognition using CNN")
st.markdown("Upload a hand gesture image to predict its class!")

# -----------------------
# Upload Image Section
# -----------------------
uploaded_file = st.file_uploader(" Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption=" Uploaded Image", use_column_width=True)

    # Ensure RGB format
    image = image.convert("RGB")

    # Convert to NumPy array and resize
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display prediction
    st.success(f"Predicted Gesture Class: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")
