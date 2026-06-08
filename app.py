import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection")
st.markdown("### EfficientNetB0 Driver Drowsiness Classifier")
st.divider()

# ─────────────────────────────────────────────
# Hugging Face Model
# ─────────────────────────────────────────────
REPO_ID = "Syeda-fatima-Shah/driver-drowsiness-detection"

MODEL_FILES = {
    "EfficientNetB0": "efficientnetb0_best.h5",
}

IMG_SIZE = (128, 128)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_keras_model(model_name):
    os.makedirs("models", exist_ok=True)

    model_path = f"models/{model_name}"

    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading model from Hugging Face..."):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=model_name,
                local_dir="models"
            )

    return tf.keras.models.load_model(model_path)

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(model, img):

    arr = preprocess_image(img)

    prob = float(model.predict(arr, verbose=0)[0][0])

    label = "ALERT" if prob > 0.5 else "DROWSY"

    confidence = prob if prob > 0.5 else 1 - prob

    return label, confidence, prob

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

model = load_keras_model(MODEL_FILES[model_choice])

st.sidebar.success("✅ Model Loaded")

# ─────────────────────────────────────────────
# Process Image
# ─────────────────────────────────────────────
def process_image(img):

    st.image(
        img,
        caption="Input Image",
        use_container_width=True
    )

    with st.spinner("🔍 Analyzing Driver State..."):

        label, confidence, prob = predict(model, img)

    st.write(f"Raw Probability: {prob:.4f}")

    st.divider()

    if label == "DROWSY":

        st.error(
            f"😴 DROWSY DETECTED\n\nConfidence: {confidence*100:.2f}%"
        )

    else:

        st.success(
            f"✅ ALERT — Safe to Drive!\n\nConfidence: {confidence*100:.2f}%"
        )

# ─────────────────────────────────────────────
# Input Mode
# ─────────────────────────────────────────────
input_mode = st.radio(
    "Choose Input Method",
    ["📁 Image Upload", "📷 Webcam"],
    horizontal=True
)

st.divider()

# ─────────────────────────────────────────────
# Image Upload
# ─────────────────────────────────────────────
if input_mode == "📁 Image Upload":

    uploaded = st.file_uploader(
        "Upload Driver Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        img = Image.open(uploaded)

        process_image(img)

# ─────────────────────────────────────────────
# Webcam
# ─────────────────────────────────────────────
elif input_mode == "📷 Webcam":

    img_file = st.camera_input("Capture Driver Image")

    if img_file:

        img = Image.open(img_file)

        process_image(img)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()

st.caption(
    "Developed with ❤️ | EfficientNetB0 | TensorFlow | Streamlit"
)
