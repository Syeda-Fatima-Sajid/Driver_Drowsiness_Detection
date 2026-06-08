import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

# ─── Page Config ─────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection")
st.markdown("### EfficientNetB0 Drowsiness Detector")
st.divider()

# ─── Hugging Face Model Config ─────────────────
REPO_ID = "Syeda-fatima-Shah/driver-drowsiness-detection"
MODEL_FILE = "efficientnetb0_best.h5"

# ─── Load Model ───────────────────────────────
@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)

    model_path = f"models/{MODEL_FILE}"

    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading model from Hugging Face..."):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILE,
                local_dir="models"
            )

    return tf.keras.models.load_model(model_path)

model = load_model()

st.success("✅ EfficientNetB0 Loaded Successfully")

# ─── Preprocessing (Same as ngrok) ────────────
IMG_SIZE = (128, 128)

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr

# ─── Prediction ───────────────────────────────
def predict_image(img):

    arr = preprocess_image(img)

    prob = float(model.predict(arr, verbose=0)[0][0])

    if prob > 0.5:
        label = "✅ ALERT"
        confidence = prob
    else:
        label = "😴 DROWSY"
        confidence = 1 - prob

    return label, confidence, prob

# ─── Upload / Webcam ──────────────────────────
input_mode = st.radio(
    "Choose Input Method",
    ["📁 Upload Image", "📷 Webcam"],
    horizontal=True
)

st.divider()

if input_mode == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        with st.spinner("Predicting..."):

            label, conf, prob = predict_image(img)

        st.divider()

        st.write(f"### Raw Probability: `{prob:.4f}`")

        if "DROWSY" in label:
            st.error(
                f"😴 DROWSY DETECTED\n\nConfidence: {conf*100:.2f}%"
            )
        else:
            st.success(
                f"✅ ALERT\n\nConfidence: {conf*100:.2f}%"
            )

elif input_mode == "📷 Webcam":

    camera_image = st.camera_input("Take a Picture")

    if camera_image:

        img = Image.open(camera_image)

        st.image(
            img,
            caption="Captured Image",
            use_container_width=True
        )

        with st.spinner("Predicting..."):

            label, conf, prob = predict_image(img)

        st.divider()

        st.write(f"### Raw Probability: `{prob:.4f}`")

        if "DROWSY" in label:
            st.error(
                f"😴 DROWSY DETECTED\n\nConfidence: {conf*100:.2f}%"
            )
        else:
            st.success(
                f"✅ ALERT\n\nConfidence: {conf*100:.2f}%"
            )

st.divider()
st.caption("Developed with ❤️ using TensorFlow + Streamlit")
