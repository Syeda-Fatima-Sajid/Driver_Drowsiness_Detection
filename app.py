import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection")
st.markdown("### EfficientNetB0 Model")

# --------------------------------------------------
# HUGGING FACE
# --------------------------------------------------
REPO_ID = "Syeda-fatima-Shah/driver-drowsiness-detection"
MODEL_FILE = "efficientnetb0_best.h5"

IMG_SIZE = (128, 128)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():

    os.makedirs("models", exist_ok=True)

    model_path = f"models/{MODEL_FILE}"

    if not os.path.exists(model_path):

        with st.spinner("Downloading model from Hugging Face..."):

            hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILE,
                local_dir="models"
            )

    model = tf.keras.models.load_model(model_path)

    return model

model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙ Settings")

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.01
)

st.sidebar.write(f"Current Threshold: {threshold}")

# --------------------------------------------------
# PREPROCESS
# SAME AS NGROK
# --------------------------------------------------
def preprocess_image(img):

    img = img.convert("RGB")

    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32)

    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

# --------------------------------------------------
# PREDICT
# --------------------------------------------------
def predict_image(img):

    arr = preprocess_image(img)

    prob = float(model.predict(arr, verbose=0)[0][0])

    if prob >= threshold:
        label = "✅ ALERT"
    else:
        label = "😴 DROWSY"

    return label, prob

# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------
def process_image(img):

    st.image(
        img,
        caption="Input Image",
        use_container_width=True
    )

    label, prob = predict_image(img)

    st.write("### Prediction Details")
    st.write(f"Raw Probability: **{prob:.4f}**")
    st.write(f"Threshold: **{threshold:.2f}**")

    if label == "✅ ALERT":
        st.success(
            f"{label}\n\nProbability: {prob*100:.2f}%"
        )
    else:
        st.error(
            f"{label}\n\nProbability: {prob*100:.2f}%"
        )

# --------------------------------------------------
# INPUT MODE
# --------------------------------------------------
mode = st.radio(
    "Choose Input Method",
    ["📁 Image Upload", "📷 Camera"],
    horizontal=True
)

st.divider()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
if mode == "📁 Image Upload":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        process_image(image)

# --------------------------------------------------
# CAMERA
# --------------------------------------------------
if mode == "📷 Camera":

    camera_file = st.camera_input(
        "Take a Picture"
    )

    if camera_file:

        image = Image.open(camera_file)

        process_image(image)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "EfficientNetB0 | Streamlit | Hugging Face"
)
