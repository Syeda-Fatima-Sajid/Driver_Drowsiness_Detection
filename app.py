import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection")
st.write("Live Webcam Monitoring using EfficientNetB0")

# -----------------------------
# Hugging Face Model
# -----------------------------
REPO_ID = "Syeda-fatima-Shah/driver-drowsiness-detection"
MODEL_FILE = "efficientnetb0_best.h5"

# -----------------------------
# Load Model
# -----------------------------
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

    return tf.keras.models.load_model(model_path)

model = load_model()

# -----------------------------
# Preprocessing
# SAME AS NGROK VERSION
# -----------------------------
IMG_SIZE = (128, 128)

def preprocess(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img)

    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32)

    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

# -----------------------------
# Video Processor
# -----------------------------
class DrowsinessProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        try:

            arr = preprocess(img)

            prob = float(
                model.predict(arr, verbose=0)[0][0]
            )

            if prob > 0.5:
                label = "ALERT"
                color = (0, 255, 0)
            else:
                label = "DROWSY"
                color = (0, 0, 255)

            cv2.putText(
                img,
                f"{label} ({prob:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        except Exception as e:

            cv2.putText(
                img,
                f"ERROR",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            print(e)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# -----------------------------
# Webcam
# -----------------------------
webrtc_streamer(
    key="drowsiness",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.markdown("---")
st.caption("EfficientNetB0 | Streamlit WebRTC | Hugging Face")
