import streamlit as st
import tensorflow as tf
import numpy as np
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗"
)

st.title("🚗 Real-Time Drowsiness Detection")

IMG_SIZE = (128, 128)

# -----------------------------
# LOAD TFLITE
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path="drowsiness_model.tflite"
    )

    interpreter.allocate_tensors()

    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )

interpreter, input_details, output_details = load_model()

# -----------------------------
# FRAME PROCESSOR
# -----------------------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.history = deque(maxlen=15)

    def predict(self, frame):

        frame = frame[:, :, ::-1]  # BGR -> RGB

        frame = tf.image.resize(
            frame,
            IMG_SIZE
        ).numpy()

        frame = frame.astype(np.float32) / 255.0

        frame = np.expand_dims(frame, axis=0)

        interpreter.set_tensor(
            input_details[0]["index"],
            frame
        )

        interpreter.invoke()

        prob = float(
            interpreter.get_tensor(
                output_details[0]["index"]
            )[0][0]
        )

        return prob

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        prob = self.predict(img)

        self.history.append(prob)

        avg_prob = np.mean(self.history)

        if avg_prob >= 0.85:
            label = "ALERT"
        elif avg_prob >= 0.70:
            label = "UNCERTAIN"
        else:
            label = "DROWSY"

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# -----------------------------
# START CAMERA
# -----------------------------
ctx = webrtc_streamer(
    key="drowsiness",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

st.info(
    "Camera ko seedha face karein. "
    "Prediction multiple frames ka average use karegi."
)
