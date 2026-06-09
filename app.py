import streamlit as st
import tensorflow as tf
import numpy as np
import av
import cv2
from collections import deque
from threading import Lock
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗"
)

st.title("🚗 Real-Time Drowsiness Detection")

IMG_SIZE = (128, 128)

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():

    interpreter = tf.lite.Interpreter(
        model_path="drowsiness_model_fixed.tflite"
    )

    interpreter.allocate_tensors()

    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )

interpreter, input_details, output_details = load_model()

# Thread safety
interpreter_lock = Lock()

# Shared state
if "probability" not in st.session_state:
    st.session_state.probability = 0.0

if "status" not in st.session_state:
    st.session_state.status = "Waiting..."

# -----------------------------
# VIDEO PROCESSOR
# -----------------------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.history = deque(maxlen=15)

    def predict(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb, IMG_SIZE)

        rgb = rgb.astype(np.float32) / 255.0

        rgb = np.expand_dims(rgb, axis=0)

        with interpreter_lock:

            interpreter.set_tensor(
                input_details[0]["index"],
                rgb
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

        avg_prob = float(np.mean(self.history))

        # Save for UI
        st.session_state.probability = avg_prob

        if avg_prob >= 0.85:
            st.session_state.status = "✅ ALERT"

        elif avg_prob >= 0.70:
            st.session_state.status = "⚠️ UNCERTAIN"

        else:
            st.session_state.status = "😴 DROWSY"

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# -----------------------------
# WEBRTC
# -----------------------------
webrtc_streamer(
    key="drowsiness",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

st.divider()

st.subheader("Prediction Details")

st.write(
    f"Raw Probability: "
    f"{st.session_state.probability:.4f}"
)

prob = st.session_state.probability

if prob >= 0.85:

    st.success(
        f"✅ ALERT\n\nProbability: {prob*100:.2f}%"
    )

elif prob >= 0.70:

    st.warning(
        f"⚠️ UNCERTAIN\n\nProbability: {prob*100:.2f}%"
    )

else:

    st.error(
        f"😴 DROWSY\n\nProbability: {prob*100:.2f}%"
    )

st.info(
    "Camera ke samne seedha baith kar test karein."
)
