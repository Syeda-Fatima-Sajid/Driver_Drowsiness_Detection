import streamlit as st
import tensorflow as tf
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗"
)

st.title("🚗 Real-Time Drowsiness Detection")

IMG_SIZE = (128, 128)

# -----------------------------
# LOAD MODEL
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

# -----------------------------
# PROCESSOR
# -----------------------------
class VideoProcessor(VideoProcessorBase):

    def predict(self, frame):

        rgb = frame[:, :, ::-1]

        rgb = tf.image.resize(
            rgb,
            IMG_SIZE
        ).numpy()

        rgb = rgb.astype(np.float32)

        rgb = rgb / 255.0

        rgb = np.expand_dims(rgb, axis=0)

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

        img = frame.to_ndarray(
            format="bgr24"
        )

        prob = self.predict(img)

        print("Probability:", prob)

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

st.info(
    "Camera ke samne baith kar test karein."
)
