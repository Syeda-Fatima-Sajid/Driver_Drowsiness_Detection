import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection (TFLite)")
st.markdown("### EfficientNetB0 - TensorFlow Lite")

IMG_SIZE = (128, 128)

# --------------------------------------------------
# LOAD TFLITE MODEL
# --------------------------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(
        model_path="drowsiness_model.tflite"
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


interpreter, input_details, output_details = load_tflite_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙ Settings")
st.sidebar.write("Threshold = 0.85")

# --------------------------------------------------
# PREPROCESS
# SAME AS YOUR NGROK VERSION
# RGB -> 128x128 -> /255
# --------------------------------------------------
def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32)

    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr


# --------------------------------------------------
# TFLITE PREDICTION
# --------------------------------------------------
def predict_image(img):

    arr = preprocess_image(img)

    interpreter.set_tensor(
        input_details[0]["index"],
        arr.astype(np.float32)
    )

    interpreter.invoke()

    prob = float(
        interpreter.get_tensor(
            output_details[0]["index"]
        )[0][0]
    )

    return prob
    


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------
# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------
def process_image(img):

    st.image(
        img,
        caption="Input Image",
        use_container_width=True
    )

    prob = predict_image(img)

    st.write("### Prediction Details")
    st.write(f"Raw Probability: {prob:.4f}")

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
elif mode == "📷 Camera":

    camera_file = st.camera_input(
        "Take a Picture"
    )

    if camera_file:

        image = Image.open(camera_file)

        process_image(image)

st.divider()

st.caption(
    "EfficientNetB0 • TensorFlow Lite • Streamlit"
)
