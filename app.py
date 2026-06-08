import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
    <style>
        .main { background-color: #0f1117; }
        .stTitle { color: #ffffff; }
        .alert-box {
            background-color: #1a3a1a;
            border: 2px solid #00c853;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            font-size: 24px;
            color: #00e676;
            font-weight: bold;
        }
        .drowsy-box {
            background-color: #3a1a1a;
            border: 2px solid #ff1744;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            font-size: 24px;
            color: #ff5252;
            font-weight: bold;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Driver Drowsiness Detection")
st.markdown("**MobileNetV2 | EfficientNetB0 | TFLite**")
st.divider()

# ─── Hugging Face Config ───────────────────────────────────────
REPO_ID = "Syeda-fatima-Shah/driver-drowsiness-detection"

MODEL_FILES = {
    "MobileNetV2 (.h5)":   "mobilenetv2_best.h5",
    "EfficientNetB0 (.h5)": "efficientnetb0_best.h5",
    "TFLite (Fast ⚡)":     "drowsiness_model.tflite",
}

# ─── Model Loading ─────────────────────────────────────────────
@st.cache_resource
def load_keras_model(model_name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}"
    if not os.path.exists(path):
        with st.spinner(f"⬇️ Hugging Face se download ho raha hai... (ek baar hoga)"):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=model_name,
                local_dir="models"
            )
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_tflite_model(model_name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}"
    if not os.path.exists(path):
        with st.spinner("⬇️ TFLite model download ho raha hai..."):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=model_name,
                local_dir="models"
            )
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

# ─── Prediction Functions ──────────────────────────────────────
def preprocess_image(img: Image.Image, size=(128, 128)):
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_keras(model, img_array):
    prob = model.predict(img_array, verbose=0)[0][0]
    label = "✅ ALERT — Eyes Open" if prob > 0.5 else "😴 DROWSY — Wake Up!"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, float(confidence), float(prob)

def predict_tflite(interpreter, img_array):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = "✅ ALERT — Eyes Open" if prob > 0.5 else "😴 DROWSY — Wake Up!"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, float(confidence), float(prob)

# ─── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown(f"**Model Source:** [Hugging Face 🤗]({f'https://huggingface.co/{REPO_ID}'})")
st.sidebar.divider()

model_choice = st.sidebar.selectbox(
    "Model chunein:",
    list(MODEL_FILES.keys())
)

model_file = MODEL_FILES[model_choice]
is_tflite  = model_file.endswith(".tflite")

# ─── Load Selected Model ───────────────────────────────────────
if is_tflite:
    interpreter = load_tflite_model(model_file)
    st.sidebar.success("✅ TFLite ready!")
else:
    model = load_keras_model(model_file)
    st.sidebar.success(f"✅ {model_choice} ready!")

st.sidebar.divider()
st.sidebar.info("💡 TFLite sabse fast hai aur kam RAM use karta hai!")

# ─── Input Mode ────────────────────────────────────────────────
input_mode = st.radio(
    "Input method chunein:",
    ["📁 Image Upload", "📷 Webcam"],
    horizontal=True
)

st.divider()

# ─── Run Prediction ────────────────────────────────────────────
def run_prediction(img: Image.Image):
    arr = preprocess_image(img)
    if is_tflite:
        return predict_tflite(interpreter, arr)
    else:
        return predict_keras(model, arr)

def show_result(label, conf, raw):
    if "DROWSY" in label:
        st.markdown(f'<div class="drowsy-box">{label}<br><small>Confidence: {conf*100:.1f}%</small></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box">{label}<br><small>Confidence: {conf*100:.1f}%</small></div>',
                    unsafe_allow_html=True)
    st.metric("Raw Probability (Alert)", f"{raw:.4f}")

# ─── Image Upload ──────────────────────────────────────────────
if input_mode == "📁 Image Upload":
    uploaded = st.file_uploader(
        "Eye/face image upload karein",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=300)
        with st.spinner("🔍 Detecting..."):
            label, conf, raw = run_prediction(img)
        show_result(label, conf, raw)

# ─── Webcam ────────────────────────────────────────────────────
elif input_mode == "📷 Webcam":
    img_file = st.camera_input("Camera se photo lo")
    if img_file:
        img = Image.open(img_file)
        with st.spinner("🔍 Detecting..."):
            label, conf, raw = run_prediction(img)
        show_result(label, conf, raw)

st.divider()
st.caption("Developed with ❤️ | MRL Eye Dataset | TensorFlow + Streamlit")
