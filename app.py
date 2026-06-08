import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import cv2
import os

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)

st.markdown("""
    <style>
        .alert-box {
            background-color: #1a3a1a;
            border: 2px solid #00c853;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            font-size: 22px;
            color: #00e676;
            font-weight: bold;
            margin: 10px 0;
        }
        .drowsy-box {
            background-color: #3a1a1a;
            border: 2px solid #ff1744;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            font-size: 22px;
            color: #ff5252;
            font-weight: bold;
            margin: 10px 0;
        }
        .no-eye-box {
            background-color: #2a2a1a;
            border: 2px solid #ffab00;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            font-size: 18px;
            color: #ffcc02;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Driver Drowsiness Detection")
st.markdown("**Auto Eye Detection — MobileNetV2 | EfficientNetB0**")
st.divider()

# ─── Hugging Face Config ───────────────────────────────────────
REPO_ID = "Syeda-fatima-Shah/driver-drowsiness-detection"

MODEL_FILES = {
    "MobileNetV2":    "mobilenetv2_best.h5",
    "EfficientNetB0": "efficientnetb0_best.h5",
}

# ─── Load Haar Cascades ────────────────────────────────────────
CASCADE_PATH  = cv2.data.haarcascades
face_cascade  = cv2.CascadeClassifier(CASCADE_PATH + "haarcascade_frontalface_default.xml")
eye_cascade   = cv2.CascadeClassifier(CASCADE_PATH + "haarcascade_eye.xml")

# ─── Model Loading ─────────────────────────────────────────────
@st.cache_resource
def load_keras_model(model_name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}"
    if not os.path.exists(path):
        with st.spinner(f"⬇️ Hugging Face se download ho raha hai..."):
            hf_hub_download(repo_id=REPO_ID, filename=model_name, local_dir="models")
    return tf.keras.models.load_model(path)

# ─── Eye Detection & Crop ──────────────────────────────────────
def detect_and_crop_eyes(pil_img):
    """
    Returns list of cropped eye PIL images detected in the image.
    Tries: face → eyes inside face. If no face, tries whole image for eyes.
    """
    img_np  = np.array(pil_img.convert("RGB"))
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    cropped = []

    # Try face first
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            roi_gray  = gray[fy:fy+fh, fx:fx+fw]
            roi_color = img_np[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                cropped.append(Image.fromarray(eye_img))
    else:
        # No face found — try directly on full image
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            eye_img = img_np[ey:ey+eh, ex:ex+ew]
            cropped.append(Image.fromarray(eye_img))

    return cropped

def draw_eye_boxes(pil_img):
    """Draw green boxes around detected eyes for visualization."""
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(img_np, (fx, fy), (fx+fw, fy+fh), (0, 255, 100), 2)
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img_np, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (0, 200, 255), 2)
    else:
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img_np, (ex, ey), (ex+ew, ey+eh), (0, 200, 255), 2)

    return Image.fromarray(img_np)

# ─── Prediction ────────────────────────────────────────────────
def preprocess_eye(eye_img: Image.Image):
    eye_img = eye_img.convert("RGB").resize((128, 128))
    arr = np.array(eye_img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, eye_img: Image.Image):
    arr  = preprocess_eye(eye_img)
    prob = model.predict(arr, verbose=0)[0][0]
    label = "✅ ALERT — Eyes Open" if prob > 0.5 else "😴 DROWSY — Eyes Closed!"
    conf  = prob if prob > 0.5 else 1 - prob
    return label, float(conf), float(prob)

# ─── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown(f"🤗 [Hugging Face Models](https://huggingface.co/{REPO_ID})")
st.sidebar.divider()

model_choice = st.sidebar.selectbox("Model chunein:", list(MODEL_FILES.keys()))
model_file   = MODEL_FILES[model_choice]
model        = load_keras_model(model_file)
st.sidebar.success(f"✅ {model_choice} ready!")

st.sidebar.divider()
st.sidebar.info("👁️ App automatically aankh dhundh kar detect karti hai — poora face upload kar sakte hain!")

# ─── Input Mode ────────────────────────────────────────────────
input_mode = st.radio("Input method chunein:", ["📁 Image Upload", "📷 Webcam"], horizontal=True)
st.divider()

# ─── Process Image ─────────────────────────────────────────────
def process_image(img: Image.Image):
    # Show annotated image
    annotated = draw_eye_boxes(img)
    st.image(annotated, caption="Detected Eyes (boxes)", use_container_width=True)

    # Detect eyes
    eyes = detect_and_crop_eyes(img)

    if len(eyes) == 0:
        st.markdown('<div class="no-eye-box">⚠️ Koi aankh detect nahi hui!<br><small>Seedhi, clear aur well-lit face photo try karein</small></div>', unsafe_allow_html=True)
        return

    st.markdown(f"**{len(eyes)} aankh(en) detect hui — predictions:**")

    # Predict each eye
    results = []
    cols = st.columns(min(len(eyes), 3))

    for i, eye_img in enumerate(eyes[:3]):  # max 3 eyes show
        label, conf, raw = predict(model, eye_img)
        results.append(label)
        with cols[i]:
            st.image(eye_img.resize((100, 100)), caption=f"Eye {i+1}")
            if "DROWSY" in label:
                st.error(f"😴 DROWSY\n{conf*100:.1f}%")
            else:
                st.success(f"✅ ALERT\n{conf*100:.1f}%")

    st.divider()

    # Overall verdict — if any eye is drowsy → DROWSY
    drowsy_count = sum(1 for r in results if "DROWSY" in r)
    if drowsy_count > 0:
        st.markdown('<div class="drowsy-box">😴 OVERALL: DROWSY DETECTED! — Please Rest!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box">✅ OVERALL: ALERT — Safe to Drive!</div>', unsafe_allow_html=True)

# ─── Image Upload ──────────────────────────────────────────────
if input_mode == "📁 Image Upload":
    uploaded = st.file_uploader("Face ya eye image upload karein", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        with st.spinner("👁️ Aankh dhundhi ja rahi hai..."):
            process_image(img)

# ─── Webcam ────────────────────────────────────────────────────
elif input_mode == "📷 Webcam":
    img_file = st.camera_input("Camera se photo lo")
    if img_file:
        img = Image.open(img_file)
        with st.spinner("👁️ Detecting..."):
            process_image(img)

st.divider()
st.caption("Developed with ❤️ | MRL Eye Dataset | TensorFlow + OpenCV + Streamlit")
