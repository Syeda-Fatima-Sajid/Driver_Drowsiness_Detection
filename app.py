import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import cv2
import mediapipe as mp
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
st.markdown("**MediaPipe Eye Detection — MobileNetV2 | EfficientNetB0**")
st.divider()

# ─── Hugging Face Config ───────────────────────────────────────
REPO_ID = "Syeda-fatima-Shah/driver-drowsiness-detection"

MODEL_FILES = {
    "MobileNetV2":    "mobilenetv2_best.h5",
    "EfficientNetB0": "efficientnetb0_best.h5",
}

# ─── MediaPipe Setup ───────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

# MediaPipe Face Mesh landmark indices for eyes
# Left eye  outer region landmarks
LEFT_EYE_LANDMARKS  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Right eye outer region landmarks
RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ─── Model Loading ─────────────────────────────────────────────
@st.cache_resource
def load_keras_model(model_name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}"
    if not os.path.exists(path):
        with st.spinner(f"⬇️ Hugging Face se download ho raha hai..."):
            hf_hub_download(repo_id=REPO_ID, filename=model_name, local_dir="models")
    return tf.keras.models.load_model(path)

# ─── MediaPipe Eye Crop ────────────────────────────────────────
def get_eye_crop(img_np, landmarks, eye_indices, padding=0.3):
    """Crop eye region using MediaPipe landmarks with padding."""
    h, w = img_np.shape[:2]

    xs = [landmarks[i].x * w for i in eye_indices]
    ys = [landmarks[i].y * h for i in eye_indices]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # Add padding around eye
    pw = int((x_max - x_min) * padding)
    ph = int((y_max - y_min) * padding)

    x_min = max(0, x_min - pw)
    x_max = min(w, x_max + pw)
    y_min = max(0, y_min - ph * 2)  # more padding on top
    y_max = min(h, y_max + ph * 2)

    crop = img_np[y_min:y_max, x_min:x_max]
    return crop, (x_min, y_min, x_max, y_max)

def detect_eyes_mediapipe(pil_img):
    """
    Uses MediaPipe Face Mesh to detect and crop both eyes.
    Works for both open AND closed eyes.
    Returns: list of (eye_pil_image, bbox), annotated_pil_image
    """
    img_np  = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    annotated = img_np.copy()
    eyes = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4
    ) as face_mesh:

        results = face_mesh.process(img_np)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark

                # Left eye
                left_crop, left_bbox = get_eye_crop(img_np, lm, LEFT_EYE_LANDMARKS)
                if left_crop.size > 0:
                    eyes.append(("Left Eye", Image.fromarray(left_crop)))
                    x1, y1, x2, y2 = left_bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(annotated, "L", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

                # Right eye
                right_crop, right_bbox = get_eye_crop(img_np, lm, RIGHT_EYE_LANDMARKS)
                if right_crop.size > 0:
                    eyes.append(("Right Eye", Image.fromarray(right_crop)))
                    x1, y1, x2, y2 = right_bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 150), 2)
                    cv2.putText(annotated, "R", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,150), 2)

    return eyes, Image.fromarray(annotated)

# ─── Prediction ────────────────────────────────────────────────
def preprocess_eye(eye_img: Image.Image):
    eye_img = eye_img.convert("RGB").resize((128, 128))
    arr = np.array(eye_img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, eye_img: Image.Image):
    arr  = preprocess_eye(eye_img)
    prob = model.predict(arr, verbose=0)[0][0]
    label = "✅ ALERT" if prob > 0.5 else "😴 DROWSY"
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
st.sidebar.info("👁️ MediaPipe se band aankhen bhi detect hoti hain!")

# ─── Input Mode ────────────────────────────────────────────────
input_mode = st.radio("Input method chunein:", ["📁 Image Upload", "📷 Webcam"], horizontal=True)
st.divider()

# ─── Process Image ─────────────────────────────────────────────
def process_image(img: Image.Image):
    with st.spinner("👁️ MediaPipe se aankh dhundhi ja rahi hai..."):
        eyes, annotated = detect_eyes_mediapipe(img)

    st.image(annotated, caption="Detected Eyes", use_container_width=True)

    if len(eyes) == 0:
        st.markdown('''<div class="no-eye-box">
            ⚠️ Koi aankh detect nahi hui!<br>
            <small>Seedha camera ki taraf dekho, acha lighting hona chahiye</small>
        </div>''', unsafe_allow_html=True)
        return

    st.markdown(f"**{len(eyes)} aankh(en) detect hui:**")

    results = []
    cols = st.columns(len(eyes))

    for i, (eye_name, eye_img) in enumerate(eyes):
        label, conf, raw = predict(model, eye_img)
        results.append(label)
        with cols[i]:
            st.image(eye_img.resize((120, 80)), caption=eye_name)
            if "DROWSY" in label:
                st.error(f"😴 DROWSY\n{conf*100:.1f}%")
            else:
                st.success(f"✅ ALERT\n{conf*100:.1f}%")

    st.divider()

    # Overall verdict
    drowsy_count = sum(1 for r in results if "DROWSY" in r)
    if drowsy_count >= 1:
        st.markdown('<div class="drowsy-box">😴 DROWSY DETECTED! — Kirpya aaraam karein!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box">✅ ALERT — Safe to Drive!</div>', unsafe_allow_html=True)

# ─── Image Upload ──────────────────────────────────────────────
if input_mode == "📁 Image Upload":
    uploaded = st.file_uploader("Face image upload karein", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        process_image(img)

# ─── Webcam ────────────────────────────────────────────────────
elif input_mode == "📷 Webcam":
    img_file = st.camera_input("Camera se photo lo")
    if img_file:
        img = Image.open(img_file)
        process_image(img)

st.divider()
st.caption("Developed with ❤️ | MRL Eye Dataset | TensorFlow + MediaPipe + Streamlit")
