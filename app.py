import streamlit as st
import tensorflow as tf
import os

st.title("TFLite Debug")

st.write("File exists:", os.path.exists("drowsiness_model.tflite"))
st.write("File size (MB):", os.path.getsize("drowsiness_model.tflite") / (1024 * 1024))

try:
    interpreter = tf.lite.Interpreter(
        model_path="drowsiness_model.tflite"
    )

    interpreter.allocate_tensors()

    st.success("✅ TFLite loaded successfully!")

except Exception as e:
    st.error(str(e))
