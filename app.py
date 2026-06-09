import streamlit as st
import tensorflow as tf

st.title("TFLite Test")

interpreter = tf.lite.Interpreter(
    model_path="drowsiness_model_fixed.tflite"
)

interpreter.allocate_tensors()

st.success("✅ Model Loaded Successfully")
