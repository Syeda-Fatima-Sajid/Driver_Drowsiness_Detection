import streamlit as st
import tensorflow as tf

st.title("TensorFlow Test")

interpreter = tf.lite.Interpreter(
    model_path="drowsiness_model.tflite"
)

interpreter.allocate_tensors()

st.success("TFLite Loaded Successfully")
