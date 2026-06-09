import streamlit as st
import os

st.title("File Check")

st.write("Current directory:")
st.write(os.getcwd())

st.write("Files:")
st.write(os.listdir("."))
