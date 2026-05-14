import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("yolov8n.pt")

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:

    # image = Image.open(uploaded_file)
    try:
    image = Image.open(uploaded_file)
    except Exception:
    st.error("Please upload a valid image file (JPG, JPEG, PNG)")
    st.stop()

    img_array = np.array(image)

    results = model(img_array)

    annotated = results[0].plot()

    st.image(annotated, channels="BGR")

