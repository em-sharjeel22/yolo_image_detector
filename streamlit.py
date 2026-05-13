import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load model
model = YOLO("yolov8m.pt")

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:

    image = Image.open(uploaded_file)

    img_array = np.array(image)

    results = model(img_array)

    annotated = results[0].plot()

    st.image(annotated, channels="BGR")