# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np

# model = YOLO("yolov8n.pt")

# st.title("YOLO Object Detection")

# uploaded_file = st.file_uploader("Upload Image")

# if uploaded_file is not None:

#     image = Image.open(uploaded_file)
    
#     img_array = np.array(image)

#     results = model(img_array)

#     annotated = results[0].plot()

#     st.image(annotated, channels="BGR")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "YOLO Object Detection",
    page_icon  = "✈",
    layout     = "centered"
)

# ── Load model ONCE and cache it ──────────────────────────────
# Without @st.cache_resource, the model reloads on EVERY user
# interaction — making the app very slow.
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")   # downloads once, cached after

model = load_model()

# ── UI ────────────────────────────────────────────────────────
st.title("🔍 YOLO Object Detection")
st.write("Upload an image and the model will detect all objects in it.")

confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

# ── Run detection ─────────────────────────────────────────────
if uploaded_file is not None:

    image     = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        results   = model(img_array, conf=confidence)
        annotated = results[0].plot()                    # returns BGR numpy array

        # ✅ Fix: convert BGR → RGB before showing in Streamlit
        annotated_rgb = annotated[:, :, ::-1]

    # Count detections
    num_detections = len(results[0].boxes)

    st.success(f"✅ Found **{num_detections}** object(s)")
    st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

    # Show detection details in a table
    if num_detections > 0:
        st.subheader("Detection Details")
        data = []
        for box in results[0].boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            conf_score = float(box.conf[0])
            data.append({"Object": class_name, "Confidence": f"{conf_score:.2%}"})
        st.table(data)
