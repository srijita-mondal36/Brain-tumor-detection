import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
    <div style="text-align:center; padding:10px;">
        <h1 style="color:#00ffcc;">🧠 NeuroScan AI</h1>
        <h4 style="color:#cccccc;">Brain Tumor Detection & Classification System</h4>
        <hr style="border:1px solid #333;">
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = load_model("brain_tumor_model.h5")
labels = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# ----------------------------
# SIDEBAR INFO
# ----------------------------
st.sidebar.title("📌 About System")
st.sidebar.info(
    "This AI model analyzes MRI scans using Deep Learning (MobileNetV2) "
    "to classify brain tumor types."
)

st.sidebar.warning("⚠️ For educational purposes only")

# ----------------------------
# UPLOAD SECTION
# ----------------------------
st.subheader("📤 Upload MRI Scan")

file = st.file_uploader("Choose MRI Image", type=["jpg", "png", "jpeg"])

if file:

    # loading animation
    with st.spinner("🔍 Analyzing MRI scan..."):
        time.sleep(1.5)

    # read image
    image = Image.open(file)
    img = np.array(image)

    # layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

    # preprocessing
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # prediction
    prediction = model.predict(img_resized)
    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    with col2:
        st.markdown("### 🧾 Diagnosis Report")
        st.success(f"Prediction: {labels[index]}")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        if confidence < 60:
            st.warning("Low confidence — consider retesting image")

    # probability section
    st.subheader("📊 Probability Distribution")

    st.bar_chart({
        labels[0]: [float(prediction[0][0])],
        labels[1]: [float(prediction[0][1])],
        labels[2]: [float(prediction[0][2])],
        labels[3]: [float(prediction[0][3])]
    })

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""
    <hr>
    <div style="text-align:center; color:gray;">
        Developed using Deep Learning | Streamlit | TensorFlow
    </div>
""", unsafe_allow_html=True)