import streamlit as st
from PIL import Image
import os
from utils import inference  # update this based on your folder structure
import numpy as np

# Path to your trained model
MODEL_PATH = "models/best_model.h5"

# Class names (update if needed)
CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

# Sample images (adjust paths for your local project)
SAMPLE_IMAGES = {
    "Sample 1": "data/test_images/BloodImage_00000.jpg",
    "Sample 2": "data/test_images/BloodImage_00410.jpg",
    "Sample 3": "data/test_images/BloodImage_00200.jpg"
}

def show():
    st.title("🧪 Model Demo – Blood Cell Image Classification")
    st.markdown("---")

    st.info("Select a sample image or upload your own for inference.")

    image = None
    input_type = st.radio("Choose input type:", ["📁 Upload Image", "🖼️ Use Sample Image"])

    if input_type == "📁 Upload Image":
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        selected_sample = st.selectbox("Choose a sample", list(SAMPLE_IMAGES.keys()))
        sample_path = SAMPLE_IMAGES[selected_sample]
        if os.path.exists(sample_path):
            image = Image.open(sample_path).convert("RGB")
        else:
            st.warning("Sample image not found. Check your file paths.")

    if image:
        st.image(image, caption="Selected Image", use_column_width=True)
        st.markdown("### 🔄 Preprocessing...")

        try:
            model = inference.load_model(MODEL_PATH)
            img_array = inference.preprocess_image(image)
            st.markdown("### 🤖 Predicting...")

            class_idx, probs = inference.predict(model, img_array)

            st.markdown("### 🔍 Class Probabilities:")
            for i, prob in enumerate(probs[0]):
                st.write(f"**{CLASS_NAMES[i]}**: {prob:.2%}")

            st.success(f"🧠 Predicted Class: **{CLASS_NAMES[class_idx]}**")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
