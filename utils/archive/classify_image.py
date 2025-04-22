import streamlit as st
import numpy as np
from PIL import Image
import os

from app import inference


# Predefined class labels
CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

# Placeholder images (you’ll replace with actual image paths)
SAMPLE_IMAGES = {
    "Sample 1": "assets/data/dataset-master/dataset-master/JPEGImages/BloodImage_00000.jpg",
    "Sample 2": "assets/data/dataset-master/dataset-master/JPEGImages/BloodImage_00410.jpg",
    "Sample 3": "assets/data/dataset-master/dataset-master/JPEGImages/BloodImage_00200.jpg"
}

def fake_predict(image_array):
    """Simulated softmax prediction"""
    return np.random.dirichlet(np.ones(len(CLASS_NAMES)), size=1)[0]


def show():
    st.title("🖼️ Classify a Blood Cell Image")
    st.markdown("---")

    st.info("Choose a sample image or upload your own to classify.")

    # Option selector
    option = st.radio("Select input type:", ["📁 Upload Image", "🖼️ Use Sample Image"])

    image = None
    if option == "📁 Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        selected_sample = st.selectbox("Choose a sample image", list(SAMPLE_IMAGES.keys()))
        sample_path = SAMPLE_IMAGES[selected_sample]
        if os.path.exists(sample_path):
            image = Image.open(sample_path).convert("RGB")
        else:
            st.warning("Sample image not found — replace this with your actual image files.")

    # Display + Predict
    if image:
        st.image(image, caption="Input Image", use_column_width=True)
        st.markdown("### 🔄 Preprocessing...")
        img_array = inference.preprocess_image(image)

        try:
            st.markdown("### 🤖 Predicting...")
            model = inference.load_model(MODEL_PATH)
            probs = inference.predict_image(model, img_array)

            for i, prob in enumerate(probs):
                st.write(f"**{inference.CLASS_NAMES[i]}**: {prob:.2%}")

            predicted_class = inference.CLASS_NAMES[probs.argmax()]
            st.success(f"🧠 Predicted Class: **{predicted_class}**")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
