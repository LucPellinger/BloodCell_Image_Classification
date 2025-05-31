# utils/inference.py


import tensorflow as tf
import numpy as np
from PIL import Image
import os

from app.components.path_utils import get_project_root
from utils.preprocessing import get_image_config

#import streamlit as st

# Constants
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CLASS_NAMES = get_image_config()
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)  #  (240, 320)
NUM_CHANNELS = IMG_CHANNELS # 3
CLASS_NAMES = CLASS_NAMES # ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

#@st.cache_resource
def load_model(model_name: str):
    """Load a Keras model from file, cached for performance"""
    model_path = os.path.join(get_project_root(), "assets", "models", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)

def preprocess_image(image: Image.Image):
    """Preprocess a PIL image to model-ready format"""
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)
    return img_array

def predict_image(model, image_array):
    """Run inference and return prediction probabilities"""
    probs = model.predict(image_array)[0]  # Remove batch dimension
    class_idx = np.argmax(probs)
    return class_idx, probs
