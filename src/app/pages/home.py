#app/pages/home.py

import gradio as gr
import sys
import os

# Adds the parent of the current script's folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.components.shared import app_header, markdown_header
from app.components.logger import get_logger  # ← NEW
logger = get_logger("home")

def home_page():
    header_path = os.path.join(os.path.dirname(__file__), "..", "components", "markdown", "header_home.md")
    logger.info("📥 Rendering Home Page.")
    with gr.Column() as layout:
        #app_header()
        markdown_header(header_path)

        gr.Markdown("""
        ### Welcome to the Blood Cell Image Classifier Web App

        This app allows you to:
        - 📊 Explore dataset statistics
        - 🔍 Visualize training performance
        - 🧠 Test different CNN & pretrained models
        - 📸 Upload an image and predict the cell type
        - 🧪 Evaluate models with confusion matrix, ROC, PR curves

        **Tech Stack:** TensorFlow · Keras · Optuna · Gradio

        """)
        
    return layout
