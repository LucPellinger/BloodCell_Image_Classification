import gradio as gr
import os
import sys
import os

# Adds the parent of the current script's folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def app_header():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("src/assets/app_images/App_Logo.png", label="BloodCell Classifier", show_label=False, height=100)
        with gr.Column(scale=5):
            gr.Markdown("## 🧬 Blood Cell Image Classifier\nA visual interface to explore models trained on blood cell images.")

def page_title(title: str):
    gr.Markdown(f"### {title}")
