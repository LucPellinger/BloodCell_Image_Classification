#app/components/shared.py


import gradio as gr
import os
import sys
import os
from app.components.path_utils import get_project_root

# Adds the parent of the current script's folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def app_header():
    project_root = get_project_root()
    logo_path = os.path.join(project_root, "assets", "app_images", "App_Logo.png")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(logo_path, label="BloodCell Classifier", show_label=False, height=100)
        with gr.Column(scale=5):
            gr.Markdown("## 🧬 Blood Cell Image Classifier\nA visual interface to explore models trained on blood cell images.")

def page_title(title: str):
    gr.Markdown(f"### {title}")


def markdown_header(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    gr.Markdown(content)