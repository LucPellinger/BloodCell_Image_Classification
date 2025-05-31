#app/pages/dataset_overview.py

import gradio as gr
import sys
import os

# Adds the parent of the current script's folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from components.shared import app_header, page_title, markdown_header
from utils.preprocessing import load_datasets
from utils.eda import display_class_representatives_go, plot_dataset_distributions
from app.components.logger import get_logger
from app.components.path_utils import get_project_root

logger = get_logger("dataset_page")

# Global cache for dataset (lazy load)
_cached_data = {
    "train_ds": None,
    "val_ds": None,
    "test_ds": None,
    "class_names": None,
}

def safe_load_datasets():
    if _cached_data["train_ds"] is None:
        try:
            logger.info("Loading datasets lazily...")
            train_ds, val_ds, test_ds, class_names = load_datasets()
            _cached_data.update({
                "train_ds": train_ds,
                "val_ds": val_ds,
                "test_ds": test_ds,
                "class_names": class_names
            })
        except Exception as e:
            logger.exception("Error while loading class distributions")
            raise RuntimeError(f"Failed to load datasets: {e}")
    return _cached_data["train_ds"], _cached_data["val_ds"], _cached_data["test_ds"], _cached_data["class_names"]

def dataset_overview():
    header_path = os.path.join(os.path.dirname(__file__), "..", "components", "markdown", "header_dataset.md")

    with gr.Column() as layout:
        markdown_header(header_path)
        #page_title("📊 Dataset Overview")

        with gr.Row():
            btn_plot_dist = gr.Button("📈 Show Class Distribution")
            btn_show_samples = gr.Button("🖼️ Show Class Examples")

        with gr.Column() as output_area:
            output_plot = gr.Plot(visible=False)
            output_gallery = gr.Gallery(label="Class Examples", visible=False, columns=4, height="auto")

        def plot_distributions_wrapper():
            try:
                logger.info("Plotting class distributions")
                train_ds, val_ds, test_ds, class_names = safe_load_datasets()
                fig = plot_dataset_distributions(train_ds, val_ds, test_ds, 240, 320, class_names)
                return gr.Plot(value=fig, visible=True), gr.Gallery(visible=False)
            except Exception as e:
                logger.exception("Failed to plot distributions")
                return gr.Plot(), f"❌ Error plotting distributions: {e}"

        def show_samples_wrapper():
            try:
                data_path = os.path.join(
                    get_project_root(), "assets", "data", "dataset2-master", "dataset2-master", "images", "TRAIN"
                )

                class_folders = sorted([
                    f for f in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, f))
                ])

                results = []
                for folder in class_folders:
                    folder_path = os.path.join(data_path, folder)
                    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if not image_files:
                        continue
                    img_path = os.path.join(folder_path, image_files[0])
                    results.append((img_path, folder))

                return gr.Plot(visible=False), gr.Gallery(value=results, visible=True)

            except Exception as e:
                logger.exception("Failed to show class samples")
                return gr.Plot(visible=False), gr.Gallery(visible=False)

#        def show_samples_wrapper():
#            try:
#                data_path = os.path.join(
#                    get_project_root(), "assets", "data", "dataset2-master", "dataset2-master", "images", "TRAIN"
#                )
#                fig = display_class_representatives_go(
#                    data_dir=data_path,
#                    img_height=100,
#                    img_width=100
#                )
#                return fig
#            except Exception as e:
#                logger.exception("Failed to show class samples")
#                return None

    btn_plot_dist.click(fn=plot_distributions_wrapper, outputs=[output_plot, output_gallery])
    btn_show_samples.click(fn=show_samples_wrapper, outputs=[output_plot, output_gallery])

    return layout
