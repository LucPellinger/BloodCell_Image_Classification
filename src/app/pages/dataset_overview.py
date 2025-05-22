import gradio as gr
import sys
import os

# Adds the parent of the current script's folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from components.shared import app_header, page_title
from utils.preprocessing import load_datasets, plot_dataset_distributions
from utils.eda import display_class_representatives
from components.logger import get_logger
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
    with gr.Blocks() as demo:
        app_header()
        page_title("📊 Dataset Overview")

        with gr.Row():
            btn_plot_dist = gr.Button("📈 Show Class Distribution")
            btn_show_samples = gr.Button("🖼️ Show Class Examples")

        output_plot = gr.Plot()
        output_gallery = gr.HTML()

        def plot_distributions_wrapper():
            try:
                logger.info("Plotting class distributions")
                train_ds, val_ds, test_ds, class_names = safe_load_datasets()
                plot_dataset_distributions(train_ds, val_ds, test_ds, 240, 320, class_names)
                return gr.Plot.update(), ""  # dummy return; matplotlib shows externally
            except Exception as e:
                logger.exception("Failed to plot distributions")
                return gr.Plot.update(), f"❌ Error plotting distributions: {e}"

        def show_samples_wrapper():
            try:
                display_class_representatives(
                    data_dir="assets/data/dataset2-master/dataset2-master/images/TRAIN",
                    img_height=100,
                    img_width=100
                )
                return ""
            except Exception as e:
                return f"❌ Error displaying class examples: {e}"

        btn_plot_dist.click(fn=plot_distributions_wrapper, outputs=[output_plot, output_gallery])
        btn_show_samples.click(fn=show_samples_wrapper, outputs=output_gallery)

    return demo
