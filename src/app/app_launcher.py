import gradio as gr
from pages.home import home_page
from pages.dataset_overview import dataset_overview

def launch_app():
    with gr.Blocks() as app:
        with gr.Tabs():
            with gr.TabItem("🏠 Home"):
                home_page().render()

            with gr.TabItem("📊 Dataset Overview"):
                dataset_overview().render()

            # Future:
            # with gr.TabItem("📈 Model Training"):
            #     ...
            # with gr.TabItem("🔬 Evaluation"):
            #     ...
            # with gr.TabItem("📷 Predict"):
            #     ...

    app.launch()

if __name__ == "__main__":
    launch_app()
