#app/app_launcher.py

import gradio as gr
from pages.home import home_page
from pages.dataset_overview import dataset_overview
from components.custom_css import custom_css
import os


# =========== Gradio App Launcher ===========
# This script launches the Gradio app with different tabs for various functionalities.
# The app is structured to allow for easy addition of new tabs in the future.
# The main function `launch_app` initializes the app and sets up the tabs.
# The app is designed to be modular, with each page being defined in its own module.
# Each page module should define a function that sets up the UI components for that page.
# The app is launched using the `launch_app` function, which is called when the script is run directly.

# =========== Setting the Theme ===========
# to see customize themes run the gradio_theme_builder.py script
# and follow the instructions in the terminal

theme = gr.themes.Monochrome(
    primary_hue="purple",
    secondary_hue="purple",
)

# =========== Launching the App ===========
# The main function that launches the Gradio app.
# It initializes the app with a theme and sets up the tabs.


# Inject the URI directly into your CSS
custom_css = custom_css



def launch_app():
    with gr.Blocks(theme=theme, css=custom_css) as app:
        with gr.Tabs():
            with gr.TabItem("Home"):
                home_page()

            with gr.TabItem("Dataset Overview"):
                dataset_overview()

            # Future:
            # with gr.TabItem("📈 Model Training"):
            #     ...
            # with gr.TabItem("🔬 Evaluation"):
            #     ...
            # with gr.TabItem("📷 Predict"):
            #     ...

    app.launch(show_error=True, 
               allowed_paths=[
                    "/home/lucpellinger/projects/BloodCell_Image_Classification/src/assets/data"
               ])

if __name__ == "__main__":
    launch_app()
