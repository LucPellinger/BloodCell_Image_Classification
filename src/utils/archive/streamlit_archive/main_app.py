import streamlit as st
from utils.streamlit_app_utils.layout_utils import render_logo


st.set_page_config(page_title="Blood Cell AI", layout="wide")

render_logo()

st.markdown("""
Use the sidebar to navigate between pages.
- **Dataset**: Overview of the project and dataset.
- **Model Demo**: Upload your own image or use sample images to see the model in action.
- **Model Optimization History**: Explore the performance of different models on the dataset.
- **Model Configuration Explorer**: Visualize the performance of the best models using various metrics.
""")