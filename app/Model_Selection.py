import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import os

def show():
    st.title("🔍 Model Selection & Explorer")
    st.markdown("---")

    # === Part 1: Model Overview Table === #
    st.subheader("📊 Model Comparison Summary")

    csv_path = "data/model_selection_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        st.dataframe(df, use_container_width=True)

        metric = st.selectbox("Sort models by:", df.columns[1:], index=0)
        top_n = st.slider("Top N models to display", 1, len(df), min(5, len(df)))

        sorted_df = df.sort_values(by=metric, ascending=False).head(top_n)

        fig = px.bar(sorted_df, x="Model", y=metric, color="Model", text=metric)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("📁 model_selection_results.csv not found. Skipping overview table.")

    st.markdown("---")

    # === Part 2: Model Detail Explorer === #
    st.subheader("🔬 Model Detail Explorer")

    model_options = ["VGG16", "ResNet101", "MobileNetV2"]
    selected_model = st.selectbox("Choose a model to inspect:", model_options)

    # Dynamic hyperparameter info
    st.subheader("🛠️ Hyperparameters")
    model_configs = {
        "VGG16": {
            "Dense Layers": 2,
            "Neurons per Layer": 256,
            "Dropout Rate": 0.5,
            "Normalization": True,
            "L2 Regularization": 0.01,
            "Optimizer": "Adam (lr=0.001)"
        },
        "ResNet101": {
            "Dense Layers": 1,
            "Neurons per Layer": 128,
            "Dropout Rate": None,
            "Normalization": False,
            "L2 Regularization": None,
            "Optimizer": "Adam (lr=0.0001)"
        },
        "MobileNetV2": {
            "Dense Layers": 1,
            "Neurons per Layer": 64,
            "Dropout Rate": 0.3,
            "Normalization": True,
            "L2 Regularization": 0.005,
            "Optimizer": "Adam (lr=0.0005)"
        }
    }

    if selected_model in model_configs:
        st.json(model_configs[selected_model])
    else:
        st.info("Model config not available yet.")

    # Simulated training curve
    st.subheader("📈 Training Curves")
    curve_type = st.radio("Select curve to view", ["Accuracy", "Loss"])

    epochs = np.arange(1, 11)
    np.random.seed(hash(selected_model) % 123456)  # Keep results consistent per model
    acc = np.linspace(0.7, 0.95, num=10) + np.random.normal(0, 0.01, size=10)
    loss = np.linspace(1.0, 0.3, num=10) + np.random.normal(0, 0.02, size=10)

    fig, ax = plt.subplots()
    if curve_type == "Accuracy":
        ax.plot(epochs, acc, label="Train Accuracy")
        ax.set_ylabel("Accuracy")
    else:
        ax.plot(epochs, loss, label="Train Loss", color="red")
        ax.set_ylabel("Loss")

    ax.set_xlabel("Epoch")
    ax.set_title(f"{selected_model} – {curve_type} over Epochs")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.success("📌 Tip: You can explore training and test performance in more depth on the next page!")
