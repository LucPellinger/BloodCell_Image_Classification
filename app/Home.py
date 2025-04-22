import streamlit as st

def show():
    st.title("🧬 Blood Cell Classification with Deep Learning")
    st.markdown("---")

    st.image("https://raw.githubusercontent.com/datablist/sample-csv-files/main/images/blood-cells.jpg", caption="Red and White Blood Cells", use_column_width=True)

    st.markdown("""
    ## 🔍 Project Overview

    This application showcases the results of a Deep Learning project focused on **classifying different types of blood cells** using image data and pre-trained neural networks.

    The goals of this project include:
    - Exploring transfer learning models like **VGG16**, **ResNet101**, and others.
    - Optimizing architectures via **Grid Search** and **Bayesian Optimization**.
    - Evaluating models with advanced metrics and visualizations.
    - Providing an intuitive interface to classify new blood cell images.

    **Classes:**
    - 🩸 Eosinophil  
    - 🦠 Lymphocyte  
    - 🔬 Monocyte  
    - 🧪 Neutrophil

    ## 📚 Dataset

    The dataset comes from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells), containing over 12,000 labeled cell images categorized into 4 classes.

    ---
    """)

    st.info("Use the sidebar to explore model performance or upload your own image to classify!")

