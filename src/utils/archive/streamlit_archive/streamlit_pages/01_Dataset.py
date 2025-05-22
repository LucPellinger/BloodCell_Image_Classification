import streamlit as st

import logging
import sys
import os
from pathlib import Path


# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.streamlit_app_utils.layout_utils import render_logo
from utils.preprocessing import load_datasets, plot_dataset_distributions
#from utils.eda import plot_distribution_streamlit, display_class_representatives_streamlit #checked

st.set_page_config(page_title="🧬 Image Dataset – Blood Cell Image Classification", layout="wide")

render_logo(title="🧬 Blood Cell Image Classification Dataset")



# /opt/conda/envs/image_classification

# === SETUP LOGGING === #
logging.basicConfig(
    level=logging.DEBUG, # change to DEBUG for more verbosity and INFO for more clear logging
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("🔧 Setting up logging...")

#st.header("🧬 Blood Cell Classification with Deep Learning")

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


# === CONFIGURATION === #
IMG_HEIGHT = 240
IMG_WIDTH = 320
NUM_CHANNELS = 3
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
log_dir="assets/tf_logs"

abs_path = Path(__file__).parent.parent.parent / "assets" / "app_images" / relative_path


TRAIN_DIR = Path(__file__).parent.parent.parent / "assets" / "data" / "dataset2-master" / "dataset2-master" / "images" / "TRAIN"
TEST_DIR = Path(__file__).parent.parent.parent / "assets" / "data" / "dataset2-master" / "dataset2-master" / "images" / "TEST" # "assets/data/dataset2-master/dataset2-master/images/TEST"

# === STEP 1: Load Data === #
logger.info("📦 Loading datasets...")
train_ds, val_ds, test_ds, class_names = load_datasets(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    batch_size=BATCH_SIZE
)

# === STEP 2: EDA (Distributions + Example Images) === #
logger.info("🔍 Visualizing class distributions...")
logger.info("Class names: %s", class_names)
plot_dataset_distributions(train_ds, val_ds, test_ds, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, class_names=class_names, mode="streamlit")

logger.info("🖼️ Displaying representative images...")
#display_class_representatives_streamlit(TRAIN_DIR, IMG_HEIGHT, IMG_WIDTH)

