# utils/eda.py


# =============================
# Exploratory Data Analysis
# =============================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from app.components.logger import get_logger  # ← NEW
logger = get_logger("eda")


def get_class_distribution(dataset, class_names=None):
    """
    Analyzes the distribution of classes within a dataset to understand class imbalance.

    Args:
        dataset (tf.data.Dataset): The dataset from which to calculate class distributions.

    Returns:
        dict: A dictionary mapping class names to their respective counts in the dataset.
    """
    logger.info("📋 Calculating class distribution.")
    # Initialize a dictionary to count occurrences
    count_dict = {k: 0 for k in range(len(class_names))}

    # Iterate over the dataset
    for images, labels in dataset:
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        for key, count in zip(unique, counts):
            count_dict[key] += count

    # Return a dictionary with class names and their counts
    return {class_names[k]: v for k, v in count_dict.items()}

def display_class_representatives_plt(data_dir, img_height, img_width):
    """
    Displays one representative image per class from a dataset directory.

    Parameters:
    -----------
    data_dir : str
        Path to the dataset directory where each subfolder is a class.
    img_height : int
        Target height of displayed images.
    img_width : int
        Target width of displayed images.
    """

    logger.info("📈 Plotting class distribution charts...")
    try:
        class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        class_folders.sort()

        image_paths = {
            folder: os.path.join(data_dir, folder, os.listdir(os.path.join(data_dir, folder))[0])
            for folder in class_folders
        }

        fig, axes = plt.subplots(1, len(class_folders), figsize=(len(class_folders) * 3, 3))
        for ax, (folder, image_path) in zip(axes, image_paths.items()):
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            ax.imshow(img_array.astype('uint8'))
            ax.set_title(folder)
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        logger.info("✅ Representative images displayed.")
    except Exception as e:
        logger.exception("❌ Error loading/displaying representative images.")
        raise RuntimeError(f"Failed to display class representatives: {e}")


def display_class_representatives_go(data_dir, img_height, img_width):
    """
    Displays one representative image per class using Plotly.

    Parameters:
    -----------
    data_dir : str
        Path to dataset directory where each subfolder is a class.
    img_height : int
        Image height for resizing.
    img_width : int
        Image width for resizing.

    Returns:
    --------
    plotly.graph_objects.Figure: A figure displaying images for each class.
    """

    logger.info("📷 Creating representative image grid with Plotly...")

    try:
        class_folders = sorted([
            f for f in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, f))
        ])

        images = []
        titles = []

        for class_name in class_folders:
            folder_path = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                continue

            img_path = os.path.join(folder_path, image_files[0])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img).astype(np.uint8)
            images.append(img_array)
            titles.append(class_name)

        num_classes = len(images)
        fig = make_subplots(rows=1, cols=num_classes, subplot_titles=titles)

        for i, img in enumerate(images):
            fig.add_trace(
                go.Image(z=img),
                row=1, col=i+1
            )

        # Adjust figure width to fill screen better
        fig_width = max(img_width * num_classes, 1100)  # dynamic width based on # of classes
        fig_height = img_height + 800  # cap height to avoid too tall images

        fig.update_layout(
            title_text="🖼️ Representative Images per Class",
            height=fig_height,
            width=fig_width,
            showlegend=False,
            margin=dict(t=60, l=20, r=20, b=20),
        )

        # Remove axes
        for i in range(1, num_classes + 1):
            fig.update_xaxes(showticklabels=False, row=1, col=i)
            fig.update_yaxes(showticklabels=False, row=1, col=i)

        logger.info("✅ Responsive Plotly image grid generated.")
        return fig

    except Exception as e:
        logger.exception("❌ Error displaying class representatives with Plotly.")
        raise RuntimeError(f"Failed to display class representatives: {e}")

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_distribution_go(distributions, dataset_names):
    """
    Visualizes class distributions using Plotly for interactive plotting.

    Args:
        distributions (list of dict): Class distributions per dataset.
        dataset_names (list of str): Dataset names corresponding to each distribution.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object with subplots.
    """
    logger.info("📊 Plotting dataset class distributions with Plotly...")

    try:
        num_datasets = len(distributions)
        fig = make_subplots(rows=1, cols=num_datasets, shared_yaxes=True,
                            subplot_titles=dataset_names)

        # Get sorted unique class labels
        all_classes = sorted(set(k for dist in distributions for k in dist.keys()))

        # Get a list of colors from the 'Purp' colormap
        color_scale = px.colors.sequential.Purp
        num_colors = len(all_classes)
        
        # Map each class to a color in the scale
        class_to_color = {
            cls: color_scale[int(i * (len(color_scale) - 1) / (num_colors - 1))]
            for i, cls in enumerate(all_classes)
        }

        for i, distribution in enumerate(distributions):
            x = list(distribution.keys())
            y = list(distribution.values())
            colors = [class_to_color.get(cls, "#CCCCCC") for cls in x]

            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    text=y,
                    textposition="outside",
                    textfont=dict(size=12),
                    insidetextanchor="start",
                    cliponaxis=False,
                    name=dataset_names[i],
                    marker_color=colors
                ),
                row=1, col=i + 1
            )
            fig.update_xaxes(title_text="Classes", tickangle=45, row=1, col=i + 1)
            fig.update_yaxes(title_text="Counts", row=1, col=1)

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',  # transparent plot background
            #paper_bgcolor='rgba(0,0,0,0)',  # transparent overall background
            title_text="📊 Class Distribution Across Datasets",
            showlegend=False,
            height=400,
            margin=dict(t=80, l=50, r=50, b=50)
        )
        return fig

    except Exception as e:
        logger.exception("❌ Failed to plot class distribution.")
        raise RuntimeError(f"Failed to plot class distributions: {e}")


def plot_distribution_plt(distributions, dataset_names):
    """
    Visualizes the distribution of classes across multiple datasets to compare balance and coverage.

    Args:
        distributions (list of dict): List of dictionaries representing class distributions for each dataset.
        dataset_names (list of str): Names of the datasets corresponding to each distribution.

    Displays:
        A series of bar plots showing class counts for each dataset provided.
    """

    logger.info("📈 Plotting class distribution charts...")

    try:
        num_datasets = len(distributions)
        fig, axes = plt.subplots(1, num_datasets, figsize=(5 * num_datasets, 5), sharey=True)
        fig.suptitle('Class Distribution in Datasets')

        if num_datasets == 1:
            axes = [axes]

        # Gather all class names
        all_classes = sorted(set(k for dist in distributions for k in dist.keys()))
        cmap = plt.cm.get_cmap("Purples", len(all_classes))  # Use built-in colormap
        class_to_color = {cls: cmap(i / (len(all_classes)-1)) for i, cls in enumerate(all_classes)}

        for ax, (name, distribution) in zip(axes, zip(dataset_names, distributions)):
            classes = list(distribution.keys())
            counts = list(distribution.values())
            colors = [class_to_color[cls] for cls in classes]

            bars = ax.bar(classes, counts, color=colors)
            ax.set_title(f'{name} Dataset')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Counts')
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha='right')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, int(height),
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        logger.info("✅ Class distribution plots generated.")
    except Exception as e:
        logger.exception("❌ Failed to plot class distribution.")
        raise RuntimeError(f"Failed to plot class distributions: {e}")


def plot_dataset_distributions(train_ds, val_ds, test_ds, img_height, img_width, class_names, save_dir=None, mode="plotly"):
    """
    Plots the class distributions of the training, validation, and test datasets.

    Parameters:
    ----------
    train_ds : tf.data.Dataset
        The training dataset.
    val_ds : tf.data.Dataset
        The validation dataset.
    test_ds : tf.data.Dataset
        The test dataset.
    img_height : int
        Image height (not used here, but passed for consistency/future use).
    img_width : int
        Image width (not used here, but passed for consistency/future use).
    """
    logger.info("📊 Plotting dataset class distributions...")
    try:
        train_dist = get_class_distribution(train_ds, class_names)
        val_dist = get_class_distribution(val_ds, class_names)
        test_dist = get_class_distribution(test_ds, class_names)

        if mode == "plotly" or mode == "go":
            return plot_distribution_go(distributions=[train_dist, val_dist, test_dist], 
                              dataset_names=["Train", "Validation", "Test"])
        elif mode == "matplotlib" or mode == "plt":
            plot_distribution_plt(distributions=[train_dist, val_dist, test_dist], 
                                  dataset_names=["Train", "Validation", "Test"])
        logger.info("✅ Dataset distributions plotted.")
    except Exception as e:
        logger.exception("❌ Error while plotting distributions.")
        raise RuntimeError(f"Failed to plot class distributions: {e}")
