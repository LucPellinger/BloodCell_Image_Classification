# =============================
# Exploratory Data Analysis
# =============================
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def get_class_distribution_streamlit(dataset, class_names=None):

  """
  Analyzes the distribution of classes within a dataset to understand class imbalance.

  Args:
      dataset (tf.data.Dataset): The dataset from which to calculate class distributions.

  Returns:
      dict: A dictionary mapping class names to their respective counts in the dataset.
  """

  # Initialize a dictionary to count occurrences
  count_dict = {k: 0 for k in range(len(class_names))}

  # Iterate over the dataset
  for images, labels in dataset:
      unique, counts = np.unique(labels.numpy(), return_counts=True)
      for key, count in zip(unique, counts):
          count_dict[key] += count

  # Return a dictionary with class names and their counts
  return {class_names[k]: v for k, v in count_dict.items()}

def plot_distribution_streamlit(distributions, dataset_names):
    """
    Visualizes the distribution of classes across multiple datasets using Plotly.

    Args:
        distributions (list of dict): List of dictionaries representing class distributions for each dataset.
        dataset_names (list of str): Names of the datasets corresponding to each distribution.

    Displays:
        Interactive bar plots for class counts in each dataset.
    """
    # Gather all class names across all distributions
    all_classes = sorted(set().union(*[d.keys() for d in distributions]))

    # Set up the subplot structure
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=len(distributions),
        shared_yaxes=True,
        subplot_titles=dataset_names
    )

    # Assign consistent colors to each class
    colors = px.colors.qualitative.Plotly
    color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(all_classes)}

    for idx, (distribution, name) in enumerate(zip(distributions, dataset_names), start=1):
        for cls in all_classes:
            count = distribution.get(cls, 0)
            fig.add_trace(
                go.Bar(
                    x=[cls],
                    y=[count],
                    name=cls if idx == 1 else None,  # show legend only once
                    marker_color=color_map[cls],
                    text=str(count),
                    textposition="outside"
                ),
                row=1,
                col=idx
            )

        fig.update_xaxes(title_text="Classes", row=1, col=idx)
        if idx == 1:
            fig.update_yaxes(title_text="Counts", row=1, col=idx)

    fig.update_layout(
        title_text="Class Distribution in Datasets",
        showlegend=True,
        barmode="group",
        height=500,
        width=300 * len(distributions),
        margin=dict(t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def display_class_representatives_streamlit(data_dir, img_height, img_width):
    """
    Displays one representative image per class from a dataset directory using Streamlit.

    Parameters:
    -----------
    data_dir : str
        Path to the dataset directory where each subfolder is a class.
    img_height : int
        Target height of displayed images.
    img_width : int
        Target width of displayed images.
    """
    try:
        import tensorflow as tf  # Lazy import for Streamlit stability

        class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        class_folders.sort()

        if not class_folders:
            st.warning("No class folders found in the specified directory.")
            return

        image_paths = {
            folder: os.path.join(data_dir, folder, os.listdir(os.path.join(data_dir, folder))[0])
            for folder in class_folders
        }

        fig, axes = plt.subplots(1, len(class_folders), figsize=(len(class_folders) * 3, 3))
        if len(class_folders) == 1:
            axes = [axes]  # Ensure axes is iterable if only one class

        for ax, (folder, image_path) in zip(axes, image_paths.items()):
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            ax.imshow(img_array.astype('uint8'))
            ax.set_title(folder)
            ax.axis('off')

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while displaying images: {e}")