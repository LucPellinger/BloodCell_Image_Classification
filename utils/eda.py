# =============================
# Exploratory Data Analysis
# =============================

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_class_distribution(dataset, class_names=None):

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

def plot_distribution(distributions, dataset_names):

  """
  Visualizes the distribution of classes across multiple datasets to compare balance and coverage.

  Args:
      distributions (list of dict): List of dictionaries representing class distributions for each dataset.
      dataset_names (list of str): Names of the datasets corresponding to each distribution.

  Displays:
      A series of bar plots showing class counts for each dataset provided.
  """

  # Create subplots
  fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
  fig.suptitle('Class Distribution in Datasets')

  # Gather all class names and assign colors
  all_classes = set()
  for distribution in distributions:
      all_classes.update(distribution.keys())
  all_classes = sorted(all_classes)  # Sort to maintain order
  colors = plt.cm.get_cmap('tab20', len(all_classes))  # Get a colormap with enough colors

  # Create a color dictionary
  color_dict = {cls: colors(i) for i, cls in enumerate(all_classes)}

  # Plot each distribution
  for ax, (name, distribution) in zip(axes, zip(dataset_names, distributions)):
      classes = list(distribution.keys())
      counts = list(distribution.values())
      class_colors = [color_dict[cls] for cls in classes]  # Get the colors for each class in this distribution

      bars = ax.bar(classes, counts, color=class_colors)
      ax.set_title(f'{name} Dataset')
      ax.set_xlabel('Classes')
      ax.set_ylabel('Counts')
      ax.set_xticklabels(classes, rotation=45, ha='right')

      # Add annotations to each bar
      for bar in bars:
          yval = bar.get_height()
          ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
                  verticalalignment='bottom',  # vertical alignment
                  horizontalalignment='center',  # horizontal alignment
                  color='black', fontsize=8, rotation=0)

  # Show plot
  plt.tight_layout()
  plt.show()

def display_class_representatives(data_dir, img_height, img_width):
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