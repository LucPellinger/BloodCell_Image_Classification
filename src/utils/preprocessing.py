# utils/preprocessing.py


from app.components.logger import get_logger  # ← NEW
logger = get_logger("preprocessing")
# Add project root to PYTHONPATH
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import eda  # Make sure `utils/__init__.py` exists
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomBrightness, RandomTranslation

AUTOTUNE = tf.data.AUTOTUNE

def load_datasets(train_dir="src/assets/data/dataset2-master/dataset2-master/images/TRAIN", 
                  test_dir="src/assets/data/dataset2-master/dataset2-master/images/TEST", 
                  img_height=240, img_width=320, batch_size=32):
    """
    Load and preprocess image datasets for training, validation, and testing.

    Parameters:
    ----------
    train_dir : str
        Path to the training dataset directory.
    test_dir : str
        Path to the test dataset directory.
    img_height : int
        Height of the images.
    img_width : int
        Width of the images.
    batch_size : int
        Batch size for the datasets.

    Returns:
    -------
    train_ds : tf.data.Dataset
        Preprocessed training dataset.
    val_ds : tf.data.Dataset
        Preprocessed validation dataset.
    test_ds : tf.data.Dataset
        Preprocessed test dataset.
    """
    logger.info("🔄 Loading datasets from directories...")
    try:
        # Load training dataset
        raw_train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='int'
        )

        class_names = raw_train_ds.class_names  # ✅ Capture before transformation

        # Load validation dataset
        raw_val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='int'
        )

        # Load test dataset
        raw_test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='int'
        )

        AUTOTUNE = tf.data.AUTOTUNE

        # Performance boost: cache, shuffle (for training), and prefetch
        train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = raw_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        logger.info("✅ Datasets loaded and preprocessed.")
        return train_ds, val_ds, test_ds, class_names
    except Exception as e:
        logger.exception("❌ Failed to load datasets.")
        raise RuntimeError(f"Failed to load datasets: {e}")


def plot_dataset_distributions(train_ds, val_ds, test_ds, img_height, img_width, class_names, save_dir=None, mode=None):
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
        train_dist = eda.get_class_distribution(train_ds, class_names)
        val_dist = eda.get_class_distribution(val_ds, class_names)
        test_dist = eda.get_class_distribution(test_ds, class_names)

        if mode == None:
            eda.plot_distribution(distributions=[train_dist, val_dist, test_dist], 
                              dataset_names=["Train", "Validation", "Test"])
        logger.info("✅ Dataset distributions plotted.")
    except Exception as e:
        logger.exception("❌ Error while plotting distributions.")
        raise RuntimeError(f"Failed to plot class distributions: {e}")

    #elif mode == "st" or "streamlit":
    #    eda.plot_distribution_streamlit(distributions=[train_dist, val_dist, test_dist], 
    #                      dataset_names=["Train", "Validation", "Test"])


def get_augmentation_layer(img_height, img_width):
    return keras.Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2),
        RandomContrast(0.2),
        RandomBrightness(0.2),
        RandomTranslation(height_factor=0.1, width_factor=0.1),
    ])
