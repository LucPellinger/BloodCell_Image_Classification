# utils/preprocessing.py


from app.components.logger import get_logger  # ← NEW
logger = get_logger("preprocessing")
# Add project root to PYTHONPATH
import os
#import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomBrightness, RandomTranslation

from app.components.path_utils import get_project_root

AUTOTUNE = tf.data.AUTOTUNE

def get_image_config():
    IMG_HEIGHT = 240
    IMG_WIDTH = 320
    IMG_CHANNELS = 3
    CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
    return IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CLASS_NAMES

def load_datasets(train_dir=os.path.join(get_project_root(), "assets", "data", "dataset2-master", "dataset2-master", "images", "TRAIN"),
                test_dir=os.path.join(get_project_root(), "assets", "data", "dataset2-master", "dataset2-master", "images", "TEST"),
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

def get_augmentation_layer(img_height, img_width):
    return keras.Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2),
        RandomContrast(0.2),
        RandomBrightness(0.2),
        RandomTranslation(height_factor=0.1, width_factor=0.1),
    ])
