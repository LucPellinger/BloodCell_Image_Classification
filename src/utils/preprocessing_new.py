# utils/preprocessing.py

import os
import tensorflow as tf
from tensorflow import keras
from app.components.logger import get_logger
from app.components.path_utils import get_project_root

logger = get_logger("preprocessing")
AUTOTUNE = tf.data.AUTOTUNE

def get_image_config():
    IMG_HEIGHT = 240
    IMG_WIDTH = 320   # 4:3
    IMG_CHANNELS = 3
    CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
    return IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, CLASS_NAMES

def _opts():
    opt = tf.data.Options()
    opt.deterministic = False  # allow out-of-order for speed
    return opt

def _rebatch(ds, batch_size):
    # Give GPU static shapes for faster kernels
    return ds.unbatch().batch(batch_size, drop_remainder=True)

def load_datasets(
    train_dir=os.path.join(get_project_root(), "assets", "data", "dataset2-master", "dataset2-master", "images", "TRAIN"),
    test_dir=os.path.join(get_project_root(), "assets", "data", "dataset2-master", "dataset2-master", "images", "TEST"),
    img_height=240, img_width=320, batch_size=64    # try 64/96/128 on T4
):
    logger.info("🔄 Loading datasets from directories...")

    # 1) Disable internal shuffling; we will shuffle once after cache.
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,                          # <—
        interpolation="bilinear",
        crop_to_aspect_ratio=False              # keep whole image; no center-crop
    )
    class_names = raw_train_ds.class_names

    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,                          # <—
        interpolation="bilinear",
        crop_to_aspect_ratio=False
    )

    raw_test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,                          # <—
        interpolation="bilinear",
        crop_to_aspect_ratio=False
    )

    # 2) Cache (RAM or disk). RAM: fastest; Disk: safer on memory.
    cache_root = os.path.join(get_project_root(), "assets", "tf_cache")
    os.makedirs(cache_root, exist_ok=True)
    train_cache = os.path.join(cache_root, "train_cache")
    val_cache   = os.path.join(cache_root, "val_cache")
    test_cache  = os.path.join(cache_root, "test_cache")

    # 3) Build performant pipelines
    # Estimate reasonable shuffle buffer (in images, not batches)
    train_count = raw_train_ds.cardinality().numpy() * batch_size
    if train_count < 0:  # unknown cardinality
        train_count = 8000
    shuffle_buf = int(min(4000, max(1000, train_count)))

    train_ds = (
        raw_train_ds
        .cache(train_cache)          # to RAM: .cache() ; to disk: .cache(path)
        .with_options(_opts())
        .shuffle(shuffle_buf, reshuffle_each_iteration=True)
        .apply(lambda ds: _rebatch(ds, batch_size))     # static shapes
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        raw_val_ds
        .cache(val_cache)
        .with_options(_opts())
        .apply(lambda ds: _rebatch(ds, batch_size))
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        raw_test_ds
        .cache(test_cache)
        .with_options(_opts())
        .apply(lambda ds: _rebatch(ds, batch_size))
        .prefetch(AUTOTUNE)
    )

    logger.info("✅ Datasets loaded and preprocessed.")
    return train_ds, val_ds, test_ds, class_names

def get_augmentation_layer(img_height, img_width):
    # Keep augmentations as Keras layers so they can run on GPU.
    return keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomContrast(0.2),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomTranslation(0.1, 0.1),
    ])
