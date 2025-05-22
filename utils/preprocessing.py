import tensorflow as tf
from utils import eda  # Make sure `utils/__init__.py` exists

AUTOTUNE = tf.data.AUTOTUNE

def load_datasets(train_dir="assets/data/dataset2-master/dataset2-master/images/TRAIN", 
                  test_dir="assets/data/dataset2-master/dataset2-master/images/TEST", 
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

    return train_ds, val_ds, test_ds, class_names


def plot_dataset_distributions(train_ds, val_ds, test_ds, img_height, img_width, class_names, save_dir=None):
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
    train_dist = eda.get_class_distribution(train_ds, class_names)
    val_dist = eda.get_class_distribution(val_ds, class_names)
    test_dist = eda.get_class_distribution(test_ds, class_names)

    eda.plot_distribution(distributions=[train_dist, val_dist, test_dist], 
                          dataset_names=["Train", "Validation", "Test"])
