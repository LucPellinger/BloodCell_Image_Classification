import tensorflow as tf
import logging
import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import load_datasets, plot_dataset_distributions
from utils.base_model import BaseCNNModel
from utils.eda import display_class_representatives #checked

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
logger.info("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# === CONFIGURATION === #
IMG_HEIGHT = 240
IMG_WIDTH = 320
NUM_CHANNELS = 3
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
log_dir="assets/tf_logs"

TRAIN_DIR = "assets/data/dataset2-master/dataset2-master/images/TRAIN"
TEST_DIR = "assets/data/dataset2-master/dataset2-master/images/TEST"
MODEL_NAME = "baseline_cnn_model"

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
plot_dataset_distributions(train_ds, val_ds, test_ds, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, class_names=class_names)

logger.info("🖼️ Displaying representative images...")
display_class_representatives(TRAIN_DIR, IMG_HEIGHT, IMG_WIDTH)

# === STEP 3: Build Model === #
logger.info("🧠 Initializing baseline CNN model...")
model = BaseCNNModel(MODEL_NAME, NUM_CLASSES, (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
model.build()
model.compile()

# === STEP 4: Train Model === #
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ TensorFlow is using GPU with memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU found. Using CPU instead.")

logger.info("🚀 Starting training...")
model.train(
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=EPOCHS,
    early_stopping=True,
    plot=True,
    tensorboard_logdir=log_dir
)

# === STEP 5: Evaluate Model === #
logger.info("📊 Evaluating on test set...")
model.evaluate(test_ds)
model.evaluate_with_report(test_ds, class_names=class_names)

# === STEP 6: Save Model === #
logger.info("💾 Saving model and training history...")
model.save()

logger.info("✅ All steps completed successfully!")