import logging
import sys
import os
import datetime

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import load_datasets
from utils.eda import display_class_representatives_plt, plot_dataset_distributions
from utils.base_model import BaseModel
from utils.optimization import BayesianOptimizer

# === SETUP LOGGING === #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("transfer_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import tensorflow as tf
import gc
from tensorflow.keras import backend as K
import psutil
import subprocess

# === GPU Setup === #
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("✅ TensorFlow using GPU with memory growth enabled.")
    except RuntimeError as e:
        logger.warning("GPU config error: %s", e)
else:
    logger.warning("⚠️ No GPU found. Using CPU instead.")

# === Memory Logging Helpers === #
def log_system_memory_usage():
    mem = psutil.virtual_memory()
    logger.info(f"🧠 System Memory: {mem.percent}% used ({mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB)")

def log_gpu_memory_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        for idx, line in enumerate(result.stdout.strip().split('\n')):
            total, used, free = map(int, line.split(','))
            logger.info(f"🖥️ GPU {idx}: {used}MB used / {total}MB total ({free}MB free)")
    except Exception as e:
        logger.warning(f"Unable to query GPU memory usage: {e}")

logger.info("🔧 Logging initialized.")
logger.info("Num GPUs Available: %s", len(tf.config.list_physical_devices('GPU')))

# === Configuration === #
IMG_HEIGHT = 240
IMG_WIDTH = 320
NUM_CHANNELS = 3
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 10
TRIALS = 30
CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
TRAIN_DIR = "assets/data/dataset2-master/dataset2-master/images/TRAIN"
TEST_DIR = "assets/data/dataset2-master/dataset2-master/images/TEST"

models = [
    "VGG16", 
    #"VGG19", 
    #"EfficientNetB0", 
    #"DenseNet121",
    #"NASNetMobile", 
    #"MobileNetV2", 
    #"ResNet101", 
    #"ResNet50", 
    #"InceptionV3"
]

for model_arch in models:
    logger.info(f"🔍 Starting optimization for model: {model_arch}")

    log_system_memory_usage()
    log_gpu_memory_usage()

    # === Load and Explore Dataset === #
    logger.info("📦 Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE
    )

    # === Optimization === #
    log_dir = os.path.join("assets/tf_logs", f"{model_arch}_opt", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    optimizer = BayesianOptimizer(model_class=BaseModel)

    best_trial = optimizer.run(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        model_name=model_arch,
        num_classes=NUM_CLASSES,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_channels=NUM_CHANNELS,
        num_epochs=EPOCHS,
        n_trials=TRIALS  # You can increase this
    )

    logger.info(f"🏆 Best trial for {model_arch}: {best_trial.params}")
    log_system_memory_usage()
    log_gpu_memory_usage()

    # === Clean up === #
    logger.info("🧹 Cleaning up resources...")
    K.clear_session()
    gc.collect()
    logger.info("✅ Completed optimization for model: %s\n", model_arch)
