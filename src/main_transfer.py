import logging
import sys
import os
import datetime

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import load_datasets
from utils.base_model import BaseModel  # <-- Use BaseModel for transfer learning
from utils.eda import display_class_representatives_plt, plot_dataset_distributions

# === SETUP LOGGING === #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("transfer_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import tensorflow as tf

# === STEP 4: GPU Setup === #
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

import gc
from tensorflow.keras import backend as K
import psutil
import subprocess

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




logger.info("🔧 Setting up logging...")
logger.info("Num GPUs Available: %s", len(tf.config.list_physical_devices('GPU')))

# next we will also add a transformer based model called ViT

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

for i in models:

    logger.info(f"📊 Monitoring memory before training {i}...")
    log_system_memory_usage()
    log_gpu_memory_usage()

    logger.info("Available model: %s", i)

    # === CONFIGURATION === #
    IMG_HEIGHT = 240
    IMG_WIDTH = 320
    NUM_CHANNELS = 3
    NUM_CLASSES = 4
    BATCH_SIZE = 32 # 32
    EPOCHS = 10
    CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
    TRAIN_DIR = "assets/data/dataset2-master/dataset2-master/images/TRAIN"
    TEST_DIR = "assets/data/dataset2-master/dataset2-master/images/TEST"
    MODEL_NAME = f"{i}_transfer"
    log_dir = os.path.join("assets/tf_logs", MODEL_NAME, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    # === STEP 1: Load Data === #
    logger.info("📦 Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE
    )

    # === STEP 2: EDA === #
    logger.info("🔍 Plotting class distributions...")
    plot_dataset_distributions(train_ds, val_ds, test_ds, IMG_HEIGHT, IMG_WIDTH, class_names)
    logger.info("🖼️ Displaying representative images...")
    display_class_representatives_plt(TRAIN_DIR, IMG_HEIGHT, IMG_WIDTH)

    # === STEP 3: Build Transfer Model === #
    logger.info("🧠 Building transfer learning model...")
    model = BaseModel(MODEL_NAME, NUM_CLASSES, (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))

    model.build_pretrained(
        base_model_name=i,  # or MobileNetV2, EfficientNetB0, etc.
        num_dense_layers=1,
        neurons_dense=512,
        dropout_rate=0.5,
        normalization=True,
        l2_reg=1e-4,
        augmentation=True
    )

    model.compile()

    # === STEP 5: Training === #
    logger.info("🚀 Starting model training...")
    model.train(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=EPOCHS,
        early_stopping=True,
        plot=False,
        tensorboard_logdir=log_dir
    )

    logger.info(f"📉 Monitoring memory after training {i}...")
    log_system_memory_usage()
    log_gpu_memory_usage()    

    # === STEP 6: Evaluation === #
    logger.info("📊 Evaluating on test set...")
    model.evaluate(test_ds)
    model.evaluate_with_report(test_ds, class_names=class_names)

    # === STEP 7: Save === #
    logger.info("💾 Saving model and training history...")
    model.save()

    logger.info("✅ All steps completed successfully.")

    # === STEP 8: Cleanup === #
    logger.info("🧹 Clearing model from memory to free up resources...")
    del model
    K.clear_session()
    gc.collect()
    logger.info("✅ Resources released.\n")
