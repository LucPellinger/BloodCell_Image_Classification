import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from utils.metrics import MulticlassPrecision, MulticlassRecall
from utils.archive.train_utils import get_callbacks, performance_plot
from utils.evaluation import performance_Metrics_U_Visualizations
import logging
import subprocess




class BaseModel:
    def __init__(self, model_name, num_classes, input_shape):
        self.model_name = model_name
        self.save_dir_base = "assets/models"
        self.save_dir = os.path.join(self.save_dir_base, self.model_name)

        # Setup model-specific logger
        self.logger = logging.getLogger(f"{model_name}_logger")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('\033[34;1m%(asctime)s | MODEL | %(message)s\033[0m')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        if os.path.exists(self.save_dir):
            self.logger.warning(f"⚠️ Directory {self.save_dir} already exists. Overwriting...")
        else:
            self.logger.info(f"Creating directory {self.save_dir} for saving model and history.")
            os.makedirs(self.save_dir, exist_ok=True)

        self.configure_gpu()
        self.logger.info("✅ TensorFlow GPU configuration complete.")

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def configure_gpu(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus[0], 'GPU')  # restrict to first GPU
                self.logger.info(f"Using GPU: {gpus[0].name}")
            else:
                self.logger.info("No GPU found. Using CPU.")
        except RuntimeError as e:
            self.logger.warning(f"GPU config error: {e}")

    def log_gpu_usage(self):
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"])
            used, total = output.decode().strip().split('\n')[0].split(', ')
            self.logger.info(f"🔍 GPU Memory Usage: {used}MB / {total}MB")
        except Exception as e:
            self.logger.warning(f"Could not query GPU usage: {e}")

    @staticmethod
    def list_available_pretrained_models():
        """List all available pretrained models from keras.applications."""
        from inspect import isfunction, isclass
        import tensorflow.keras.applications as ka

        available_models = [name for name in dir(ka)
                            if not name.startswith("_") and
                            (isfunction(getattr(ka, name)) or isclass(getattr(ka, name)))]

        print("📚 Available models in keras.applications:")
        for model_name in sorted(available_models):
            print(f" - {model_name}")

        return available_models

    def build_pretrained(self, base_model_name,
                         num_dense_layers=1, neurons_dense=512,
                         dropout_rate=None, normalization=False, l2_reg=None,
                         fine_tune_at=None, unfreeze_all=False):
        self.logger.info(f"Building model using {base_model_name}")

        base_model_fn = getattr(keras.applications, base_model_name, None)
        if base_model_fn is None:
            raise ValueError(f"Model {base_model_name} not found in keras.applications")

        base_model = base_model_fn(input_shape=self.input_shape, include_top=False, weights='imagenet')

        if unfreeze_all:
            base_model.trainable = True
            self.logger.info("🔓 Unfreezing all layers for fine-tuning.")
        elif fine_tune_at is not None:
            base_model.trainable = True
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            self.logger.info(f"🔓 Fine-tuning from layer {fine_tune_at} onwards.")
        else:
            base_model.trainable = False
            self.logger.info("🔒 Freezing all layers in base model.")

        model_layers = [base_model, layers.GlobalAveragePooling2D()] # .Faltten() is old, use GlobalAveragePooling2D for memory efficiency 
        # more under: https://stackoverflow.com/questions/49295311/what-is-the-difference-between-flatten-and-globalaveragepooling2d-in-keras


        for _ in range(num_dense_layers):
            model_layers.append(layers.Dense(neurons_dense, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg else None))
            if normalization:
                model_layers.append(layers.BatchNormalization())
            if dropout_rate:
                model_layers.append(layers.Dropout(dropout_rate))

        model_layers.append(layers.Dense(self.num_classes, activation='softmax'))

        self.model = Sequential(model_layers)
        self.logger.info("✅ Model built successfully")
        self.log_gpu_usage()

    def compile(self, optimizer=None, loss='sparse_categorical_crossentropy'):
        if self.model is None:
            raise ValueError("Model not built. Call `build()` or `build_pretrained()` first.")

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                MulticlassPrecision(num_classes=self.num_classes),
                MulticlassRecall(num_classes=self.num_classes)
            ]
        )
        self.logger.info("✅ Model compiled")
        self.log_gpu_usage()

    def train(self, train_ds, val_ds, epochs=10, early_stopping=False, tensorboard_logdir=None, plot=False):
        if self.model is None:
            raise ValueError("Model not built. Call `build()` or `build_pretrained()` first.")

        callbacks = get_callbacks(tensorboard_logdir)

        if early_stopping:
            callbacks += [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                ModelCheckpoint(filepath=os.path.join(self.save_dir, 'best_model.h5'), monitor='val_loss', save_best_only=True, verbose=1)
            ]

        #tf.summary.trace_on(graph=True, profiler=True)

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            #steps_per_epoch=len(train_ds),
        )

        #tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=tensorboard_logdir)


        self.logger.info("✅ Training complete")
        if plot:
            performance_plot(self.history)
        self.log_gpu_usage()

        return self.history

    def evaluate(self, test_ds):
        if self.model is None:
            raise ValueError("Model not built. Call `build()` or `build_pretrained()` first.")

        results = self.model.evaluate(test_ds)
        self.logger.info("\n🧪 Test results:")
        for name, value in zip(self.model.metrics_names, results):
            self.logger.info(f"{name}: {value:.4f}")
        self.log_gpu_usage()

    def evaluate_with_report(self, test_ds, class_names):
        """
        Runs full evaluation: classification report, confusion matrix, ROC + PR curves.
        """
        if self.model is None:
            raise ValueError("Model not built. Call `build()` or `build_pretrained()` first.")
        performance_Metrics_U_Visualizations(model=self.model, test_ds=test_ds, class_names=class_names, save_dir=self.save_dir, logger=self.logger)

    def save(self):
        model_path = os.path.join(self.save_dir, f"{self.model_name}.keras")
        history_path = os.path.join(self.save_dir, f"{self.model_name}_training_history.pkl")

        self.model.save(model_path)
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)

        self.logger.info(f"✅ Model saved to {model_path}")
        self.logger.info(f"✅ Training history saved to {history_path}")

    def load_history(self, history_path=None):
        if history_path is None:
            history_path = os.path.join(self.save_dir, f"{self.model_name}_training_history.pkl")

        with open(history_path, 'rb') as f:
            self.history = pickle.load(f)
        performance_plot(self.history)
        self.logger.info("✅ History loaded and plotted")

    def plot_model_architecture(self):
        png_path = os.path.join(self.save_dir, f"{self.model_name}_summary.png")
        if self.model:
            plot_model(self.model, to_file=png_path, show_shapes=True, show_layer_names=True)
            self.logger.info(f"✅ Model plot saved to {png_path}")
            self.log_gpu_usage()
        else:
            self.logger.info("Model not built yet. Please build the model first.")


class BaseCNNModel(BaseModel):
    def __init__(self, model_name, num_classes, input_shape):
        super().__init__(model_name, num_classes, input_shape)

    def plot_model_summary(self):
        if self.model:
            self.model.summary()
        else:
            self.logger.info("Model not built yet. Please build the model first.")

    def build(self):
        self.logger.info("Building baseline CNN model")
        self.model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        self.logger.info("✅ Baseline CNN model built successfully")
        self.log_gpu_usage()


class BenchmarkKaggleModel(BaseModel):
    def __init__(self, model_name, num_classes, input_shape):
        super().__init__(model_name, num_classes, input_shape)

    def build(self):
        self.logger.info("Building Kaggle benchmark CNN model")
        self.model = Sequential([
            layers.Conv2D(128, (8, 8), strides=(3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),

            layers.Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(3, 3)),

            layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (1, 1), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (1, 1), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),

            layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),

            layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),

            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),

            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        self.logger.info("✅ Kaggle benchmark CNN model built successfully")
        self.log_gpu_usage()