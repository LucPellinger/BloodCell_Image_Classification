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



class BaseModel:
    def __init__(self, model_name, num_classes, input_shape):
        self.model_name = model_name
        self.save_dir_base = "assets/models"

        self.save_dir = os.path.join(self.save_dir_base, self.model_name)
        if os.path.exists(self.save_dir):
            print(f"⚠️ Directory {self.save_dir} already exists. Overwriting...")
        else:
            print(f"Creating directory {self.save_dir} for saving model and history.")
            os.makedirs(self.save_dir, exist_ok=True)

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None

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

    def build_pretrained(self, base_model_name,
                         num_dense_layers=1, neurons_dense=512,
                         dropout_rate=None, normalization=False, l2_reg=None,
                         fine_tune_at=None, unfreeze_all=False):
        print(f"Building model using {base_model_name}")

        base_model_fn = getattr(keras.applications, base_model_name, None)
        if base_model_fn is None:
            raise ValueError(f"Model {base_model_name} not found in keras.applications")

        base_model = base_model_fn(input_shape=self.input_shape, include_top=False, weights='imagenet')

        if unfreeze_all:
            base_model.trainable = True
            print("🔓 Unfreezing all layers for fine-tuning.")
        elif fine_tune_at is not None:
            base_model.trainable = True
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            print(f"🔓 Fine-tuning from layer {fine_tune_at} onwards.")
        else:
            base_model.trainable = False
            print("🔒 Freezing all layers in base model.")

        model_layers = [base_model, layers.Flatten()]

        for _ in range(num_dense_layers):
            model_layers.append(layers.Dense(neurons_dense, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg else None))
            if normalization:
                model_layers.append(layers.BatchNormalization())
            if dropout_rate:
                model_layers.append(layers.Dropout(dropout_rate))

        model_layers.append(layers.Dense(self.num_classes, activation='softmax'))

        self.model = Sequential(model_layers)
        print("✅ Model built successfully")

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
        print("✅ Model compiled")

    def train(self, train_ds, val_ds, epochs=10, early_stopping=False, tensorboard_logdir=None, plot=False):
        if self.model is None:
            raise ValueError("Model not built. Call `build()` or `build_pretrained()` first.")

        callbacks = get_callbacks(tensorboard_logdir)

        if early_stopping:
            callbacks += [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                ModelCheckpoint(filepath=os.path.join(self.save_dir, 'best_model.h5'), monitor='val_loss', save_best_only=True, verbose=1)
            ]

        tf.summary.trace_on(graph=True, profiler=True)

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=tensorboard_logdir)


        print("✅ Training complete")
        if plot:
            performance_plot(self.history)

        return self.history

    def evaluate(self, test_ds):
        if self.model is None:
            raise ValueError("Model not built. Call `build()` or `build_pretrained()` first.")

        results = self.model.evaluate(test_ds)
        print("\n🧪 Test results:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"{name}: {value:.4f}")

    def evaluate_with_report(self, test_ds, class_names):
        """
        Runs full evaluation: classification report, confusion matrix, ROC + PR curves.
        """
        if self.model is None:
            raise ValueError("Model not built. Call `build()` or `build_pretrained()` first.")
        performance_Metrics_U_Visualizations(self.model, test_ds, class_names)

    def save(self):
        model_path = os.path.join(self.save_dir, f"{self.model_name}.keras")
        history_path = os.path.join(self.save_dir, f"{self.model_name}_training_history.pkl")

        self.model.save(model_path)
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)

        print(f"✅ Model saved to {model_path}")
        print(f"✅ Training history saved to {history_path}")

    def load_history(self, history_path=None):
        if history_path is None:
            history_path = os.path.join(self.save_dir, f"{self.model_name}_training_history.pkl")

        with open(history_path, 'rb') as f:
            self.history = pickle.load(f)
        performance_plot(self.history)
        print("✅ History loaded and plotted")

    def plot_model_architecture(self):
        png_path = os.path.join(self.save_dir, f"{self.model_name}_summary.png")
        if self.model:
            plot_model(self.model, to_file=png_path, show_shapes=True, show_layer_names=True)
            print(f"✅ Model plot saved to {png_path}")
        else:
            print("Model not built yet. Please build the model first.")


class BaseCNNModel(BaseModel):
    def __init__(self, model_name, num_classes, input_shape):
        super().__init__(model_name, num_classes, input_shape)

    def plot_model_summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Please build the model first.")

    def build(self):
        print("Building baseline CNN model")
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
        print("✅ Baseline CNN model built successfully")


class BenchmarkKaggleModel(BaseModel):
    def __init__(self, model_name, num_classes, input_shape):
        super().__init__(model_name, num_classes, input_shape)

    def build(self):
        print("Building Kaggle benchmark CNN model")
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
        print("✅ Kaggle benchmark CNN model built successfully")
