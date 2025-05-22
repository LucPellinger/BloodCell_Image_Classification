import tensorflow as tf
from keras.layers import *
from keras import Sequential
from keras.regularizers import l2

from utils.train_utils import compile_and_fit, get_augmentation_layer


# =============================
# Transfer learning
# =============================

def load_model(model_name, input_shape):
  """
  Loads a pre-trained model from TensorFlow Keras applications with specific configurations.

  Arguments:
  ---------
  model_name (str):
      The model name to load.
  input_shape (tuple):
      Desired input shape for the model.

  Returns:
  --------
  tf.keras.Model:
      The loaded Keras model with ImageNet weights.
  """
  models = {
      "VGG16": keras.applications.VGG16,
      "VGG19": keras.applications.VGG19,
      "InceptionV3": keras.applications.InceptionV3,
      "Xception": keras.applications.Xception,
      "ResNet50": keras.applications.ResNet50,
      "ResNet101": keras.applications.ResNet101,
      "MobileNetV2": keras.applications.MobileNetV2,
      "DenseNet121": keras.applications.DenseNet121,
      "EfficientNetB0": keras.applications.EfficientNetB0,
      "NASNetMobile": keras.applications.NASNetMobile,
  }

  if model_name not in models:
      supported_models = ', '.join(models.keys())
      raise ValueError(f"Unsupported model name {model_name}. Choose from {supported_models}.")

  model = models[model_name](input_shape=input_shape, include_top=False, weights='imagenet')

  print("Utilized **load_model**")
  return model

def freeze(model):

  """
  Sets the layers of the given model to non-trainable.

  Arguments:
  ---------
  model (tf.keras.Model):
      The model whose layers will be set to non-trainable.

  Returns:
  --------
  None:
      The function modifies the model in place.
  """

  model.trainable = False
  print("Utilized **freeze**")

def extend_model(base_model,
                 num_classes,
                 num_dense_layers=1,
                 neurons_dense=500,
                 act_func="relu",
                 dropout_rate=None,
                 normalization=False,
                 l2_reg=None):
  """
  Extends a given base model with additional dense layers and a prediction layer.

  Parameters:
  -----------
  base_model (tf.keras.Model):
      The pre-trained model to extend.
  num_classes (int):
      The number of classes for the final prediction layer.
  num_dense_layers (int, optional):
      The number of dense layers to add (default is 1).
  neurons_dense (int, optional):
      The number of neurons in each dense layer (default is 500).
  act_func (str, optional):
      The activation function to use in the dense layers (default is "relu").
  dropout_rate (float, optional):
      The dropout rate to apply, set to `None` to skip dropout layers (default is None).
  normalization (bool, optional):
      Whether to add batch normalization layers (default is False).
  l2_reg (float, optional):
      The L2 regularization factor to apply, set to `None` to skip L2 regularization (default is None).

  Returns:
  --------
  tf.keras.Model:
      The extended model with the specified layers added.
  """
  # Build the model layer by layer
  layers = [
      # Base model
      base_model,
      Flatten()
  ]

  # Add dense layers with optional batch normalization, dropout, and regularization
  for _ in range(num_dense_layers):
      layers.append(Dense(neurons_dense, activation=act_func, kernel_regularizer=l2(l2_reg) if l2_reg else None))
      if normalization:
          layers.append(BatchNormalization())
      if dropout_rate:
          layers.append(Dropout(dropout_rate))

  # Prediction layer
  layers.append(Dense(num_classes, activation='softmax'))

  # Create the Sequential model with the specified layers
  model = tf.keras.Sequential(layers)

  print("Utilized **extend_model**")
  return model


def get_augmentation_layer(img_height, img_width):
  """
  Compiles and fits a Keras model, then evaluates it on a test dataset.

  Parameters:
  -----------
  model : tf.keras.Model
      The Keras model to be compiled and trained.
  name : str
      The name of the model, used for logging purposes.
  name_tensorboard : str
      The directory name to save TensorBoard logs.
  train_ds : tf.data.Dataset
      The dataset used for training the model.
  val_ds : tf.data.Dataset
      The dataset used for validating the model during training.
  test_ds : tf.data.Dataset
      The dataset used for evaluating the model after training.
  early_stopping : bool, optional
      Whether to stop training when a monitored metric has stopped improving.
  optimizer : tf.keras.optimizers.Optimizer, optional
      The optimizer to use during model compilation. Defaults to Adam optimizer.
  max_epochs : int, optional
      The maximum number of epochs for model training. Defaults to 10.

  Returns:
  --------
  tf.keras.callbacks.History
      The training history returned by model.fit.
  """

  return keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomContrast(0.2),
    RandomBrightness(0.2),
    RandomTranslation(height_factor=0.1, width_factor=0.1),
  ])


def load_and_run(model_name: str,
                 num_classes: int,
                 img_height: int,
                 img_width: int,
                 num_channels: int,
                 train_ds: tf.data.Dataset,
                 val_ds: tf.data.Dataset,
                 test_ds: tf.data.Dataset,
                 num_dense_layers: int = 1,
                 neurons_dense: int = 500,
                 optimizer = None,
                 dropout_rate = None,
                 normalization = False,
                 l2_reg = None,
                 data_augmentation = False,
                 early_stopping = False,
                 epochs: int = 10,
                 tensorboard = True):
  """
  Loads a pre-trained model, freezes its initial layers, adds specified number of dense layers, compiles, trains, and plots training performance.

  Parameters:
  ----------
  model_name (str):
      Name of the model to load from TensorFlow Keras applications.
  num_classes (int):
      Number of output classes for the final dense layer.
  img_height (int):
      Height of the input images.
  img_width (int):
      Width of the input images.
  num_channels (int):
      Number of color channels in the input images.
  train_ds (tf.data.Dataset):
      The dataset to use for training the model.
  val_ds (tf.data.Dataset):
      The dataset to use for validating the model during training.
  epochs (int):
      The number of epochs used for training.
  Returns:
  -------
  None:
      The function does not return a value but prints the model's summary, compiles it, and plots training performance after training.

  Raises:
  ------
  ValueError:
      If the `model_name` is not supported by the TensorFlow Keras applications.
  """

  model = load_model(model_name=model_name,
                      input_shape=[img_height, img_width, num_channels])
  freeze(model)
  model = extend_model(base_model = model,
                        num_classes = num_classes,
                        num_dense_layers = num_dense_layers,
                        neurons_dense = neurons_dense,
                        dropout_rate = dropout_rate,
                        normalization = normalization,
                        l2_reg = l2_reg)
  if data_augmentation:
        augmentation_layer = get_augmentation_layer(img_height, img_width)
        train_ds = train_ds.map(lambda x, y: (augmentation_layer(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

  try:
    # Compiling & training the model only once
    model, history = compile_and_fit(model = model,
                              name = model_name,
                              name_tensorboard = f'models/{model_name}',
                              train_ds = train_ds,
                              val_ds = val_ds,
                              test_ds = test_ds,
                              early_stopping = early_stopping,
                              optimizer = optimizer,
                              max_epochs = epochs)
    if tensorboard:
      size_histories[model_name] = history
    print("Utilized **load_and_run**")

    model_path = output_path + f"/{model_name}"
    # create directories
    os.makedirs(model_path, exist_ok=True)

    model_history_path_file = model_path + "/training_history.pkl"
    model_path_file = model_path + f"/{model_name}.keras"

    # Save the model history
    import pickle

    # Save training history
    with open(model_history_path_file, 'wb') as f:
        pickle.dump(size_histories[f'{model_name}'].history, f)

    print("✅ Training history saved to 'training_history.pkl'")

    # Save the model
    model.save(model_path_file)


    # Load training history
    with open(model_history_path_file, 'rb') as f:
        loaded_history = pickle.load(f)

    print("✅ Loaded training history")

    # plot the accuracy as graph
    import matplotlib.pyplot as plt

    plt.plot(loaded_history['accuracy'], label='Training Accuracy')
    plt.plot(loaded_history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.legend()
    plt.show()

  except Exception as e:
    print(f"Error during training: {str(e)}")
    history = None

  return model, history