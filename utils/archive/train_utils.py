import tensorflow as tf
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
from utils.metrics import MulticlassPrecision, MulticlassRecall

# =============================
# Plot for single evaluation
# =============================

def performance_plot(hist):

  """
  Visualizes the performance metrics such as accuracy and loss across epochs for training and validation phases.

  Args:
      hist (History): A TensorFlow History object containing records of training and validation statistics per epoch.

  Displays:
      Line graphs for training and validation accuracy, as well as loss.
  """

  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  loss = hist.history['loss']
  val_loss = hist.history['val_loss']

  # Use the length of the loss array to determine the number of epochs
  epochs_range = range(len(loss))

  plt.figure(figsize=(8, 8))

  # Plot Training and Validation Accuracy
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  # Plot Training and Validation Loss
  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')

  plt.show()

# =============================
# Tensorboard
# =============================

def get_callbacks(name):

  """
  Generate a list of callbacks for model training including TensorBoard logging and EpochDots for progress.

  Parameters:
  -----------
  name : str
      The directory name to save TensorBoard logs.

  Returns:
  --------
  list
      A list of TensorFlow Keras callbacks including model checkpointing and early stopping.
  """

  print("Utilized **get_callbacks**")
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.TensorBoard(logdir/name, update_freq="epoch"),
  ]


def compile_and_fit(model,
                    name,
                    name_tensorboard,
                    train_ds,
                    val_ds,
                    test_ds,
                    early_stopping = False,
                    optimizer = None,
                    max_epochs = 10):

  """
  Compile and fit a Keras model, then evaluate it on a test dataset.

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
  optimizer : tf.keras.optimizers.Optimizer, optional
      The optimizer to use during model compilation. Defaults to Adam optimizer.
  max_epochs : int, optional
      The maximum number of epochs for model training. Defaults to 10.

  Returns:
  --------
  tf.keras.callbacks.History
      The training history returned by model.fit.
  """
  if optimizer is None:
      optimizer = tf.keras.optimizers.Adam()

  model.compile(
      optimizer=optimizer,
      loss='sparse_categorical_crossentropy',
      metrics=[
          'accuracy',
          MulticlassPrecision(num_classes=4),
          MulticlassRecall(num_classes=4)
      ]
  )

  model.summary()

  callbacks = get_callbacks(name_tensorboard)

  if early_stopping:
    callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1))
    callbacks.append(ModelCheckpoint(filepath='best_model.h5',  # File path where the model will be saved
                                     monitor='val_loss',        # Metric to monitor
                                     mode='min',                # We want to minimize the validation loss
                                     save_best_only=True,       # Save only the best model
                                     verbose=1))
  history = model.fit(
      train_ds,
      epochs=max_epochs,
      validation_data=val_ds,
      callbacks=callbacks
  )

  print("\nEvaluating {} using the test-set:".format(name))
  model.evaluate(test_ds)

  print("Utilized **compile_and_fit**")
  return model, history