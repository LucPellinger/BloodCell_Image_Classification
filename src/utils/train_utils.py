# utils/train_utils.py


import tensorflow as tf
import optuna
#import tensorflow_docs.modeling
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

    # print("Utilized **get_callbacks**")
    # return [
    #   #tfdocs.modeling.EpochDots(),
    #   #tf.keras.callbacks.TensorBoard(log_dir=name, update_freq="epoch", histogram_freq=1, write_graph=True),
    #   tf.keras.callbacks.TensorBoard(log_dir=name, profile_batch=0, write_graph=False)
# 
    # ]
    print("Utilized **get_callbacks**")
    cbs = []
    if name:
        cbs.append(tf.keras.callbacks.TensorBoard(
            log_dir=name,
            profile_batch=0,     # disable profiler -> avoids CUPTI + deepcopy surprises
            histogram_freq=0,    # lighter & faster
            write_graph=False
        ))
    return cbs

class SafePruningCallback(tf.keras.callbacks.Callback):
    """Optuna pruning that won't break when Keras deepcopy()'s callbacks."""
    def __init__(self, trial, monitor="val_accuracy"):
        super().__init__()
        self._trial = trial
        self.monitor = monitor

    # Make deepcopy() / pickling ignore non-picklable fields.
    def __getstate__(self):
        return {"monitor": self.monitor, "trial_number": self._trial.number}

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        self._trial.report(float(value), step=epoch)
        if self._trial.should_prune():
            # Stop training immediately, signal Optuna to prune.
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}")