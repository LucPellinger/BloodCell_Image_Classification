import tensorflow as tf

"""
Custom metric class to calculate precision for multiclass classification tasks.
This metric computes the precision for each class separately, then averages them
to provide a global precision across all classes.
"""

@tf.keras.utils.register_keras_serializable()
class MulticlassPrecision(tf.keras.metrics.Metric):
  def __init__(self, num_classes, name='multiclass_precision', **kwargs):

    """
    Initializes the MulticlassPrecision object.

    Args:
        num_classes (int): Number of classes in the classification task.
        name (str, optional): Name of the metric instance. Defaults to 'multiclass_precision'.
        **kwargs: Additional keyword arguments passed to the parent constructor.
    """

    super(MulticlassPrecision, self).__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.precision_objects = [tf.keras.metrics.Precision() for i in range(num_classes)] #class_id=i

  def update_state(self, y_true, y_pred, sample_weight=None):

    """
    Accumulates the precision statistics for each batch of data.

    Args:
        y_true (Tensor): The true labels. Shape (batch_size, ).
        y_pred (Tensor): The predicted probabilities or class logits. Shape (batch_size, num_classes).
        sample_weight (Tensor, optional): Optional weighting of each example. Defaults to None.
    """

    y_true = tf.cast(y_true, 'int32')
    for i, precision_obj in enumerate(self.precision_objects):
        precision_obj.update_state(y_true == i, tf.argmax(y_pred, axis=-1) == i)

  def result(self):

    """
    Computes the final average precision over all classes.

    Returns:
        float: The average precision across all classes.
    """

    result = tf.add_n([p.result() for p in self.precision_objects]) / self.num_classes
    return result

  def reset_states(self):

    """
    Resets all of the metric state variables.
    This function is called between epochs/stages of training.
    """

    for p in self.precision_objects:
        p.reset_states()

  def get_config(self):
      config = super().get_config()
      config.update({
          'num_classes': self.num_classes
      })
      return config
"""
Custom metric class to calculate recall for multiclass classification tasks.
This metric computes the recall for each class separately, then averages them
to provide a global recall across all classes.
"""
@tf.keras.utils.register_keras_serializable()
class MulticlassRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='multiclass_recall', **kwargs):

      """
      Initializes the MulticlassRecall object.

      Args:
          num_classes (int): Number of classes in the classification task.
          name (str, optional): Name of the metric instance. Defaults to 'multiclass_recall'.
          **kwargs: Additional keyword arguments passed to the parent constructor.
      """

      super(MulticlassRecall, self).__init__(name=name, **kwargs)
      self.num_classes = num_classes
      self.recall_objects = [tf.keras.metrics.Recall() for i in range(num_classes)] #class_id=i

    def update_state(self, y_true, y_pred, sample_weight=None):

      """
      Accumulates the recall statistics for each batch of data.

      Args:
          y_true (Tensor): The true labels. Shape (batch_size, ).
          y_pred (Tensor): The predicted probabilities or class logits. Shape (batch_size, num_classes).
          sample_weight (Tensor, optional): Optional weighting of each example. Defaults to None.
      """

      y_true = tf.cast(y_true, 'int32')
      for i, recall_obj in enumerate(self.recall_objects):
          recall_obj.update_state(y_true == i, tf.argmax(y_pred, axis=-1) == i)

    def result(self):

      """
      Computes the final average recall over all classes.

      Returns:
          float: The average recall across all classes.
      """

      result = tf.add_n([r.result() for r in self.recall_objects]) / self.num_classes
      return result

    def reset_states(self):

      """
      Resets all of the metric state variables.
      This function is called between epochs/stages of training.
      """

      for r in self.recall_objects:
          r.reset_states()

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config