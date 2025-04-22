import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay

# =============================
# Final Evaluation
# =============================

def performance_Metrics_U_Visualizations(model,
                                         test_ds,
                                         class_names):
  """
  Evaluates the model on the test dataset and visualizes classification metrics including ROC and precision-recall curves.

  Parameters:
  ----------
  model (tf.keras.Model): The trained model to evaluate.
  test_ds (tf.data.Dataset): The test dataset.
  class_names (list): List of class names for labeling purposes in plots.

  Displays:
  --------
  Prints the classification report, shows the confusion matrix, ROC curves, and precision-recall curves.
  """

  y_true, y_pred, y_scores = [], [], []

  # Collect predictions and true labels
  for images, labels in test_ds:
      predictions = model.predict(images, verbose=0)
      y_pred.extend(np.argmax(predictions, axis=1))
      y_true.extend(labels.numpy())
      y_scores.extend(predictions)

  # Convert lists to numpy arrays
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  y_scores = np.array(y_scores)

  # Classification Report
  print("\nClassification Report:")
  print(classification_report(y_true, y_pred, target_names=class_names))

  # Confusion Matrix
  fig, ax = plt.subplots(figsize=(10, 8))
  cm = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot(ax=ax, cmap=plt.cm.Blues)
  ax.set_title('Confusion Matrix')
  plt.show()

  # Calculate and plot ROC curve and AUC for each class
  fig, ax = plt.subplots(figsize=(10, 8))
  for i, class_name in enumerate(class_names):
      fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
      roc_auc = auc(fpr, tpr)
      ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

  ax.plot([0, 1], [0, 1], 'k--', lw=2)
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.05])
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title('Receiver Operating Characteristic (ROC) Curve')
  ax.legend(loc="lower right")
  plt.show()

  # Calculate and plot Precision-Recall curve for each class
  fig, ax = plt.subplots(figsize=(10, 8))
  for i, class_name in enumerate(class_names):
      precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
      pr_auc = auc(recall, precision)
      ax.plot(recall, precision, lw=2, label=f'{class_name} (AUC = {pr_auc:.2f})')

  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.05])
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.set_title('Precision-Recall Curve')
  ax.legend(loc="lower left")
  plt.show()