import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
)
import os

# =============================
# Final Evaluation
# =============================

def performance_Metrics_U_Visualizations(model, test_ds, class_names, save_dir="assets/eval_results", logger=None):

    y_true, y_pred, y_scores = [], [], []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
        y_scores.extend(predictions)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # === 1. Save Classification Report ===
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("\nClassification Report:\n", report)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    logger.info(f"✅ classification report created.")

    # === 2. Save Confusion Matrix Plot ===
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues)
    ax_cm.set_title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close(fig_cm)
    logger.info(f"✅ Confusion matrix plot created.")

    # === 3. Save ROC Curve Plot ===
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close(fig_roc)
    logger.info(f"✅ ROC curve plot created.")

    # === 4. Save Precision-Recall Curve Plot ===
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, lw=2, label=f'{class_name} (AUC = {pr_auc:.2f})')
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"))
    plt.close(fig_pr)
    logger.info(f"✅ Precision-Recall curve plot created.")

    logger.info(f"✅ Evaluation results saved to: {save_dir}")