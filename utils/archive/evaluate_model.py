import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, precision_recall_curve

def show():
    st.title("🧪 Evaluate Model")
    st.markdown("---")

    st.info("This page displays evaluation metrics of the best model on the test dataset.")

    selected_model = st.selectbox("Select a model to evaluate", ["Best Model (Grid Search)", "Best Model (Bayesian Optimization)"])

    # Simulated metrics
    st.subheader("📊 Classification Report")
    st.text("Note: This is a simulated report. Real evaluation will be wired up later.")
    st.code(classification_report(
        [0, 1, 2, 3], [0, 1, 1, 3],
        target_names=["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
    ))

    st.subheader("🧩 Confusion Matrix")
    cm = np.array([[13, 0, 1, 1],
                   [0, 14, 0, 0],
                   [1, 1, 12, 1],
                   [0, 0, 1, 14]])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"])
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)

    st.subheader("📉 ROC Curve")
    fig, ax = plt.subplots()
    for i, cls in enumerate(["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]):
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # just a dummy curve
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {np.trapz(tpr, fpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📉 Precision-Recall Curve")
    fig, ax = plt.subplots()
    for i, cls in enumerate(["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]):
        recall = np.linspace(0, 1, 100)
        precision = 1 - recall * 0.3  # dummy values
        ax.plot(recall, precision, label=f"{cls} (AUC = {np.trapz(precision, recall):.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    st.pyplot(fig)
