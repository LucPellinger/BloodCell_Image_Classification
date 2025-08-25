# app/pages/models_training.py
import os, pickle
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from app.components.logger import get_logger
from app.components.path_utils import get_project_root
from app.components.shared import markdown_header

logger = get_logger("models_training")

# ---------- paths & discovery ----------
def _models_root():
    # src/assets/models
    return os.path.join(get_project_root(), "assets", "models")

def _discover_models():
    root = _models_root()
    if not os.path.isdir(root):
        logger.warning(f"Models root not found: {root}")
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

# ---------- loaders ----------
def _load_pickle_history(model_dir: str, folder_name: str):
    """
    Expect: <folder>/<folder>_training_history.pkl
    Returns a dict like Keras History.history (keys -> list of values) or None.
    """
    pkl_path = os.path.join(model_dir, f"{folder_name}_training_history.pkl")
    if not os.path.isfile(pkl_path):
        logger.warning(f"History pickle not found: {pkl_path}")
        return None

    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            # many projects save the dict directly
            return obj
        hist = getattr(obj, "history", None)
        if isinstance(hist, dict):
            return hist
        logger.warning(f"Pickle loaded but not a history dict: type={type(obj)} at {pkl_path}")
        return None
    except Exception as e:
        logger.exception(f"Failed to read {pkl_path}: {e}")
        return None

def _read_text_if_exists(path):
    try:
        if os.path.isfile(path):
            with open(path, "r") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"Could not read text file {path}: {e}")
    return None

def _available_images(model_dir):
    found = {}
    candidates = {
        "Confusion Matrix": ["confusion_matrix.png", "cm.png"],
        "ROC Curve": ["roc_curve.png", "roc.png"],
        "PR Curve": ["precision_recall_curve.png", "pr.png"],
        "Curves (precomputed)": ["curves.png", "training_curves.png"],
    }
    for label, names in candidates.items():
        for n in names:
            p = os.path.join(model_dir, n)
            if os.path.isfile(p):
                found[label] = p
                break
    return found

# ---------- plotting ----------
def _plot_from_history(history: dict):
    """
    Return a Plotly Figure for training/validation loss & accuracy,
    color-coded as:
      - train loss = light orange
      - val loss   = orange
      - train acc  = light green
      - val acc    = green
    Other metrics (precision/recall/auc) will get default colors.
    """
    if not isinstance(history, dict) or not history:
        return None

    # determine number of epochs from first list
    length = 0
    for v in history.values():
        if isinstance(v, (list, tuple)) and v:
            length = len(v)
            break
    if not length:
        return None

    epochs = list(range(1, length + 1))
    fig = go.Figure()

    # --- helpers ---
    def maybe_plot(name, label, color=None):
        v = history.get(name)
        if isinstance(v, (list, tuple)) and len(v) == length:
            fig.add_trace(go.Scatter(
                x=epochs, y=v,
                mode="lines",
                name=label,
                line=dict(color=color) if color else None
            ))

    # add key curves with fixed colors
    maybe_plot("loss", "train loss", "#FFD580")       # light orange
    maybe_plot("val_loss", "val loss", "#FF8C00")     # orange
    maybe_plot("accuracy", "train acc", "#90EE90")    # light green
    maybe_plot("val_accuracy", "val acc", "#006400")  # green

    # optional extra metrics, default colors
    maybe_plot("precision", "precision")
    maybe_plot("recall", "recall")
    maybe_plot("auc", "auc")
    maybe_plot("val_precision", "val precision")
    maybe_plot("val_recall", "val recall")
    maybe_plot("val_auc", "val auc")

    if not fig.data:
        return None

    fig.update_layout(
        title="Training History",
        xaxis_title="Epoch",
        yaxis_title="Value",
        template="plotly_white",
        autosize=True,
        margin=dict(l=40, r=20, t=40, b=40),
        height=400
    )
    return fig

# ---------- page ----------
def models_overview():
    header_md_path = os.path.join(os.path.dirname(__file__), "..", "components", "markdown", "header_models.md")
    with gr.Column() as layout:
        # header: use your markdown if present, otherwise a simple title
        if os.path.isfile(header_md_path):
            markdown_header(header_md_path)
        else:
            gr.Markdown("## 🧠 Models — Training Data")

        model_names = _discover_models()
        default_model = model_names[0] if model_names else None

        # precompute initial values
        if default_model:
            mdir = os.path.join(_models_root(), default_model)
            hist = _load_pickle_history(mdir, default_model)
            init_fig = _plot_from_history(hist)
            init_imgs = _available_images(mdir)
            init_report = _read_text_if_exists(os.path.join(mdir, "classification_report.txt"))
            init_table = []
            if isinstance(hist, dict):
                for k, arr in hist.items():
                    if isinstance(arr, (list, tuple)) and arr:
                        try:
                            init_table.append([f"last_{k}", round(float(arr[-1]), 6)])
                        except Exception:
                            init_table.append([f"last_{k}", arr[-1]])
        else:
            init_fig = None; init_imgs = {}; init_report = None; init_table = []

        with gr.Row():
            dd_model = gr.Dropdown(
                label="Select a model",
                choices=model_names,
                value=default_model,
                interactive=True,
            )
            btn_refresh = gr.Button("🔄 Refresh list")

        with gr.Row():
            plot_curves = gr.Plot(label="Training curves",
                                  value=init_fig,
                                  visible=init_fig is not None)
            img_curves = gr.Image(label="Curves (image)",
                                  value=init_imgs.get("Curves (precomputed)"),
                                  visible=("Curves (precomputed)" in init_imgs))

        with gr.Row():
            img_cm  = gr.Image(label="Confusion Matrix",
                               value=init_imgs.get("Confusion Matrix"),
                               visible=("Confusion Matrix" in init_imgs))
            img_roc = gr.Image(label="ROC Curve",
                               value=init_imgs.get("ROC Curve"),
                               visible=("ROC Curve" in init_imgs))
            img_pr  = gr.Image(label="PR Curve",
                               value=init_imgs.get("PR Curve"),
                               visible=("PR Curve" in init_imgs))

        with gr.Accordion("Classification report (if available)", open=bool(init_report)):
            md_report = gr.Markdown(value=(f"```\n{init_report}\n```" if init_report else ""),
                                    visible=bool(init_report))

        with gr.Accordion("Last-epoch metrics", open=bool(init_table)):
            df_metrics = gr.Dataframe(headers=["Metric", "Value"],
                                      value=init_table,
                                      row_count=(0, "dynamic"), col_count=2)

        # ---- callbacks ----
        def _on_select(folder_name):
            if not folder_name:
                return (
                    gr.update(visible=False),  # plot_curves
                    gr.update(visible=False),  # img_curves
                    gr.update(visible=False),  # img_cm
                    gr.update(visible=False),  # img_roc
                    gr.update(visible=False),  # img_pr
                    gr.update(visible=False),  # md_report
                    []                         # df_metrics
                )

            mdir = os.path.join(_models_root(), folder_name)
            hist = _load_pickle_history(mdir, folder_name)

            # plotly figure
            fig = _plot_from_history(hist)

            imgs = _available_images(mdir)
            report_text = _read_text_if_exists(os.path.join(mdir, "classification_report.txt"))

            table = []
            if isinstance(hist, dict):
                for k, arr in hist.items():
                    if isinstance(arr, (list, tuple)) and arr:
                        try:
                            table.append([f"last_{k}", round(float(arr[-1]), 6)])
                        except Exception:
                            table.append([f"last_{k}", arr[-1]])

            return (
                gr.update(value=fig, visible=fig is not None),
                gr.update(value=imgs.get("Curves (precomputed)"),
                          visible=("Curves (precomputed)" in imgs)),
                gr.update(value=imgs.get("Confusion Matrix"),
                          visible=("Confusion Matrix" in imgs)),
                gr.update(value=imgs.get("ROC Curve"),
                          visible=("ROC Curve" in imgs)),
                gr.update(value=imgs.get("PR Curve"),
                          visible=("PR Curve" in imgs)),
                gr.update(value=(f"```\n{report_text}\n```" if report_text else ""),
                          visible=bool(report_text)),
                table
            )

        def _refresh():
            names = _discover_models()
            return gr.Dropdown(choices=names, value=(names[0] if names else None), interactive=True)

        dd_model.change(
            _on_select,
            inputs=[dd_model],
            outputs=[plot_curves, img_curves, img_cm, img_roc, img_pr, md_report, df_metrics],
        )
        btn_refresh.click(_refresh, inputs=None, outputs=[dd_model])
        

        # initialize first model (if any)
        if model_names:
            _on_select(model_names[0])

        gr.Markdown("""
                    ### Project Context
                    
                    12 different models for blood cell image classification were trained initially in their vanilla form (i.e., without hyperparameter tuning) to get a first idea of their performance on this specific task. 
                    
                    The models included:
                    
                    - 🧩 A simple baseline CNN  
                    - 🏆 Benchmarking a well performing CNN architecture from strong modeling attempt posted on Kaggle  
                    - 🔁 Ten transfer learning models via Tensorflow pre-trained on large datasets like ImageNet  
                    
                    All models were tracked with TensorBoard during training. However, due to the limitations of Gradio's TensorBoard integration, the training histories were accessed using pickle files for visualizations within this interface. 
                    
                    As evaluation metrics, the experiments focused not only on **accuracy**, but also on **precision** (avoiding false positives) and **recall** (catching as many true positives as possible) — both crucial in medical diagnostics.  
                    
                    (*The F1 score was also explored, but as it was deprecated at the time, precision & recall were emphasized instead.*)
                    """)

    return layout
