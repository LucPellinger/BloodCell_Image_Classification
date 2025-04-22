import streamlit as st
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from streamlit.components.v1 import html

def show():
    st.title("🧪 Model Optimization (Optuna)")
    st.markdown("---")

    st.markdown("""
    ## 🔬 Optuna Optimization Overview

    Below you’ll find visualizations from the Bayesian hyperparameter optimization process powered by **Optuna**.
    We used this to fine-tune parameters like learning rate, dropout, regularization, and more.
    """)

    # Path to your SQLite study DB
    db_path = "models/optuna_study.db"  # Adjust if stored elsewhere
    storage_url = f"sqlite:///{db_path}"

    try:
        study = optuna.load_study(study_name="blood_cell_study", storage=storage_url)

        st.subheader("📈 Optimization History")
        fig1 = plot_optimization_history(study)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("🧬 Parameter Importances")
        fig2 = plot_param_importances(study)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🏆 Best Hyperparameters")
        st.json(study.best_params)

    except Exception as e:
        st.error(f"Failed to load Optuna study: {e}")
        st.info("Make sure the study exists in your `.db` file and the name is correct.")
