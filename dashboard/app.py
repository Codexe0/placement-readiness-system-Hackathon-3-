import os
import json
import glob
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="Placement Readiness Analytics", layout="wide")
st.title("Placement Readiness Analytics Dashboard")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------
df = pd.read_csv(project_path("data", "readiness_data.csv"))

# ------------------------------------------------------------------
# Top-level KPI metrics
# ------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
col2.metric(
    "Ready %",
    round(df["readiness"].value_counts(normalize=True).get("Ready", 0) * 100, 2),
)
col3.metric(
    "Not Ready %",
    round(df["readiness"].value_counts(normalize=True).get("Not Ready", 0) * 100, 2),
)

st.divider()

# ------------------------------------------------------------------
# Load model artifact – prefer new primary path, fall back to legacy
# ------------------------------------------------------------------
MODEL_PATH = project_path("models", "readiness_model.pkl")
LEGACY_PATH = project_path("models", "readiness_model_latest.pkl")
UPDATES_PATH = project_path("models", "model_updates.json")


def _load_artifact(path):
    """Return unpickled artifact dict or None if the file is missing/corrupt."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


artifact = _load_artifact(MODEL_PATH) or _load_artifact(LEGACY_PATH)

# Show model metadata in a collapsible expander
meta_files = sorted(glob.glob(str(project_path("models", "readiness_model_*.json"))), reverse=True)
if meta_files:
    try:
        with open(meta_files[0], "r") as f:
            model_meta = json.load(f)
        with st.expander("Model Information"):
            st.write(f"**Timestamp**: {model_meta.get('timestamp', 'N/A')}")
            if "accuracy" in model_meta:
                st.write(f"**Accuracy**: {model_meta['accuracy']:.4f}")
            if artifact and artifact.get("model_name"):
                st.write(f"**Best Model**: {artifact['model_name']}")
    except Exception:
        pass

# ------------------------------------------------------------------
# Main tab layout
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Overview", "Insights", "Model Comparison", "Prediction"]
)

# ══════════════════════════════════════════════════════════════════
# TAB 1 – Data Overview
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.sample(min(10, len(df))))

    st.subheader("Class Distribution")
    dist = df["readiness"].value_counts()
    st.bar_chart(dist)

# ══════════════════════════════════════════════════════════════════
# TAB 2 – Insights
# ══════════════════════════════════════════════════════════════════
with tab2:
    # --- Placement readiness bar chart ---
    st.subheader("Placement Readiness Distribution")
    dist_counts = df["readiness"].value_counts()
    fig_dist, ax_dist = plt.subplots()
    ax_dist.bar(dist_counts.index, dist_counts.values, color=["#4CAF50", "#F44336"])
    ax_dist.set_xlabel("Readiness")
    ax_dist.set_ylabel("Number of Students")
    ax_dist.set_title("Ready vs Not Ready")
    st.pyplot(fig_dist)

    # --- CGPA vs Readiness boxplot ---
    st.subheader("CGPA by Readiness (Boxplot)")
    ready_cgpa     = df[df["readiness"] == "Ready"]["cgpa"]
    not_ready_cgpa = df[df["readiness"] == "Not Ready"]["cgpa"]
    fig_box, ax_box = plt.subplots()
    ax_box.boxplot(
        [ready_cgpa, not_ready_cgpa],
        labels=["Ready", "Not Ready"],
        patch_artist=True,
    )
    ax_box.set_ylabel("CGPA")
    ax_box.set_title("CGPA Distribution by Placement Readiness")
    st.pyplot(fig_box)

    # --- Aptitude score histogram ---
    st.subheader("Aptitude Score Distribution")
    fig_apt, ax_apt = plt.subplots()
    ax_apt.hist(df["aptitude_score"], bins=20, color="#2196F3", edgecolor="white")
    ax_apt.set_xlabel("Aptitude Score")
    ax_apt.set_ylabel("Count")
    ax_apt.set_title("Aptitude Score Distribution")
    st.pyplot(fig_apt)

    # --- CGPA histogram ---
    st.subheader("CGPA Distribution")
    fig_cgpa, ax_cgpa = plt.subplots()
    ax_cgpa.hist(df["cgpa"], bins=20, color="#9C27B0", edgecolor="white")
    ax_cgpa.set_xlabel("CGPA")
    ax_cgpa.set_ylabel("Count")
    ax_cgpa.set_title("CGPA Distribution")
    st.pyplot(fig_cgpa)

    # --- Correlation heatmap ---
    st.subheader("Feature Correlation Heatmap")
    corr_df = df.drop(columns=["student_id"]).copy()
    corr_df["readiness_num"] = corr_df["readiness"].map({"Not Ready": 0, "Ready": 1})
    corr_df = corr_df.drop(columns=["readiness"])
    corr_matrix = corr_df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    cax = ax_corr.matshow(corr_matrix, cmap="coolwarm")
    fig_corr.colorbar(cax)
    ax_corr.set_xticks(range(len(corr_matrix.columns)))
    ax_corr.set_yticks(range(len(corr_matrix.columns)))
    ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, ha="left", fontsize=9)
    ax_corr.set_yticklabels(corr_matrix.columns, fontsize=9)
    ax_corr.set_title("Feature Correlation Matrix", pad=20)
    st.pyplot(fig_corr)

    # --- Feature importance from the trained model ---
    st.subheader("Model Feature Importance")
    if artifact:
        estimator = artifact.get("estimator")
        if estimator is None:
            # Try to extract from named pipeline steps
            try:
                steps = artifact["model"].named_steps
                estimator = steps.get("clf") or steps.get("rf")
            except Exception:
                estimator = None

        if estimator is not None and hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            feat_names  = artifact.get("feature_names", corr_matrix.columns.tolist())
            sorted_idx  = np.argsort(importances)
            fig_imp, ax_imp = plt.subplots()
            ax_imp.barh(
                [feat_names[i] for i in sorted_idx],
                importances[sorted_idx],
                color="#FF9800",
            )
            ax_imp.set_xlabel("Importance")
            ax_imp.set_title("Feature Importances (Best Model)")
            st.pyplot(fig_imp)
        else:
            st.info("Feature importances are not available for the selected model type.")
    else:
        st.info("Train the model first to see feature importance. Run `scripts/train_model.py`.")

# ══════════════════════════════════════════════════════════════════
# TAB 3 – Model Comparison
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Comparison")

    if os.path.exists(UPDATES_PATH):
        with open(UPDATES_PATH, "r") as f:
            updates = json.load(f)

        st.write(
            f"**Version**: {updates.get('version', 'N/A')}  |  "
            f"**Date**: {updates.get('date', 'N/A')}"
        )
        st.write(f"**Best Model**: `{updates.get('best_model', 'N/A')}`")

        models_data = updates.get("models_tested", {})
        if models_data:
            # Summary table
            comp_df = pd.DataFrame(models_data).T.reset_index()
            comp_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
            comp_df = comp_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
            st.dataframe(
                comp_df.style.format({
                    "Accuracy":  "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall":    "{:.4f}",
                    "F1 Score":  "{:.4f}",
                })
            )

            # Grouped bar chart for all four metrics
            metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
            x     = np.arange(len(comp_df))
            width = 0.2
            fig_cmp, ax_cmp = plt.subplots(figsize=(10, 5))
            for i, metric in enumerate(metrics):
                ax_cmp.bar(x + i * width, comp_df[metric], width, label=metric)
            ax_cmp.set_xticks(x + width * 1.5)
            ax_cmp.set_xticklabels(comp_df["Model"], fontsize=11)
            ax_cmp.set_ylim(0, 1.15)
            ax_cmp.set_ylabel("Score")
            ax_cmp.set_title("Model Performance Comparison")
            ax_cmp.legend()
            st.pyplot(fig_cmp)

            # Accuracy-only horizontal bar for quick ranking view
            st.subheader("Accuracy Ranking")
            best_model_name = updates.get("best_model", "")
            colors = [
                "#4CAF50" if m == best_model_name else "#2196F3"
                for m in comp_df["Model"]
            ]
            fig_acc, ax_acc = plt.subplots()
            ax_acc.barh(comp_df["Model"], comp_df["Accuracy"], color=colors)
            ax_acc.set_xlim(0, 1.1)
            ax_acc.set_xlabel("Accuracy")
            ax_acc.set_title("Model Accuracy Ranking  (green = best)")
            for i, v in enumerate(comp_df["Accuracy"]):
                ax_acc.text(v + 0.005, i, f"{v:.4f}", va="center")
            st.pyplot(fig_acc)
    else:
        st.info(
            "No model comparison data found. "
            "Run `scripts/train_model.py` to generate comparison results."
        )

# ══════════════════════════════════════════════════════════════════
# TAB 4 – Prediction
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Predict Placement Readiness")

    # Sidebar inputs (kept exactly as before)
    st.sidebar.header("Enter Student Details")
    cgpa          = st.sidebar.slider("CGPA", 5.0, 10.0, 7.0)
    aptitude      = st.sidebar.slider("Aptitude Score", 30, 100, 60)
    coding        = st.sidebar.slider("Coding Skill (1-10)", 1, 10, 6)
    communication = st.sidebar.slider("Communication Skill (1-10)", 1, 10, 6)
    mock          = st.sidebar.slider("Mock Interview Score", 20, 100, 60)
    internships   = st.sidebar.slider("Internships", 0, 3, 1)

    # Resolve model and feature metadata
    if artifact:
        model         = artifact["model"]
        features      = artifact["feature_names"]
        inv_label_map = {v: k for k, v in artifact.get("label_map", {}).items()}
    else:
        model         = None
        features      = [
            "cgpa",
            "aptitude_score",
            "coding_skill",
            "communication_skill",
            "mock_interview_score",
            "internships",
        ]
        inv_label_map = {1: "Ready", 0: "Not Ready"}

    if st.button("Predict Readiness"):
        row = {
            "cgpa":                  cgpa,
            "aptitude_score":        aptitude,
            "coding_skill":          coding,
            "communication_skill":   communication,
            "mock_interview_score":  mock,
            "internships":           internships,
        }
        X_pred = pd.DataFrame([row])[features]

        if model is None:
            st.error(
                "Model not available. Please run `scripts/train_model.py` first."
            )
        else:
            pred  = model.predict(X_pred)[0]
            label = inv_label_map.get(pred, str(pred))
            if label == "Ready":
                st.success("You are placement ready.")
            else:
                st.error("Not ready yet. Needs improvement.")

st.caption("Model predictions are indicative and based on synthetic data.")





