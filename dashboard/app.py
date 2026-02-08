import os
import json
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Placement Readiness Analytics", layout="wide")

st.title("Placement Readiness Analytics Dashboard")

# Load data
df = pd.read_csv(os.path.join("data", "readiness_data.csv"))

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
col2.metric("Ready %", round((df["readiness"].value_counts(normalize=True).get("Ready", 0)) * 100, 2))
col3.metric("Not Ready %", round((df["readiness"].value_counts(normalize=True).get("Not Ready", 0)) * 100, 2))

st.divider()

# Load and display model metadata
latest_model_path = os.path.join("models", "readiness_model_latest.pkl")
model_meta = None
if os.path.exists(latest_model_path):
    # Try to find and load the corresponding metadata JSON
    import glob
    meta_files = sorted(glob.glob(os.path.join("models", "readiness_model_*.json")), reverse=True)
    if meta_files:
        try:
            with open(meta_files[0], "r") as f:
                model_meta = json.load(f)
            with st.expander("📊 Model Information"):
                if model_meta:
                    st.write(f"**Timestamp**: {model_meta.get('timestamp', 'N/A')}")
                    if "accuracy" in model_meta:
                        st.write(f"**Accuracy**: {model_meta['accuracy']:.4f}")
        except Exception:
            pass

# Tabs for clean UI
tab1, tab2, tab3 = st.tabs(["Data Overview", "Insights", "Prediction"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.sample(10))

with tab2:
    st.subheader("Feature Distributions")

    fig1 = plt.figure()
    df["cgpa"].hist()
    st.pyplot(fig1)

    fig2 = plt.figure()
    df["aptitude_score"].hist()
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap (Matplotlib)")

    corr = df.drop(columns=["student_id"]).copy()
    corr["readiness_num"] = corr["readiness"].map({"Not Ready": 0, "Ready": 1})
    corr = corr.drop(columns=["readiness"])

    corr_matrix = corr.corr()

    fig3, ax = plt.subplots()
    cax = ax.matshow(corr_matrix)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    fig3.colorbar(cax)
    st.pyplot(fig3)

    # Feature Importance from packaged model
    try:
        artifact = pickle.load(open(latest_model_path, "rb"))
        model = artifact["model"]
        estimator = artifact.get("estimator")
        if estimator is None:
            try:
                estimator = model.named_steps.get("rf")
            except Exception:
                estimator = None

        if estimator is not None and hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            features = artifact.get("feature_names", corr_matrix.columns)

            fig4 = plt.figure()
            plt.barh(features, importances)
            st.subheader("Model Feature Importance")
            st.pyplot(fig4)
        else:
            st.info("Feature importances not available for the current model.")
    except Exception:
        st.info("Train the model to see feature importance.")

with tab3:
    st.subheader("Predict Placement Readiness")

    st.sidebar.header("Enter Student Details")

    cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, 7.0)
    aptitude = st.sidebar.slider("Aptitude Score", 30, 100, 60)
    coding = st.sidebar.slider("Coding Skill (1-10)", 1, 10, 6)
    communication = st.sidebar.slider("Communication Skill (1-10)", 1, 10, 6)
    mock = st.sidebar.slider("Mock Interview Score", 20, 100, 60)
    internships = st.sidebar.slider("Internships", 0, 3, 1)

    try:
        artifact = pickle.load(open(latest_model_path, "rb"))
        model = artifact["model"]
        features = artifact["feature_names"]
        inv_label_map = {v: k for k, v in artifact.get("label_map", {}).items()}
    except Exception:
        model = None
        features = [
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
            "cgpa": cgpa,
            "aptitude_score": aptitude,
            "coding_skill": coding,
            "communication_skill": communication,
            "mock_interview_score": mock,
            "internships": internships,
        }

        X = pd.DataFrame([row])[features]

        if model is None:
            st.error("Model not available. Train the model first.")
        else:
            pred = model.predict(X)[0]
            label = inv_label_map.get(pred, str(pred))
            if label == "Ready":
                st.success("Placement Ready")
            else:
                st.error("Not Ready – Needs Improvement")

st.caption("Model predictions are indicative and based on synthetic data.")

