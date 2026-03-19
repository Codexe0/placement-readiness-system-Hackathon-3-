import os
import json
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Placement Readiness Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)


# ------------------------------------------------------------------
# Custom CSS – dark premium theme with glassmorphism
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---------- Global ---------- */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* ---------- Hide default Streamlit decorations ---------- */
#MainMenu, header, footer {visibility: hidden;}

/* ---------- Hero title ---------- */
.hero-title {
    text-align: center;
    padding: 1.2rem 0 0.2rem;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6fd8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.hero-sub {
    text-align: center;
    color: #9ca3af;
    font-size: 1rem;
    margin-bottom: 1.6rem;
    font-weight: 400;
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 0.5rem;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(123,47,247,0.18);
}
.metric-icon {
    font-size: 2rem;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-label {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-top: 0.15rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.accent-green  { border-bottom: 3px solid #22c55e; }
.accent-red    { border-bottom: 3px solid #ef4444; }
.accent-blue   { border-bottom: 3px solid #3b82f6; }
.accent-purple { border-bottom: 3px solid #a855f7; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #9ca3af;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
    color: #fff !important;
    border-radius: 10px;
}

/* ---------- Section headers ---------- */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid rgba(123,47,247,0.4);
    display: inline-block;
}

/* ---------- Prediction result cards ---------- */
.result-card {
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
    animation: fadeSlideIn 0.5s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-ready {
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.06));
    border: 1px solid rgba(34,197,94,0.35);
}
.result-not-ready {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.06));
    border: 1px solid rgba(239,68,68,0.35);
}
.result-icon  { font-size: 3.5rem; }
.result-label { font-size: 1.6rem; font-weight: 700; color: #fff; margin-top: 0.4rem; }
.result-desc  { font-size: 0.95rem; color: #9ca3af; margin-top: 0.3rem; }

/* ---------- Info badge ---------- */
.info-badge {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    margin: 0.2rem;
    color: #cbd5e1;
    font-size: 0.85rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e, #0f0c29);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #9ca3af;
}

/* ---------- Plotly chart background ---------- */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ---------- Dataframe styling ---------- */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Expander ---------- */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2.4rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(123,47,247,0.3) !important;
}

/* ---------- Slider ---------- */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #7b2ff7, #00d2ff) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Plotly template (dark, transparent background)
# ------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=30, t=50, b=40),
)

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------
df = pd.read_csv(project_path("data", "readiness_data.csv"))

# ------------------------------------------------------------------
# Hero title
# ------------------------------------------------------------------
st.markdown('<p class="hero-title">🎓 Placement Readiness Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Interactive ML-powered insights for campus placement readiness</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# KPI metric cards
# ------------------------------------------------------------------
total = len(df)
ready_pct = round(df["readiness"].value_counts(normalize=True).get("Ready", 0) * 100, 1)
not_ready_pct = round(df["readiness"].value_counts(normalize=True).get("Not Ready", 0) * 100, 1)
avg_cgpa = round(df["cgpa"].mean(), 2)

c1, c2, c3, c4 = st.columns(4)

for col, icon, value, label, accent in [
    (c1, "👥", f"{total:,}",       "Total Students",  "accent-blue"),
    (c2, "✅", f"{ready_pct}%",    "Ready",           "accent-green"),
    (c3, "❌", f"{not_ready_pct}%","Not Ready",       "accent-red"),
    (c4, "📊", str(avg_cgpa),      "Avg CGPA",        "accent-purple"),
]:
    col.markdown(
        f"""
        <div class="metric-card {accent}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model artifact
# ------------------------------------------------------------------
MODEL_PATH = project_path("models", "readiness_model.pkl")
LEGACY_PATH = project_path("models", "readiness_model_latest.pkl")
UPDATES_PATH = project_path("models", "model_updates.json")


def _load_artifact(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


artifact = _load_artifact(MODEL_PATH) or _load_artifact(LEGACY_PATH)

# Model info badges
meta_files = sorted(glob.glob(str(project_path("models", "readiness_model_*.json"))), reverse=True)
if meta_files:
    try:
        with open(meta_files[0], "r") as f:
            model_meta = json.load(f)
        with st.expander("📋 Model Information"):
            badges = ""
            ts = model_meta.get("timestamp", "N/A")
            badges += f'<span class="info-badge">🕐 Trained: {ts}</span>'
            if "accuracy" in model_meta:
                badges += f'<span class="info-badge">🎯 Accuracy: {model_meta["accuracy"]:.4f}</span>'
            if artifact and artifact.get("model_name"):
                badges += f'<span class="info-badge">🏆 Best: {artifact["model_name"]}</span>'
            st.markdown(badges, unsafe_allow_html=True)
    except Exception:
        pass

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Data Overview", "📈 Insights", "⚡ Model Comparison", "🔮 Prediction"]
)

# ══════════════════════════════════════════════════════════════════
# TAB 1 – Data Overview
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.sample(min(10, len(df))), use_container_width=True)

    st.markdown('<p class="section-header">Class Distribution</p>', unsafe_allow_html=True)
    dist = df["readiness"].value_counts().reset_index()
    dist.columns = ["Readiness", "Count"]
    fig_dist = px.bar(
        dist,
        x="Readiness",
        y="Count",
        color="Readiness",
        color_discrete_map={"Ready": "#22c55e", "Not Ready": "#ef4444"},
        text="Count",
    )
    fig_dist.update_layout(**PLOTLY_LAYOUT, showlegend=False)
    fig_dist.update_traces(
        textposition="outside",
        marker_line_width=0,
        opacity=0.92,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 – Insights
# ══════════════════════════════════════════════════════════════════
with tab2:
    # Row 1: Readiness distribution + CGPA boxplot (side by side)
    left, right = st.columns(2)

    with left:
        st.markdown('<p class="section-header">Readiness Distribution</p>', unsafe_allow_html=True)
        dist_counts = df["readiness"].value_counts().reset_index()
        dist_counts.columns = ["Readiness", "Count"]
        fig_rd = px.bar(
            dist_counts,
            x="Readiness",
            y="Count",
            color="Readiness",
            color_discrete_map={"Ready": "#22c55e", "Not Ready": "#ef4444"},
            text="Count",
        )
        fig_rd.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Ready vs Not Ready")
        fig_rd.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_rd, use_container_width=True)

    with right:
        st.markdown('<p class="section-header">CGPA by Readiness</p>', unsafe_allow_html=True)
        fig_box = px.box(
            df,
            x="readiness",
            y="cgpa",
            color="readiness",
            color_discrete_map={"Ready": "#22c55e", "Not Ready": "#ef4444"},
            points="outliers",
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="CGPA Distribution by Readiness")
        st.plotly_chart(fig_box, use_container_width=True)

    # Row 2: Aptitude Histogram + CGPA Histogram
    left2, right2 = st.columns(2)

    with left2:
        st.markdown('<p class="section-header">Aptitude Score Distribution</p>', unsafe_allow_html=True)
        fig_apt = px.histogram(
            df,
            x="aptitude_score",
            nbins=25,
            color_discrete_sequence=["#3b82f6"],
        )
        fig_apt.update_layout(**PLOTLY_LAYOUT, title="Aptitude Scores", bargap=0.05)
        st.plotly_chart(fig_apt, use_container_width=True)

    with right2:
        st.markdown('<p class="section-header">CGPA Distribution</p>', unsafe_allow_html=True)
        fig_cgpa = px.histogram(
            df,
            x="cgpa",
            nbins=25,
            color_discrete_sequence=["#a855f7"],
        )
        fig_cgpa.update_layout(**PLOTLY_LAYOUT, title="CGPA Scores", bargap=0.05)
        st.plotly_chart(fig_cgpa, use_container_width=True)

    # Row 3: Correlation Heatmap (full width)
    st.markdown('<p class="section-header">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    corr_df = df.drop(columns=["student_id"]).copy()
    corr_df["readiness_num"] = corr_df["readiness"].map({"Not Ready": 0, "Ready": 1})
    corr_df = corr_df.drop(columns=["readiness"])
    corr_matrix = corr_df.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig_corr.update_layout(
        **PLOTLY_LAYOUT,
        title="Feature Correlation Matrix",
        height=520,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Row 4: Feature Importance
    st.markdown('<p class="section-header">Model Feature Importance</p>', unsafe_allow_html=True)
    if artifact:
        estimator = artifact.get("estimator")
        if estimator is None:
            try:
                steps = artifact["model"].named_steps
                estimator = steps.get("clf") or steps.get("rf")
            except Exception:
                estimator = None

        if estimator is not None and hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            feat_names = artifact.get("feature_names", corr_matrix.columns.tolist())
            sorted_idx = np.argsort(importances)

            fig_imp = go.Figure(
                go.Bar(
                    x=[importances[i] for i in sorted_idx],
                    y=[feat_names[i] for i in sorted_idx],
                    orientation="h",
                    marker=dict(
                        color=[importances[i] for i in sorted_idx],
                        colorscale=[[0, "#7b2ff7"], [1, "#00d2ff"]],
                        line_width=0,
                    ),
                )
            )
            fig_imp.update_layout(
                **PLOTLY_LAYOUT,
                title="Feature Importances (Best Model)",
                xaxis_title="Importance",
                height=380,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importances are not available for the selected model type.")
    else:
        st.info("Train the model first to see feature importance. Run `scripts/train_model.py`.")

# ══════════════════════════════════════════════════════════════════
# TAB 3 – Model Comparison
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Model Performance Comparison</p>', unsafe_allow_html=True)

    if os.path.exists(UPDATES_PATH):
        with open(UPDATES_PATH, "r") as f:
            updates = json.load(f)

        # Info badges for metadata
        ver = updates.get("version", "N/A")
        dt = updates.get("date", "N/A")
        bm = updates.get("best_model", "N/A")
        st.markdown(
            f'<span class="info-badge">📌 Version: {ver}</span>'
            f'<span class="info-badge">📅 Date: {dt}</span>'
            f'<span class="info-badge">🏆 Best: {bm}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        models_data = updates.get("models_tested", {})
        if models_data:
            # Summary table
            comp_df = pd.DataFrame(models_data).T.reset_index()
            comp_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
            comp_df = comp_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
            st.dataframe(
                comp_df.style.format(
                    {
                        "Accuracy": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1 Score": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Grouped bar chart
            metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
            metric_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
            fig_cmp = go.Figure()
            for metric, color in zip(metrics, metric_colors):
                fig_cmp.add_trace(
                    go.Bar(
                        name=metric,
                        x=comp_df["Model"],
                        y=comp_df[metric],
                        marker_color=color,
                        text=comp_df[metric].apply(lambda v: f"{v:.3f}"),
                        textposition="outside",
                    )
                )
            fig_cmp.update_layout(
                **PLOTLY_LAYOUT,
                barmode="group",
                title="All Metrics — Side by Side",
                yaxis=dict(range=[0, 1.15]),
                height=470,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Accuracy ranking
            st.markdown('<p class="section-header">Accuracy Ranking</p>', unsafe_allow_html=True)
            best_model_name = updates.get("best_model", "")
            colors = [
                "#22c55e" if m == best_model_name else "#3b82f6"
                for m in comp_df["Model"]
            ]
            fig_acc = go.Figure(
                go.Bar(
                    x=comp_df["Accuracy"],
                    y=comp_df["Model"],
                    orientation="h",
                    marker_color=colors,
                    text=comp_df["Accuracy"].apply(lambda v: f"{v:.4f}"),
                    textposition="outside",
                )
            )
            fig_acc.update_layout(
                **PLOTLY_LAYOUT,
                title="Model Accuracy Ranking (green = best)",
                xaxis=dict(range=[0, 1.12]),
                height=350,
            )
            st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info(
            "No model comparison data found. "
            "Run `scripts/train_model.py` to generate comparison results."
        )

# ══════════════════════════════════════════════════════════════════
# TAB 4 – Prediction
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Predict Placement Readiness</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#9ca3af;margin-bottom:1rem;">Fill in the student details below and hit <b>Predict</b> to get an ML-powered readiness assessment.</p>',
        unsafe_allow_html=True,
    )

    # Input form — 2-column layout inside the main area
    col_l, col_r = st.columns(2)
    with col_l:
        cgpa = st.slider("📚 CGPA", 5.0, 10.0, 7.0, step=0.1)
        aptitude = st.slider("🧠 Aptitude Score", 30, 100, 60)
        coding = st.slider("💻 Coding Skill (1–10)", 1, 10, 6)
    with col_r:
        communication = st.slider("🗣️ Communication Skill (1–10)", 1, 10, 6)
        mock = st.slider("🎤 Mock Interview Score", 20, 100, 60)
        internships = st.slider("💼 Internships", 0, 3, 1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Resolve model and feature metadata
    if artifact:
        model = artifact["model"]
        features = artifact["feature_names"]
        inv_label_map = {v: k for k, v in artifact.get("label_map", {}).items()}
    else:
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

    # Center the predict button
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        predict_btn = st.button("🚀  Predict Readiness", use_container_width=True)

    if predict_btn:
        row = {
            "cgpa": cgpa,
            "aptitude_score": aptitude,
            "coding_skill": coding,
            "communication_skill": communication,
            "mock_interview_score": mock,
            "internships": internships,
        }
        X_pred = pd.DataFrame([row])[features]

        if model is None:
            st.error("Model not available. Please run `scripts/train_model.py` first.")
        else:
            pred = model.predict(X_pred)[0]
            label = inv_label_map.get(pred, str(pred))

            # Attempt to get confidence from predict_proba
            confidence_str = ""
            try:
                proba = model.predict_proba(X_pred)[0]
                confidence = max(proba) * 100
                confidence_str = f'<div class="result-desc">Confidence: {confidence:.1f}%</div>'
            except Exception:
                pass

            if label == "Ready":
                st.markdown(
                    f"""
                    <div class="result-card result-ready">
                        <div class="result-icon">✅</div>
                        <div class="result-label">Placement Ready!</div>
                        <div class="result-desc">This student meets the readiness criteria.</div>
                        {confidence_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card result-not-ready">
                        <div class="result-icon">❌</div>
                        <div class="result-label">Not Ready Yet</div>
                        <div class="result-desc">Needs improvement in key areas before placement.</div>
                        {confidence_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ------------------------------------------------------------------
# Sidebar – About section
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 About")
    st.markdown(
        """
        **Placement Readiness Analytics**  
        An ML-powered dashboard to assess student
        readiness for campus placements.

        **Models**: Logistic Regression, Decision Tree, Random Forest  
        **Stack**: Python · Streamlit · Plotly · scikit-learn

        ---
        *Version 1.1 · Hackathon 3*
        """,
    )

# Footer
st.markdown(
    '<div style="text-align:center;color:#4b5563;margin-top:2rem;font-size:0.8rem;">'
    "Model predictions are indicative and based on synthetic data."
    "</div>",
    unsafe_allow_html=True,
)
