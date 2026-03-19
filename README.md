<div align="center">

# 🎓 Placement Readiness Assessment System

### *ML-Powered Campus Placement Prediction & Analytics*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.43-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-6.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

---

An end-to-end machine learning system that predicts whether a student is **Placement Ready** or **Not Ready** based on academic performance and employability skills — complete with a **premium interactive Streamlit dashboard**.

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Project Structure](#️-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📊 Dashboard](#-dashboard)
- [🤖 Models & Metrics](#-models--metrics)
- [🔄 What's New in v1.1](#-whats-new-in-v11)
- [📅 Model Update & Retraining Timeline](#-model-update--retraining-timeline-next-6-weeks)
- [⚡ Retraining Trigger Rules](#-retraining-trigger-rules)
- [🛠️ Tech Stack](#️-tech-stack)

---

## ✨ Features

| Feature | Description |
|:---|:---|
| 🧪 **Synthetic Data Generation** | Balanced dataset with score-based readiness rules (~56/44 split) |
| 🤖 **Multi-Model Training** | Trains & compares Logistic Regression, Decision Tree, and Random Forest |
| 📊 **Interactive Dashboard** | Dark-themed Streamlit app with Plotly charts, glassmorphism UI, and live predictions |
| 🏆 **Auto Best Model Selection** | Selects highest-accuracy model and saves it automatically |
| 📈 **Rich Analytics** | Correlation heatmaps, feature importance, readiness distributions, boxplots |
| 🔮 **Real-time Predictions** | Input student details → get ML-powered readiness prediction with confidence % |
| 📋 **Model Versioning** | Every training run is archived with metrics in `model_updates.json` |

---

## 🏗️ Project Structure

```
Hackathon 3/
├── 📂 dashboard/
│   └── app.py                  # Streamlit dashboard (premium UI)
├── 📂 data/
│   └── readiness_data.csv      # Synthetic student dataset
├── 📂 database/
│   └── db_connection.py        # SQL database connection utility
├── 📂 models/
│   ├── readiness_model.pkl     # Best trained model (primary)
│   ├── model_updates.json      # Model comparison report
│   └── *.pkl / *.json          # Timestamped model archives
├── 📂 scripts/
│   ├── data_generator.py       # Synthetic data generation
│   ├── train_model.py          # Multi-model training pipeline
│   ├── predict.py              # CLI prediction script
│   └── retrain_model.py        # Model retraining utility
├── requirements.txt
├── AIM_OR_PURPOSE.md
└── README.md
```

---

## 🚀 Quick Start

> **Prerequisites:** Python 3.10+ on Windows PowerShell

**0. Move into the project folder**

```powershell
cd "Hackathon 3"
```

**1. Create virtual environment & install dependencies**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

<details>
<summary>💡 Optional: Activate the virtual environment in PowerShell</summary>

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

</details>

**2. Generate synthetic data**

```powershell
.\.venv\Scripts\python.exe scripts\data_generator.py
```

**3. Train model** *(creates `models/readiness_model.pkl`)*

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py
```

**4. Run a sample prediction**

```powershell
.\.venv\Scripts\python.exe scripts\predict.py
```

**5. Launch the dashboard** 🚀

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

---

## 📊 Dashboard

The Streamlit dashboard features a **premium dark-themed UI** with glassmorphism styling, interactive Plotly charts, and a modern layout.

### Dashboard Tabs

| Tab | What It Shows |
|:---|:---|
| 📋 **Data Overview** | Dataset preview, class distribution chart |
| 📈 **Insights** | Readiness distribution, CGPA boxplots, aptitude & CGPA histograms, correlation heatmap, feature importance |
| ⚡ **Model Comparison** | Side-by-side metrics (Accuracy, Precision, Recall, F1) for all models + accuracy ranking |
| 🔮 **Prediction** | Input student details → ML prediction with confidence score and animated result card |

### UI Highlights

- 🌑 **Dark gradient theme** with frosted-glass card effects
- 📊 **Interactive Plotly charts** — hover, zoom, pan
- 🏆 **4 KPI metric cards** — Total Students, Ready %, Not Ready %, Avg CGPA
- 🎯 **Confidence percentage** on predictions via `predict_proba`
- ✨ **Micro-animations** on hover and result display
- 🎨 **Google Fonts (Inter)** for clean typography

---

## 🤖 Models & Metrics

Three models are trained and compared on every run:

| Model | Pipeline |
|:---|:---|
| **Logistic Regression** | StandardScaler → LogisticRegression (max_iter=500) |
| **Decision Tree** | StandardScaler → DecisionTreeClassifier |
| **Random Forest** | StandardScaler → RandomForestClassifier |

**Metrics tracked per model:** Accuracy · Precision · Recall · F1 Score

The best model (by accuracy) is automatically saved to `models/readiness_model.pkl` and consumed by the dashboard.

### Input Features

| Feature | Range | Description |
|:---|:---|:---|
| `cgpa` | 5.0 – 10.0 | Cumulative Grade Point Average |
| `aptitude_score` | 30 – 100 | Aptitude test score |
| `coding_skill` | 1 – 10 | Self-rated coding proficiency |
| `communication_skill` | 1 – 10 | Self-rated communication ability |
| `mock_interview_score` | 20 – 100 | Mock interview performance |
| `internships` | 0 – 3 | Number of internships completed |

---

## 🔄 What's New in v1.1

| Area | Change |
|:---|:---|
| **Data generation** | Switched from strict AND-rule (~5% Ready) to score-based rule (≥3/5 conditions) — gives a balanced ~56/44 split |
| **Model training** | Now trains and compares three models: Logistic Regression, Decision Tree, Random Forest |
| **Metrics tracked** | Accuracy, Precision, Recall, F1 Score for every model |
| **Best model selection** | Automatically selects the highest-accuracy model and saves it as `models/readiness_model.pkl` |
| **Model comparison report** | Written to `models/model_updates.json` after every training run |
| **Dashboard — Insights tab** | Added readiness distribution bar chart, CGPA boxplot, aptitude histogram, colour-mapped correlation heatmap, feature importance chart |
| **Dashboard — Model Comparison tab** | Grouped bar chart (all 4 metrics) + accuracy ranking chart, reads live from `model_updates.json` |
| **Dashboard — Prediction tab** | Output now shows ✅ Placement Ready / ❌ Not Ready with confidence percentage |
| **Dashboard — UI** | Premium dark theme with glassmorphism, Plotly interactive charts, animated result cards |

---

## 📅 Model Update & Retraining Timeline (Next 6 Weeks)

> This section addresses the professor's feedback:  
> *"There needs to be clear goals about updates and retrainings of the model, with a timeline, for the next four to six weeks."*

### Week 1 (Mar 17 – Mar 23) — Baseline Stabilisation
**Goal:** Lock in a clean, reproducible baseline before any further experiments.
- [ ] Review current model metrics (LR: 86.5%, DT: 99.95%, RF: 100%) and confirm they are realistic on the balanced dataset
- [ ] Add a `scripts/evaluate_model.py` script that loads the saved model and prints metrics without retraining
- [ ] Document baseline results in `model_updates.json` as version `v1.1`
- [ ] Confirm the dashboard loads correctly and all charts render after the v1.1 changes

### Week 2 (Mar 24 – Mar 30) — Feature Engineering
**Goal:** Improve the quality of input features to give models more signal.
- [ ] Analyse feature importances from the Random Forest — identify weak features
- [ ] Add at least one engineered feature (e.g. a composite `overall_score = weighted average of all five inputs`)
- [ ] Regenerate dataset with the new feature included
- [ ] Retrain all three models, compare against Week 1 baseline
- [ ] Save results as version `v1.2` in `model_updates.json`

### Week 3 (Mar 31 – Apr 6) — Hyperparameter Tuning
**Goal:** Squeeze genuine performance out of the models using systematic tuning.
- [ ] Apply `GridSearchCV` or `RandomizedSearchCV` to Random Forest (tune `n_estimators`, `max_depth`, `min_samples_split`)
- [ ] Apply tuning to Decision Tree (`max_depth`, `min_samples_leaf`)
- [ ] Retrain with best hyperparameters found
- [ ] Log best params alongside metrics in `model_updates.json`
- [ ] Save as version `v1.3`

### Week 4 (Apr 7 – Apr 13) — Model Expansion & Cross-Validation
**Goal:** Add a fourth model and switch to robust k-fold evaluation.
- [ ] Add **Gradient Boosting** (or XGBoost if installed) as a fourth candidate
- [ ] Replace single train/test split with **5-fold cross-validation** — report mean ± std for each metric
- [ ] Update `train_model.py` and `model_updates.json` schema to store CV scores
- [ ] Update the dashboard Model Comparison tab to display CV mean ± std
- [ ] Save as version `v1.4`

### Week 5 (Apr 14 – Apr 20) — Threshold & Bias Analysis
**Goal:** Ensure the model is fair and the decision boundary is well-calibrated.
- [ ] Plot ROC curves for all models in the Insights tab
- [ ] Experiment with classification threshold (default 0.5) — check if 0.4 or 0.6 gives better recall for "Ready" students
- [ ] Check for feature bias: do students with high CGPA alone score Ready even if other features are low?
- [ ] Document findings and adjust data generation rules if any bias is found
- [ ] Retrain and save as version `v1.5`

### Week 6 (Apr 21 – Apr 27) — Final Review & Presentation-Ready Build
**Goal:** Polish the system into a submission-quality state.
- [ ] Final retraining on the complete cleaned dataset — save as version `v2.0`
- [ ] Update `model_updates.json` to `"version": "v2.0"` with all final metrics
- [ ] Clean up all temporary/timestamped `.pkl` files from the `models/` directory
- [ ] Ensure the dashboard works end-to-end from a fresh clone (no pre-existing model files)
- [ ] Update this README with final accuracy numbers and a summary table
- [ ] Prepare a 2-minute demo walkthrough of the dashboard for the presentation

---

## ⚡ Retraining Trigger Rules

Outside the scheduled weeks, retrain the model immediately if any of these occur:

| Trigger | Action |
|:---|:---|
| Accuracy drops below **85%** on test set | Investigate data quality, retune and retrain |
| A new feature is added to the dataset | Retrain all models from scratch |
| Dataset size grows beyond **15,000 rows** | Retrain and re-evaluate cross-validation scores |
| A new model type is added | Full comparison run, update `model_updates.json` |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|:---|:---|
| **Python 3.10+** | Core language |
| **pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **scikit-learn** | Model training, pipelines, evaluation |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive visualisations |
| **matplotlib** | Static chart fallback |
| **SQL** | Data storage layer |

---

<div align="center">

**Made with ❤️ for Hackathon 3**

*Placement Readiness Assessment System · v1.1*

</div>
