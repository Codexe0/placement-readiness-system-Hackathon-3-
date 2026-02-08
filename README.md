# Placement Readiness Assessment — Project README

This repository contains a simple end-to-end placement readiness demo: synthetic data generation, model training (with preprocessing pipeline), and a Streamlit dashboard for inspection and prediction.

Quick commands (Windows PowerShell)

1. Create virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Generate synthetic data

```powershell
python scripts\data_generator.py
```

3. Train model (creates `models/readiness_model.pkl`)

```powershell
python scripts\train_model.py
```

4. Run a sample prediction

```powershell
python scripts\predict.py
```

5. Run the Streamlit dashboard

```powershell
streamlit run dashboard\app.py
```

Notes
- The training script now builds a small preprocessing pipeline (`StandardScaler` + `RandomForestClassifier`) and saves a packaged artifact with keys: `model` (pipeline), `estimator` (raw RandomForest), `feature_names`, and `label_map`.
- The dashboard will use the packaged artifact for predictions and will display feature importances if available.
- For reproducibility consider pinning dependency versions in `requirements.txt`.

If you want, I can add model versioning (timestamped artifacts) and a small CI test next.
