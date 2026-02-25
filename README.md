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
