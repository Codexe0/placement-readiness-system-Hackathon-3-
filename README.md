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


Aim & Purpose – Placement Readiness Assessment System
Aim
The primary aim of this project is to design and deploy a machine learning–based system that assesses a student's readiness for campus placements. The system predicts whether a student is “Placement Ready” or “Not Ready” based on academic performance and employability-related skills. The goal is to move beyond notebook-based experimentation and create a usable, production-like pipeline with data storage, model lifecycle management, and an interactive dashboard.

Purpose
The purpose of this project is to provide universities and training cells with a data-driven tool to:

Identify students who are ready for placements
Detect students who require additional training or mentoring
Support early interventions to improve overall placement outcomes
Visualize key readiness indicators and model insights
Enable continuous improvement of predictions as new data becomes available
What We Are Trying To Do
This project attempts to bridge the gap between building a machine learning model and deploying it as a usable application. Specifically, we aim to:

Collect and store student readiness data in a structured SQL database
Train traditional machine learning models to predict placement readiness
Compare multiple models and select the best-performing one
Serialize and deploy the trained model for real-time predictions
Build an interactive dashboard that allows users to:
View data insights
Analyze feature distributions and correlations
Input student details and obtain readiness predictions
What We Want To Achieve
By the end of this project, we aim to achieve:

A production-ready ML pipeline that can:
Ingest new data
Retrain models periodically
Maintain model versions
A user-friendly dashboard for non-technical stakeholders
A version-controlled codebase demonstrating:
Data engineering practices
Model lifecycle management
Deployment readiness
A scalable framework that can be extended to:
Placement probability prediction
Skill gap analysis
Personalized training recommendations
Scope of the Project
University-focused predictive analytics
Traditional machine learning models (no neural networks)
SQL-based data storage
Streamlit-based web dashboard
Periodic model retraining and performance tracking
Public version control repository for development history
Expected Impact
Improved understanding of placement readiness factors
Better decision-making for placement training programs
Early identification of students needing support
Enhanced transparency of model behavior through analytics
Demonstration of end-to-end ML deployment engineering
Future Enhancements
Integration with real university placement data
Role-based dashboards for students and placement officers
Recommendation system for skill improvement
Automated data pipelines and scheduled retraining
Model performance monitoring and drift detection
