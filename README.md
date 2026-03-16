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

---

## What Has Been Improved (v1.1)

| Area | Change |
|---|---|
| **Data generation** | Switched from strict AND-rule (~5% Ready) to score-based rule (≥3/5 conditions) — gives a balanced ~56/44 split |
| **Model training** | Now trains and compares three models: Logistic Regression, Decision Tree, Random Forest |
| **Metrics tracked** | Accuracy, Precision, Recall, F1 Score for every model |
| **Best model selection** | Automatically selects the highest-accuracy model and saves it as `models/readiness_model.pkl` |
| **Model comparison report** | Written to `models/model_updates.json` after every training run |
| **Dashboard — Insights tab** | Added readiness distribution bar chart, CGPA boxplot, aptitude histogram, colour-mapped correlation heatmap, feature importance chart |
| **Dashboard — Model Comparison tab** | Grouped bar chart (all 4 metrics) + accuracy ranking chart, reads live from `model_updates.json` |
| **Dashboard — Prediction tab** | Output now shows ✅ Placement Ready / ❌ Not Ready clearly |

---

## Model Update & Retraining Timeline (Next 6 Weeks)

This section addresses the professor's feedback:  
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

### Retraining Trigger Rules

Outside the scheduled weeks, retrain the model immediately if any of these occur:

| Trigger | Action |
|---|---|
| Accuracy drops below **85%** on test set | Investigate data quality, retune and retrain |
| A new feature is added to the dataset | Retrain all models from scratch |
| Dataset size grows beyond **15,000 rows** | Retrain and re-evaluate cross-validation scores |
| A new model type is added | Full comparison run, update `model_updates.json` |
