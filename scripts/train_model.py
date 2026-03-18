import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ------------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)


df = pd.read_csv(project_path("data", "readiness_data.csv"))

X = df.drop(["student_id", "readiness"], axis=1)
feature_names = X.columns.tolist()
y = df["readiness"].map({"Not Ready": 0, "Ready": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------
# Define candidate models wrapped in pipelines with standard scaling
# ------------------------------------------------------------------
model_candidates = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ]),
    "DecisionTree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ]),
}

# ------------------------------------------------------------------
# Train every candidate and record performance metrics
# ------------------------------------------------------------------
results = {}
for name, pipeline in model_candidates.items():
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall":    float(recall_score(y_test, pred, zero_division=0)),
        "f1_score":  float(f1_score(y_test, pred, zero_division=0)),
    }
    results[name] = metrics

    print(f"\n=== {name} ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print(classification_report(y_test, pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# ------------------------------------------------------------------
# Select the best model by accuracy
# ------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_pipeline = model_candidates[best_name]
print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")

# ------------------------------------------------------------------
# Package and save the best model
# ------------------------------------------------------------------
os.makedirs(project_path("models"), exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Extract raw estimator for feature importance plots in the dashboard
try:
    estimator = best_pipeline.named_steps["clf"]
except Exception:
    estimator = None

artifact = {
    "model": best_pipeline,
    "estimator": estimator,
    "feature_names": feature_names,
    "label_map": {"Not Ready": 0, "Ready": 1},
    "timestamp": timestamp,
    "model_name": best_name,
}

# Primary path used by the dashboard
best_model_path = project_path("models", "readiness_model.pkl")
with open(best_model_path, "wb") as f:
    pickle.dump(artifact, f)

# Legacy path kept for backward compatibility
latest_model_path = project_path("models", "readiness_model_latest.pkl")
with open(latest_model_path, "wb") as f:
    pickle.dump(artifact, f)

# Timestamped archive copy
model_path = project_path("models", f"readiness_model_{timestamp}.pkl")
with open(model_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"\nBest model saved to  : {best_model_path}")
print(f"Legacy path updated  : {latest_model_path}")
print(f"Timestamped archive  : {model_path}")

# ------------------------------------------------------------------
# Save model_updates.json – comparison report consumed by dashboard
# ------------------------------------------------------------------
today = datetime.now().strftime("%Y-%m-%d")
model_updates = {
    "version": "v1.1",
    "date": today,
    "models_tested": results,
    "best_model": best_name,
}
updates_path = project_path("models", "model_updates.json")
with open(updates_path, "w") as f:
    json.dump(model_updates, f, indent=2)
print(f"Model comparison report saved to {updates_path}")

# ------------------------------------------------------------------
# Save per-run metadata JSON (legacy format used by dashboard expander)
# ------------------------------------------------------------------
meta_path = project_path("models", f"readiness_model_{timestamp}.json")
metadata = {
    "timestamp": timestamp,
    "accuracy": results[best_name]["accuracy"],
    "feature_names": feature_names,
    "label_map": {"Not Ready": 0, "Ready": 1},
    "confusion_matrix": confusion_matrix(y_test, best_pipeline.predict(X_test)).tolist(),
}
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to {meta_path}")
