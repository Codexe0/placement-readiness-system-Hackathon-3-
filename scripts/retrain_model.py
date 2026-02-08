import os
import json
import pickle
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(os.path.join("data", "readiness_data.csv"))
X = df.drop(["student_id", "readiness"], axis=1)
feature_names = X.columns.tolist()
y = df["readiness"].map({"Not Ready": 0, "Ready": 1})

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42)),
    ]
)

pipeline.fit(X, y)
estimator = pipeline.named_steps["rf"]

# Create timestamped artifact filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join("models", f"readiness_model_{timestamp}.pkl")
meta_path = os.path.join("models", f"readiness_model_{timestamp}.json")
latest_model_path = os.path.join("models", "readiness_model_latest.pkl")

artifact = {
    "model": pipeline,
    "estimator": estimator,
    "feature_names": feature_names,
    "label_map": {"Not Ready": 0, "Ready": 1},
    "timestamp": timestamp,
}

metadata = {
    "timestamp": timestamp,
    "feature_names": feature_names,
    "label_map": {"Not Ready": 0, "Ready": 1},
    "note": "Retrained on full dataset (no test set metrics)",
}

os.makedirs("models", exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump(artifact, f)
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
with open(latest_model_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"Model retrained and packaged to {model_path}")
print(f"Metadata saved to {meta_path}")
print(f"Latest symlink saved to {latest_model_path}")
