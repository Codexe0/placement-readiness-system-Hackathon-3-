import pickle
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts):
	return PROJECT_ROOT.joinpath(*parts)


# Load from latest timestamped model (or fallback to legacy path)
latest_model_path = project_path("models", "readiness_model_latest.pkl")
legacy_model_path = project_path("models", "readiness_model.pkl")

model_path = latest_model_path if os.path.exists(latest_model_path) else legacy_model_path

with open(model_path, "rb") as model_file:
	artifact = pickle.load(model_file)
model = artifact["model"]
features = artifact["feature_names"]
inv_label_map = {v: k for k, v in artifact.get("label_map", {}).items()}

sample_values = [7.8, 70, 7, 7, 75, 1]
df = pd.DataFrame([sample_values], columns=features)

pred = model.predict(df)[0]
label = inv_label_map.get(pred, str(pred))
print(label)
