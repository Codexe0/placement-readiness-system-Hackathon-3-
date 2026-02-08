import pickle
import os
import pandas as pd

# Load from latest timestamped model (or fallback to legacy path)
latest_model_path = os.path.join("models", "readiness_model_latest.pkl")
legacy_model_path = os.path.join("models", "readiness_model.pkl")

model_path = latest_model_path if os.path.exists(latest_model_path) else legacy_model_path

artifact = pickle.load(open(model_path, "rb"))
model = artifact["model"]
features = artifact["feature_names"]
inv_label_map = {v: k for k, v in artifact.get("label_map", {}).items()}

sample_values = [7.8, 70, 7, 7, 75, 1]
df = pd.DataFrame([sample_values], columns=features)

pred = model.predict(df)[0]
label = inv_label_map.get(pred, str(pred))
print(label)
