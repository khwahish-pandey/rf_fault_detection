from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

ROOT = Path.cwd()
MODEL_PATH = ROOT / "models/autoencoder_rf_anomaly_v4.joblib"
DATA_PATH = ROOT / "data/processed/rf_features_unseen_v3.csv"
OUT_DIR = ROOT / "results/unseen_test_v3"
OUT_PATH = OUT_DIR / "autoencoder_v4_predictions.csv"

bundle = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

features = bundle["features"]
scaler = bundle["scaler"]
model = bundle["model"]
threshold = bundle["threshold"]

X_raw = df[features].to_numpy(dtype=np.float64)
if not np.isfinite(X_raw).all():
    raise ValueError("Input features contain non-finite values.")

X = scaler.transform(X_raw)
reconstructed = model.predict(X)
errors = np.mean((X - reconstructed) ** 2, axis=1)

if not np.isfinite(errors).all():
    raise ValueError("Non-finite reconstruction errors.")

# Binary anomaly detection: normal versus any fault.
actual = (df["label"].to_numpy() != "normal").astype(int)
predicted = (errors > threshold).astype(int)

print(f"Threshold: {threshold:.6f}")
print(f"Windows evaluated: {len(df)}")
print(f"Reconstruction errors finite: {np.isfinite(errors).all()}")
print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")

print("\nClassification report (0=normal, 1=fault):")
print(classification_report(
    actual,
    predicted,
    labels=[0, 1],
    target_names=["normal", "fault"],
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(actual, predicted, labels=[0, 1]))

results = df[["label", "recording_id", "window_id"]].copy()
results["reconstruction_error"] = errors
results["predicted_anomaly"] = predicted
results["actual_anomaly"] = actual

OUT_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(OUT_PATH, index=False)

print("\nAnomaly rate by actual class:")
print(results.groupby("label")["predicted_anomaly"].mean().round(4))

print(f"\nPredictions saved to: {OUT_PATH}")
