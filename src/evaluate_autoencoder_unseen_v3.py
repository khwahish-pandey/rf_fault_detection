from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/autoencoder_rf_anomaly_v3.joblib"
DATA_PATH = ROOT / "data/processed/rf_features_unseen_v3.csv"
OUTPUT_DIR = ROOT / "results/unseen_test_v3"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
threshold = bundle["threshold"]

df = pd.read_csv(DATA_PATH)

if not set(features).issubset(df.columns):
    raise ValueError("Required Autoencoder features are missing.")

if df[features].isna().any().any():
    raise ValueError("Missing feature values found.")

X = scaler.transform(df[features])
reconstructed = model.predict(X)
errors = np.mean((X - reconstructed) ** 2, axis=1)

df["reconstruction_error"] = errors
df["anomaly"] = errors > threshold

df["actual_anomaly"] = df["label"] != "normal"

print("AUTOENCODER UNSEEN-DATA EVALUATION")
print(f"Threshold: {threshold:.6f}")
print(f"Windows evaluated: {len(df)}")

print("\nAnomaly detection report (0 = normal, 1 = fault):")
print(classification_report(
    df["actual_anomaly"].astype(int),
    df["anomaly"].astype(int),
    labels=[0, 1],
    target_names=["normal", "fault"],
    zero_division=0,
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(
    df["actual_anomaly"].astype(int),
    df["anomaly"].astype(int),
    labels=[0, 1],
))

print("\nAnomaly rate by class:")
print(
    df.groupby("label")["anomaly"]
    .mean()
    .mul(100)
    .round(2)
    .rename("flagged_as_anomaly_percent")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_DIR / "autoencoder_predictions.csv"
df.to_csv(out_path, index=False)

print(f"\nSaved predictions: {out_path}")
