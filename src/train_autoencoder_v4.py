from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path.cwd()
DATA_PATH = ROOT / "data/processed/rf_features_v3_clean.csv"
MODEL_PATH = ROOT / "models/autoencoder_rf_anomaly_v4.joblib"

# Exclude peak_frequency_hz: the normal training carrier was constant.
# Exclude crest_factor: it was also effectively constant.
FEATURES = [
    "magnitude_range",
    "power_std",
    "spectral_spread_hz",
    "phase_diff_std",
    "spectral_entropy",
    "spectral_flatness",
]

df = pd.read_csv(DATA_PATH)

required = FEATURES + ["label", "recording_id"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

normal = df[df["label"] == "normal"].copy()

# Split by recording so windows from one recording cannot leak
# between training and validation.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, val_idx = next(
    splitter.split(normal[FEATURES], groups=normal["recording_id"])
)

train_normal = normal.iloc[train_idx]
val_normal = normal.iloc[val_idx]

# Drop features with essentially no variation in normal training data.
stds = train_normal[FEATURES].std()
selected = stds[stds > 1e-10].index.tolist()

if len(selected) < 2:
    raise ValueError(
        "Too few varying features for the Autoencoder. "
        "Inspect feature generation before proceeding."
    )

print("Selected features:", selected)
print("Normal training recordings:", train_normal["recording_id"].nunique())
print("Normal validation recordings:", val_normal["recording_id"].nunique())
print("Training windows:", len(train_normal))
print("Validation windows:", len(val_normal))

X_train_raw = train_normal[selected].to_numpy(dtype=np.float64)
X_val_raw = val_normal[selected].to_numpy(dtype=np.float64)

if not np.isfinite(X_train_raw).all() or not np.isfinite(X_val_raw).all():
    raise ValueError("Non-finite feature values found.")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

# Train a fresh model only on normal training recordings.
# Early stopping is disabled because we control validation explicitly.
model = MLPRegressor(
    hidden_layer_sizes=(4, 2, 4),
    activation="tanh",
    solver="adam",
    alpha=0.01,
    batch_size=256,
    learning_rate_init=0.0005,
    max_iter=500,
    early_stopping=False,
    random_state=42,
)

model.fit(X_train, X_train)

reconstructed_val = model.predict(X_val)
val_errors = np.mean((X_val - reconstructed_val) ** 2, axis=1)

if not np.isfinite(val_errors).all():
    raise ValueError("Non-finite validation errors; model training is unstable.")

# A 99th-percentile threshold aims for about 1% validation false alarms.
threshold = float(np.quantile(val_errors, 0.99))

bundle = {
    "model": model,
    "scaler": scaler,
    "features": selected,
    "threshold": threshold,
    "threshold_method": "99th percentile of held-out normal recordings",
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, MODEL_PATH)

print("\nAutoencoder V4 trained.")
print(f"Validation reconstruction error median: {np.median(val_errors):.6g}")
print(f"Validation reconstruction error 99th percentile: {threshold:.6g}")
print(f"Saved model: {MODEL_PATH}")
print("\nThis is a new model; existing model files were not overwritten.")
