from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/rf_features_v3_train.csv"
VAL_PATH = ROOT / "data/processed/rf_features_v3_test.csv"
OUTPUT_DIR = ROOT / "models"

FEATURES = [
    "magnitude_range",
    "power_std",
    "spectral_spread_hz",
    "crest_factor",
    "phase_diff_std",
    "spectral_entropy",
    "spectral_flatness",
    "peak_frequency_hz",
]

RANDOM_STATE = 42
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)

# Fit scaler using normal training data only.
normal_train = train.loc[train["label"] == "normal", FEATURES]
normal_val = validation.loc[validation["label"] == "normal", FEATURES]

if normal_train.empty or normal_val.empty:
    raise ValueError("Normal training and validation windows are required.")

scaler = StandardScaler()
X_train = scaler.fit_transform(normal_train)
X_val_normal = scaler.transform(normal_val)

# A compact neural network learns to reconstruct normal feature vectors.
autoencoder = MLPRegressor(
    hidden_layer_sizes=(6, 3, 6),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=RANDOM_STATE,
)

print("Training autoencoder on normal training windows only...")
autoencoder.fit(X_train, X_train)

def reconstruction_error(model, X):
    reconstructed = model.predict(X)
    return np.mean((X - reconstructed) ** 2, axis=1)

normal_errors = reconstruction_error(autoencoder, X_val_normal)

# Choose threshold using normal validation data only.
threshold = float(np.percentile(normal_errors, 99))

print(f"\nTraining windows: {len(X_train)}")
print(f"Normal validation windows: {len(X_val_normal)}")
print(f"Training iterations: {autoencoder.n_iter_}")
print(f"99th-percentile normal validation threshold: {threshold:.6f}")
print(f"Normal validation windows above threshold: "
      f"{np.mean(normal_errors > threshold) * 100:.2f}%")

bundle = {
    "model": autoencoder,
    "scaler": scaler,
    "features": FEATURES,
    "threshold": threshold,
    "threshold_method": "99th percentile of normal validation reconstruction error",
}

model_path = OUTPUT_DIR / "autoencoder_rf_anomaly_v3.joblib"
joblib.dump(bundle, model_path)

print(f"\nSaved autoencoder: {model_path}")
print("The unseen test dataset has not been used for training or threshold selection.")
