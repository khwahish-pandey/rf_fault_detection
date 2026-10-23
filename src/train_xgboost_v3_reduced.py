from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

train_df = pd.read_csv(DATA_DIR / "rf_features_v3_train.csv")
test_df = pd.read_csv(DATA_DIR / "rf_features_v3_test.csv")

for feature in FEATURES:
    if feature not in train_df.columns or feature not in test_df.columns:
        raise ValueError(f"Missing required feature: {feature}")

if train_df[FEATURES].isna().any().any():
    raise ValueError("Training features contain missing values.")

if test_df[FEATURES].isna().any().any():
    raise ValueError("Test features contain missing values.")

classes = sorted(train_df["label"].unique())
label_to_id = {label: i for i, label in enumerate(classes)}

X_train = train_df[FEATURES]
y_train = train_df["label"].map(label_to_id)

X_test = test_df[FEATURES]
y_test = test_df["label"].map(label_to_id)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes),
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=4,
)

print("Training reduced-feature XGBoost...")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Feature count:", len(FEATURES))
print("Features:", FEATURES)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(
    y_test,
    predictions,
    labels=list(range(len(classes))),
))

model_path = MODEL_DIR / "xgboost_rf_faults_v3_reduced.joblib"
joblib.dump({
    "model": model,
    "feature_columns": FEATURES,
    "label_to_id": label_to_id,
}, model_path)

print(f"\nSaved reduced model: {model_path}")
