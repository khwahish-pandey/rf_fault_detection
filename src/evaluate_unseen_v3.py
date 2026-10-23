from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/xgboost_rf_faults_v3_reduced.joblib"
DATA_PATH = ROOT / "data/processed/rf_features_unseen_v3.csv"
RESULTS_DIR = ROOT / "results/unseen_test_v3"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["feature_columns"]
label_to_id = bundle["label_to_id"]

df = pd.read_csv(DATA_PATH)

missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing model features: {missing_features}")

if df[features].isna().any().any():
    raise ValueError("Missing values found in model features.")

classes = list(label_to_id.keys())
id_to_label = {v: k for k, v in label_to_id.items()}

X = df[features]
y_true = df["label"].map(label_to_id)

if y_true.isna().any():
    raise ValueError("Dataset contains labels unknown to the saved model.")

y_pred = model.predict(X)

print("\nUNSEEN SYNTHETIC DATASET EVALUATION")
print(f"Recordings: {df['recording_id'].nunique()}")
print(f"Windows: {len(df)}")
print(f"Features used: {len(features)}")
print(f"Features: {features}")
print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
))

df["predicted_id"] = y_pred
df["predicted_label"] = df["predicted_id"].map(id_to_label)
df["correct"] = df["label"] == df["predicted_label"]

window_results = RESULTS_DIR / "window_predictions.csv"
df.to_csv(window_results, index=False)

recording_results = (
    df.groupby(["recording_id", "label"])["predicted_label"]
    .agg(lambda values: values.value_counts().idxmax())
    .reset_index(name="recording_prediction")
)
recording_results["correct"] = (
    recording_results["label"] == recording_results["recording_prediction"]
)

recording_results.to_csv(
    RESULTS_DIR / "recording_predictions.csv", index=False
)

print("\nRecording-level majority-vote accuracy:",
      f"{recording_results['correct'].mean():.4f}")
print(f"Saved window predictions: {window_results}")
print(f"Saved recording predictions: {RESULTS_DIR / 'recording_predictions.csv'}")
