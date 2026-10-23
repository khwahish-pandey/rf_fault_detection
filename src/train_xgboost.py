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
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "rf_features_train.csv")
test = pd.read_csv(DATA / "rf_features_test.csv")

metadata = {"label", "recording_id", "window_id", "start_sample"}

feature_columns = [
    c for c in train.columns if c not in metadata
]

if set(feature_columns) != {
    c for c in test.columns if c not in metadata
}:
    raise ValueError("Training and testing feature columns do not match.")

X_train = train[feature_columns].apply(pd.to_numeric, errors="raise")
X_test = test[feature_columns].apply(pd.to_numeric, errors="raise")

if not X_train.notna().all().all() or not X_test.notna().all().all():
    raise ValueError("Missing feature values detected.")

if not X_train.map(lambda x: abs(x) != float("inf")).all().all():
    raise ValueError("Infinite values detected in training features.")

if not X_test.map(lambda x: abs(x) != float("inf")).all().all():
    raise ValueError("Infinite values detected in test features.")

labels = sorted(train["label"].unique())
label_to_id = {label: i for i, label in enumerate(labels)}

if set(test["label"].unique()) != set(labels):
    raise ValueError("Train and test labels differ.")

y_train = train["label"].map(label_to_id)
y_test = test["label"].map(label_to_id)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", len(feature_columns))
print("Classes:", labels)
print("\nTraining XGBoost...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(labels),
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=4,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
    )
)

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=list(range(len(labels)))))

model_path = MODELS / "xgboost_rf_faults.joblib"
joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "label_to_id": label_to_id,
    },
    model_path,
)

print("\nSaved model:", model_path)
print(
    "Note: performance is on synthetic data; it does not establish "
    "performance on real-world SDR recordings."
)
