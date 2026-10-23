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

train_df = pd.read_csv(DATA_DIR / "rf_features_v3_train.csv")
test_df = pd.read_csv(DATA_DIR / "rf_features_v3_test.csv")

metadata = {"label", "recording_id", "window_id", "start_sample"}

feature_columns = [
    column for column in train_df.columns
    if column not in metadata
]

# Ensure training and testing use exactly the same features.
assert feature_columns == [
    column for column in test_df.columns
    if column not in metadata
]

assert not train_df[feature_columns].isna().any().any()
assert not test_df[feature_columns].isna().any().any()

classes = sorted(train_df["label"].unique())
label_to_id = {label: index for index, label in enumerate(classes)}

X_train = train_df[feature_columns]
y_train = train_df["label"].map(label_to_id)

X_test = test_df[feature_columns]
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

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Features: {len(feature_columns)}")
print(f"Classes: {classes}")
print("\nTraining XGBoost...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"\nTest window accuracy: {accuracy_score(y_test, predictions):.4f}")
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

# Save separately from the V2 model.
model_path = MODEL_DIR / "xgboost_rf_faults_v3.joblib"
joblib.dump({
    "model": model,
    "feature_columns": feature_columns,
    "label_to_id": label_to_id,
}, model_path)

print(f"\nSaved model: {model_path}")

print("\nTop 15 feature importances:")
importance = pd.Series(
    model.feature_importances_,
    index=feature_columns,
).sort_values(ascending=False)

print(importance.head(15).to_string())
