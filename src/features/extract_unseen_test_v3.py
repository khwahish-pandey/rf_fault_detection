import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

from feature_calculator import extract_features

WINDOW_SIZE = 1024
DATA_DIR = PROJECT_ROOT / "data/raw/gnuradio/unseen_test_v3"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/rf_features_unseen_v3.csv"

rows = []

for file_path in sorted(DATA_DIR.glob("*/*.cfile")):
    signal = np.fromfile(file_path, dtype=np.complex64)
    label = file_path.parent.name
    recording_id = file_path.stem

    usable_samples = (len(signal) // WINDOW_SIZE) * WINDOW_SIZE

    for window_id, start in enumerate(range(0, usable_samples, WINDOW_SIZE)):
        iq_window = signal[start:start + WINDOW_SIZE]
        features = extract_features(iq_window)

        rows.append({
            "label": label,
            "recording_id": recording_id,
            "window_id": window_id,
            "start_sample": start,
            **features,
        })

df = pd.DataFrame(rows)

metadata = {"label", "recording_id", "window_id", "start_sample"}
feature_columns = [c for c in df.columns if c not in metadata]

df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved features to: {OUTPUT_FILE}")
print(f"Dataset shape: {df.shape}")
print(f"Recordings: {df['recording_id'].nunique()}")
print(f"Windows per class:\n{df.groupby('label')['window_id'].count()}")
print(f"Missing feature values: {df[feature_columns].isna().sum().sum()}")
print(f"Feature count: {len(feature_columns)}")
