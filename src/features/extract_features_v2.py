
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the verified feature-extraction function.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

from feature_calculator import extract_features

WINDOW_SIZE = 1024

DATA_DIR = PROJECT_ROOT / "data/raw/gnuradio/recordings_v2"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "rf_features_v2.csv"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

rows = []
recording_count = 0

for file_path in sorted(DATA_DIR.glob("*/*.cfile")):
    signal = np.fromfile(file_path, dtype=np.complex64)

    if signal.size == 0 or not np.isfinite(signal).all():
        print(f"Skipping empty/invalid file: {file_path}")
        continue

    label = file_path.parent.name
    recording_id = file_path.stem
    usable_samples = (len(signal) // WINDOW_SIZE) * WINDOW_SIZE

    if usable_samples == 0:
        print(f"Skipping short recording: {file_path}")
        continue

    signal = signal[:usable_samples]
    recording_count += 1

    for window_id, start in enumerate(
        range(0, usable_samples, WINDOW_SIZE)
    ):
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

if df.empty:
    raise RuntimeError("No feature rows generated.")

metadata = {"label", "recording_id", "window_id", "start_sample"}
feature_columns = [c for c in df.columns if c not in metadata]

df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction complete.")
print("Recordings processed:", recording_count)
print("Dataset shape:", df.shape)
print("Candidate features:", len(feature_columns))
print("\nRows per class:")
print(df.groupby("label").size().to_string())
print("\nRecordings per class:")
print(df.groupby("label")["recording_id"].nunique().to_string())
print("\nNon-finite feature values:",
      int((~np.isfinite(df[feature_columns].to_numpy())).sum()))
print("\nSaved to:", OUTPUT_FILE)

