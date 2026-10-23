
from pathlib import Path
import ast
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/raw/kaggle/rf_signal_data/logged_data.csv"

if not FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {FILE}")

# Load metadata without printing the potentially large I/Q field
df = pd.read_csv(FILE)

print("\n========== DATASET OVERVIEW ==========")
print("File:", FILE)
print("Full dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\n========== MISSING VALUES ==========")
print(df.isna().sum().sort_values(ascending=False).to_string())

for col in [
    "Interference Type", "Modulation", "Device Type",
    "Antenna Type", "Device Status", "Power Source"
]:
    if col in df.columns:
        print(f"\n========== {col} DISTRIBUTION ==========")
        print(df[col].value_counts(dropna=False).to_string())

for col in ["Frequency", "Signal Strength", "Bandwidth"]:
    if col in df.columns:
        print(f"\n========== {col} SUMMARY ==========")
        print(pd.to_numeric(df[col], errors="coerce").describe().to_string())

print("\n========== I/Q DATA CHECK ==========")
if "I/Q Data" not in df.columns:
    print("No I/Q Data column found.")
else:
    for idx, value in df["I/Q Data"].dropna().head(5).items():
        try:
            samples = ast.literal_eval(value)
            samples = np.asarray(samples, dtype=np.complex128)
            print(
                f"Row {idx}: samples={len(samples)}, "
                f"finite={np.isfinite(samples).all()}, "
                f"mean_power={np.mean(np.abs(samples)**2):.5f}"
            )
        except Exception as exc:
            print(f"Row {idx}: unable to parse I/Q data: {exc}")

print("\nInspection complete. No dataset files were modified.")
