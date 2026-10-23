
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/raw/kaggle/rf_signal_data/logged_data.csv"

df = pd.read_csv(FILE)

print("\n========== DATASET SIZE ==========")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\n========== INTERFERENCE LABELS ==========")
print(df["Interference Type"].value_counts(dropna=False))

print("\n========== LABEL VS MODULATION ==========")
print(pd.crosstab(
    df["Interference Type"],
    df["Modulation"],
    dropna=False
).to_string())

print("\n========== LABEL VS DEVICE TYPE ==========")
print(pd.crosstab(
    df["Interference Type"],
    df["Device Type"],
    dropna=False
).to_string())

print("\n========== LABEL VS DEVICE STATUS ==========")
print(pd.crosstab(
    df["Interference Type"],
    df["Device Status"],
    dropna=False
).to_string())

print("\n========== TIMESTAMP CHECK ==========")
timestamps = pd.to_datetime(df["Timestamp"], errors="coerce")
print("Invalid timestamps:", timestamps.isna().sum())
print("Unique timestamps:", timestamps.nunique())
print("Time range:", timestamps.min(), "to", timestamps.max())

print("\n========== DUPLICATE CHECK ==========")
print("Fully duplicated rows:", df.duplicated().sum())

print("\n========== I/Q LENGTH CHECK ==========")
print("Rows with I/Q data:", df["I/Q Data"].notna().sum())
print("Rows without I/Q data:", df["I/Q Data"].isna().sum())

print("\nAnalysis complete. No dataset files were modified.")
