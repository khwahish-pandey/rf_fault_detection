from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rf_features_unseen_v3.csv"
OUT = ROOT / "results/unseen_test_v3"

features = [
    "magnitude_range",
    "power_std",
    "spectral_spread_hz",
    "crest_factor",
    "phase_diff_std",
    "spectral_entropy",
    "spectral_flatness",
    "peak_frequency_hz",
]

df = pd.read_csv(DATA)

for label in ["noise", "combined_fault"]:
    subset = df[df["label"] == label]
    print(f"\n--- {label} ---")
    print(subset[features].describe().loc[["mean", "std", "min", "50%", "max"]].round(4))

means = df.groupby("label")[features].mean().T
means.to_csv(OUT / "feature_means_by_class.csv")

print(f"\nSaved class feature means to: {OUT / 'feature_means_by_class.csv'}")
