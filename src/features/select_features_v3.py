from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/rf_features_v3_train.csv"
OUT = ROOT / "results/feature_selection_v3"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

metadata = {"label", "recording_id", "window_id", "start_sample"}
feature_cols = [c for c in df.columns if c not in metadata]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y = df["label"]

# Remove features that are entirely missing or constant.
valid_cols = [
    c for c in feature_cols
    if X[c].notna().all() and X[c].nunique() > 1
]
X = X[valid_cols]

# ANOVA ranking on training data only.
f_scores, p_values = f_classif(X, y)

anova = pd.DataFrame({
    "feature": X.columns,
    "anova_f_score": f_scores,
    "anova_p_value": p_values,
}).sort_values("anova_f_score", ascending=False)

anova.to_csv(OUT / "anova_ranking.csv", index=False)

# Absolute Pearson correlations between features.
corr = X.corr(method="pearson").abs()
corr.to_csv(OUT / "absolute_pearson_correlation.csv")

# List highly correlated feature pairs (threshold 0.95).
pairs = []
cols = list(corr.columns)

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        value = corr.iloc[i, j]
        if pd.notna(value) and value >= 0.95:
            pairs.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "absolute_pearson_r": value,
            })

pairs_df = pd.DataFrame(
    pairs,
    columns=["feature_1", "feature_2", "absolute_pearson_r"],
)

if not pairs_df.empty:
    pairs_df = pairs_df.sort_values(
        "absolute_pearson_r", ascending=False
    )

pairs_df.to_csv(OUT / "high_correlation_pairs.csv", index=False)

print("Valid candidate features:", len(valid_cols))
print("\nTop 20 ANOVA-ranked features:")
print(anova.head(20).to_string(index=False))

print("\nHighly correlated pairs (|r| >= 0.95):")
print(pairs_df.to_string(index=False) if not pairs_df.empty else "None")

print(f"\nResults saved in: {OUT}")
