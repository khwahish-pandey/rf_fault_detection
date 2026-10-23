from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/rf_features_50.csv"
OUT = ROOT / "data/processed/features"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(INPUT)
target = "label"
excluded = {target, "window_id", "start_sample"}
feature_cols = [
    c for c in df.select_dtypes(include=np.number).columns
    if c not in excluded
]
X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = df[target]

scores, pvalues = f_classif(X, y)
ranking = pd.DataFrame({
    "feature": feature_cols,
    "f_score": scores,
    "p_value": pvalues,
}).replace([np.inf, -np.inf], np.nan)
ranking = ranking.sort_values("f_score", ascending=False)
ranking.to_csv(OUT / "feature_ranking.csv", index=False)
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = (
    upper.stack()
    .rename("absolute_correlation")
    .reset_index()
    .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
)
pairs = pairs.sort_values("absolute_correlation", ascending=False)
pairs.to_csv(OUT / "feature_correlations.csv", index=False)
print("\n========== DATASET ==========")
print("Shape:", df.shape)
print("Feature count:", len(feature_cols))
print("\nClass counts:")
print(y.value_counts().to_string())
print("\n========== TOP 15 FEATURES ==========")
print(ranking.head(15).to_string(index=False))
print("\n========== TOP 15 CORRELATED PAIRS ==========")
print(pairs.head(15).to_string(index=False))
print("\nSaved:")
print(OUT / "feature_ranking.csv")
print(OUT / "feature_correlations.csv")
