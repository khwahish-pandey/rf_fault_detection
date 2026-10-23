
from pathlib import Path
import ast
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/raw/kaggle/rf_signal_data/logged_data.csv"

df = pd.read_csv(FILE)

print("\n========== I/Q AVAILABILITY VS LABEL ==========")
print(pd.crosstab(
    df["Interference Type"].fillna("MISSING"),
    df["I/Q Data"].notna().map({True: "Available", False: "Missing"})
).to_string())

print("\n========== I/Q SAMPLE LENGTHS ==========")
lengths = []
parse_errors = 0
invalid_samples = 0

for value in df["I/Q Data"].dropna():
    try:
        samples = np.asarray(ast.literal_eval(value), dtype=np.complex128)
        lengths.append(len(samples))
        if not np.isfinite(samples).all():
            invalid_samples += 1
    except (ValueError, SyntaxError, TypeError, OverflowError):
        parse_errors += 1

if lengths:
    print("Successfully parsed rows:", len(lengths))
    print("Parse errors:", parse_errors)
    print("Rows with non-finite samples:", invalid_samples)
    print(pd.Series(lengths, name="sample_count").describe().to_string())
    print("Unique sample lengths:", sorted(set(lengths))[:30])
else:
    print("No I/Q records were successfully parsed.")

print("\n========== LABELS AMONG VALID I/Q ROWS ==========")
valid_iq = df[df["I/Q Data"].notna()]
print(valid_iq["Interference Type"].value_counts(dropna=False).to_string())

print("\nCheck complete. No dataset files were modified.")
