
import numpy as np
from pathlib import Path

# Project configuration
SAMPLE_RATE = 32000
INPUT_FILE = Path("data/raw/gnuradio/normal_001.cfile")
OUTPUT_DIR = Path("data/raw/gnuradio/faults")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the verified normal complex IQ samples
normal = np.fromfile(INPUT_FILE, dtype=np.complex64)

if normal.size == 0:
    raise ValueError(f"No IQ samples found in {INPUT_FILE}")

if not np.isfinite(normal).all():
    raise ValueError("Input contains invalid numerical values")

rng = np.random.default_rng(42)

# 1. Frequency offset: shift carrier by 200 Hz
n = np.arange(normal.size)
frequency_offset = 200
frequency_fault = (
    normal * np.exp(
        2j * np.pi * frequency_offset * n / SAMPLE_RATE
    )
).astype(np.complex64)

# 2. Amplitude degradation: reduce amplitude by 50%
amplitude_fault = (0.5 * normal).astype(np.complex64)

# 3. Additive white Gaussian noise
noise_power = 0.1
noise = np.sqrt(noise_power / 2) * (
    rng.standard_normal(normal.size)
    + 1j * rng.standard_normal(normal.size)
)
noise_fault = (normal + noise).astype(np.complex64)

# 4. Combined fault: frequency offset + noise
combined_fault = (frequency_fault + noise).astype(np.complex64)

signals = {
    "normal": normal,
    "frequency_offset": frequency_fault,
    "amplitude_degradation": amplitude_fault,
    "noise": noise_fault,
    "combined_fault": combined_fault,
}

for name, signal in signals.items():
    output_file = OUTPUT_DIR / f"{name}.cfile"
    signal.tofile(output_file)

    print(
        f"{name:24s} "
        f"samples={signal.size:6d} "
        f"finite={np.isfinite(signal).all()} "
        f"size={output_file.stat().st_size / (1024**2):.2f} MiB"
    )

print("\nFault signal generation complete.")
