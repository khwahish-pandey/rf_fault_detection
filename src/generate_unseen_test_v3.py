import numpy as np
from pathlib import Path

SAMPLE_RATE = 32000
NUM_SAMPLES = 320000
NUM_RECORDINGS = 10
CARRIER_HZ = 3500.0

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/raw/gnuradio/unseen_test_v3"
rng = np.random.default_rng(9031)

classes = [
    "normal",
    "amplitude_degradation",
    "frequency_offset",
    "noise",
    "combined_fault",
]

t = np.arange(NUM_SAMPLES) / SAMPLE_RATE

for label in classes:
    (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

for recording_id in range(NUM_RECORDINGS):
    amplitude = rng.uniform(0.65, 1.35)
    phase = rng.uniform(-np.pi, np.pi)

    clean = amplitude * np.exp(
        1j * (2 * np.pi * CARRIER_HZ * t + phase)
    )

    degradation = rng.uniform(0.10, 0.85)
    offset_hz = rng.choice([-1, 1]) * rng.uniform(450, 900)
    noise_power = rng.uniform(0.30, 0.60)

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(NUM_SAMPLES)
        + 1j * rng.standard_normal(NUM_SAMPLES)
    )

    signals = {
        "normal": clean,
        "amplitude_degradation": degradation * clean,
        "frequency_offset": clean * np.exp(
            2j * np.pi * offset_hz * t
        ),
        "noise": clean + noise,
        "combined_fault": (
            clean * np.exp(2j * np.pi * offset_hz * t) + noise
        ),
    }

    for label, signal in signals.items():
        filename = f"{label}_{recording_id:03d}.cfile"
        path = OUTPUT_DIR / label / filename
        signal.astype(np.complex64).tofile(path)

print(f"Generated {NUM_RECORDINGS * len(classes)} recordings.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Samples per recording: {NUM_SAMPLES}")
print(f"Duration per recording: {NUM_SAMPLES / SAMPLE_RATE:.1f} seconds")
print(f"Nominal carrier frequency: {CARRIER_HZ} Hz")
print(f"Classes: {', '.join(classes)}")
