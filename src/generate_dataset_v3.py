import numpy as np
from pathlib import Path

SAMPLE_RATE = 32000
NUM_SAMPLES = 320000
NUM_RECORDINGS = 40
CARRIER_HZ = 2000.0

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/raw/gnuradio/recordings_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2027)

classes = [
    "normal",
    "amplitude_degradation",
    "frequency_offset",
    "noise",
    "combined_fault",
]

for label in classes:
    (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

t = np.arange(NUM_SAMPLES) / SAMPLE_RATE

for recording_id in range(NUM_RECORDINGS):
    # Vary amplitude and phase between recording groups.
    amplitude = rng.uniform(0.8, 1.2)
    phase = rng.uniform(-np.pi, np.pi)

    clean = amplitude * np.exp(
        1j * (2 * np.pi * CARRIER_HZ * t + phase)
    )

    degradation = rng.uniform(0.3, 0.7)
    offset_hz = rng.choice([-1, 1]) * rng.uniform(150, 400)
    noise_power = rng.uniform(0.05, 0.25)

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
