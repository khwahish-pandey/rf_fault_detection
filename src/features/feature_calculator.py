import numpy as np
from scipy.stats import skew, kurtosis

SAMPLE_RATE = 32000
EPS = 1e-12


def extract_features(iq):
    """Extract 50 candidate features from one complex IQ window."""

    i = iq.real.astype(np.float64)
    q = iq.imag.astype(np.float64)
    mag = np.abs(iq).astype(np.float64)
    power = mag ** 2

    # Remove the mean for spectral analysis.
    centered = iq - np.mean(iq)
    spectrum = np.fft.fftshift(np.fft.fft(centered))
    psd = np.abs(spectrum) ** 2 / len(iq)

    freqs = np.fft.fftshift(
        np.fft.fftfreq(len(iq), d=1 / SAMPLE_RATE)
    )

    total_power = np.sum(psd) + EPS
    probabilities = psd / total_power

    peak_idx = int(np.argmax(psd))
    centroid = np.sum(freqs * psd) / total_power
    spread = np.sqrt(
        np.sum((freqs - centroid) ** 2 * psd) / total_power
    )

    cumulative = np.cumsum(psd)
    low_idx = min(
        np.searchsorted(cumulative, 0.05 * total_power),
        len(freqs) - 1,
    )
    high_idx = min(
        np.searchsorted(cumulative, 0.95 * total_power),
        len(freqs) - 1,
    )

    occupied_bw = abs(freqs[high_idx] - freqs[low_idx])

    # Central moments of magnitude and power.
    magnitude_skew = skew(mag, bias=False)
    magnitude_kurtosis = kurtosis(mag, fisher=False, bias=False)

    # Amplitude distribution percentiles.
    percentiles = np.percentile(mag, [5, 10, 25, 50, 75, 90, 95])

    # I/Q statistics.
    i_var = np.var(i)
    q_var = np.var(q)
    corr_iq = np.corrcoef(i, q)[0, 1]

    # Frequency-domain power proportions.
    band_edges = np.linspace(-SAMPLE_RATE / 2, SAMPLE_RATE / 2, 9)
    band_powers = []

    for left, right in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= left) & (freqs < right)
        band_powers.append(np.sum(psd[mask]) / total_power)

    # Instantaneous phase differences.
    phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))

    # Autocorrelation at selected lags.
    normalized = centered / np.sqrt(np.mean(np.abs(centered) ** 2) + EPS)
    autocorr = [
        abs(np.mean(normalized[lag:] * np.conj(normalized[:-lag])))
        if lag > 0 else 1.0
        for lag in (1, 2, 4, 8)
    ]

    features = {
        # 1–12: Time-domain statistics
        "mean_i": np.mean(i),
        "mean_q": np.mean(q),
        "std_i": np.std(i),
        "std_q": np.std(q),
        "var_i": i_var,
        "var_q": q_var,
        "rms_i": np.sqrt(np.mean(i ** 2)),
        "rms_q": np.sqrt(np.mean(q ** 2)),
        "mean_magnitude": np.mean(mag),
        "std_magnitude": np.std(mag),
        "mean_power": np.mean(power),
        "power_std": np.std(power),

        # 13–20: Amplitude distribution
        "peak_magnitude": np.max(mag),
        "magnitude_skewness": magnitude_skew,
        "magnitude_kurtosis": magnitude_kurtosis,
        "crest_factor": np.max(mag) / (np.sqrt(np.mean(power)) + EPS),
        "mag_p05": percentiles[0],
        "mag_p10": percentiles[1],
        "mag_p25": percentiles[2],
        "mag_median": percentiles[3],

        # 21–32: FFT and spectral features
        "peak_frequency_hz": freqs[peak_idx],
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "spectral_entropy": -np.sum(
            probabilities * np.log2(probabilities + EPS)
        ),
        "spectral_flatness": np.exp(
            np.mean(np.log(psd + EPS))
        ) / (np.mean(psd) + EPS),
        "peak_power_fraction": np.max(psd) / total_power,
        "spectral_rolloff_5_95_hz": occupied_bw,
        "negative_frequency_power_fraction": (
            np.sum(psd[freqs < 0]) / total_power
        ),
        "positive_frequency_power_fraction": (
            np.sum(psd[freqs >= 0]) / total_power
        ),
        "dc_power_fraction": psd[np.argmin(np.abs(freqs))] / total_power,
        "fft_power_std": np.std(psd),
        "fft_power_max": np.max(psd),

        # 33–40: PSD features
        "psd_mean": np.mean(psd),
        "psd_median": np.median(psd),
        "psd_std": np.std(psd),
        "psd_min": np.min(psd),
        "psd_max": np.max(psd),
        "psd_p25": np.percentile(psd, 25),
        "psd_p75": np.percentile(psd, 75),
        "psd_noise_floor_estimate": np.percentile(psd, 10),

        # 41–45: Signal quality proxies
        "papr": np.max(power) / (np.mean(power) + EPS),
        "power_db": 10 * np.log10(np.mean(power) + EPS),
        "magnitude_range": np.ptp(mag),
        "phase_diff_mean": np.mean(phase_diff),
        "phase_diff_std": np.std(phase_diff),

        # 46–50: I/Q and temporal dependence
        "iq_correlation": (
            corr_iq if np.isfinite(corr_iq) else 0.0
        ),
        "iq_variance_ratio": i_var / (q_var + EPS),
        "autocorr_lag_1": autocorr[0],
        "autocorr_lag_2": autocorr[1],
        "autocorr_lag_4": autocorr[2],
    }

    assert len(features) == 50, f"Expected 50 features, got {len(features)}"
    return features
