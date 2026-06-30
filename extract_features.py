"""
Feature Extraction for ISRUC Sleep Staging

Loads the epoched data (isruc_data.npz) and turns each 30s epoch
(6000 raw samples) into a compact feature vector that shallow
classifiers can use.

Features per epoch (single channel C3-A2):
  Time Domain:
    - mean, std, variance
    - skewness, kurtosis
    - Hjorth: activity, mobility, complexity
    - zero-crossing rate
  Frequency Domain (the most discriminative for sleep):
    - relative band power: delta, theta, alpha, sigma, beta
    - spectral edge frequency (SEF95)

Output:
    features -> shape (n_epochs, n_features)
    feature_names -> list of column names
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis


SFREQ = 200  # Hz

# Frequency bands (Hz) — these define sleep stages physiologically
BANDS = {
    "delta": (0.5, 4),    # dominates deep sleep (N3)
    "theta": (4, 8),      # present in N1, drowsiness
    "alpha": (8, 13),     # relaxed wakefulness, eyes closed
    "sigma": (12, 16),    # sleep spindles -> marker of N2
    "beta":  (16, 30),    # active wakefulness
}


# Time-Domain Features
def hjorth_parameters(epoch):
    """Hjorth activity, mobility, complexity — capture signal dynamics."""
    # First and second derivatives (differences)
    d1 = np.diff(epoch)
    d2 = np.diff(d1)

    var_zero = np.var(epoch)     # activity = variance of the signal
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)

    # mobility = sqrt(var of 1st derivative / var of signal)
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    # complexity = ratio of mobilities of derivative and signal
    mob_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 0 else 0
    complexity = mob_d1 / mobility if mobility > 0 else 0

    return var_zero, mobility, complexity


def zero_crossing_rate(epoch):
    """How often the signal crosses zero — rough frequency proxy."""
    signs = np.sign(epoch)
    signs[signs == 0] = 1
    return np.sum(np.abs(np.diff(signs)) > 0) / len(epoch)


def time_domain_features(epoch):
    mean = np.mean(epoch)
    std = np.std(epoch)
    var = np.var(epoch)
    sk = skew(epoch)
    kurt = kurtosis(epoch)
    activity, mobility, complexity = hjorth_parameters(epoch)
    zcr = zero_crossing_rate(epoch)
    return [mean, std, var, sk, kurt, activity, mobility, complexity, zcr]


TIME_NAMES = ["mean", "std", "var", "skew", "kurtosis",
              "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
              "zero_crossing_rate"]


# Frequency-Domain Features
def frequency_domain_features(epoch):
    # Welch's method estimates the power spectral density (PSD)
    freqs, psd = sp_signal.welch(epoch, fs=SFREQ, nperseg=min(len(epoch), 2 * SFREQ))

    total_power = np.trapezoid(psd, freqs)          # total area under the PSD
    feats = []

    # Relative power in each band = band power / total power
    for (low, high) in BANDS.values():
        idx = (freqs >= low) & (freqs <= high)
        band_power = np.trapezoid(psd[idx], freqs[idx])
        rel_power = band_power / total_power if total_power > 0 else 0
        feats.append(rel_power)

    # Spectral Edge Frequency 95%: frequency below which 95% of power lies
    cumulative = np.cumsum(psd)
    if cumulative[-1] > 0:
        sef95 = freqs[np.searchsorted(cumulative, 0.95 * cumulative[-1])]
    else:
        sef95 = 0
    feats.append(sef95)

    return feats


FREQ_NAMES = [f"rel_power_{b}" for b in BANDS.keys()] + ["sef95"]


# Combine: one feature vector per epoch
def extract_features_one_epoch(epoch):
    return time_domain_features(epoch) + frequency_domain_features(epoch)


FEATURE_NAMES = TIME_NAMES + FREQ_NAMES


# Main
def main():
    data = np.load("isruc_data.npz")
    X_raw, y, groups = data["X"], data["y"], data["groups"]

    print(f"Loaded {X_raw.shape[0]} epochs of {X_raw.shape[1]} samples each")
    print(f"Extracting {len(FEATURE_NAMES)} features per epoch...")

    features = np.array([extract_features_one_epoch(epoch) for epoch in X_raw])

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    # Sanity check: any NaN or inf?
    n_bad = np.sum(~np.isfinite(features))
    print(f"\nNon-finite values: {n_bad}")
    if n_bad > 0:
        print("  WARNING: replacing non-finite values with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    np.savez_compressed("isruc_features.npz",
                        features=features, y=y, groups=groups,
                        feature_names=FEATURE_NAMES)
    print("\nSaved to isruc_features.npz")


if __name__ == "__main__":
    main()