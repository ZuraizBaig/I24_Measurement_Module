"""
Frequency-domain feature extraction for I24 EMG signals.

PSD estimation uses Welch's method. All features are calculated from
the same PSD to ensure consistent spectral definitions.

Dependencies:
    numpy
    scipy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.signal import welch


def _validate_signal(signal: ArrayLike) -> NDArray[np.float64]:
    """Return a validated, finite, non-empty 1D signal."""
    try:
        x = np.asarray(signal, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Signal must contain numeric values.") from exc

    if x.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if x.size == 0:
        raise ValueError("Signal must not be empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal must contain only finite values.")

    return x


def _validate_positive(value: float, name: str) -> float:
    """Validate a finite, strictly positive parameter."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc

    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")

    return value


def _validate_frequency_range(
    frequency_range: tuple[float, float] | None,
    sampling_rate_hz: float,
) -> tuple[float, float] | None:
    """Validate an optional frequency range in Hz."""
    if frequency_range is None:
        return None

    if len(frequency_range) != 2:
        raise ValueError("frequency_range must contain (low, high).")

    low, high = frequency_range

    try:
        low = float(low)
        high = float(high)
    except (TypeError, ValueError) as exc:
        raise ValueError("Frequency limits must be numeric.") from exc

    nyquist = sampling_rate_hz / 2.0

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Frequency limits must be finite.")

    if low < 0 or high <= low or high > nyquist:
        raise ValueError(
            f"Require 0 <= low < high <= Nyquist ({nyquist:g} Hz)."
        )

    return low, high


def calculate_psd(
    signal: ArrayLike,
    *,
    sampling_rate_hz: float = 4000.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str = "constant",
    frequency_range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate power spectral density using Welch's method.

    Returns:
        frequencies_hz: Frequency bins in Hz.
        psd: Power spectral density in signal_units**2 / Hz.

    The optional frequency range selects bins after PSD estimation.
    """
    x = _validate_signal(signal)
    fs = _validate_positive(sampling_rate_hz, "sampling_rate_hz")

    frequency_range = _validate_frequency_range(
        frequency_range, fs
    )

    if nperseg is None:
        nperseg = min(256, x.size)

    if (
        not isinstance(nperseg, (int, np.integer))
        or isinstance(nperseg, bool)
        or nperseg < 2
    ):
        raise ValueError("nperseg must be an integer >= 2.")

    if nperseg > x.size:
        raise ValueError("nperseg cannot exceed signal length.")

    if noverlap is not None:
        if (
            not isinstance(noverlap, (int, np.integer))
            or isinstance(noverlap, bool)
            or noverlap < 0
            or noverlap >= nperseg
        ):
            raise ValueError(
                "noverlap must be an integer from 0 to nperseg - 1."
            )

    if detrend not in ("constant", "linear", False):
        raise ValueError(
            "detrend must be 'constant', 'linear', or False."
        )

    frequencies, psd = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
    )

    if frequency_range is not None:
        low, high = frequency_range
        mask = (frequencies >= low) & (frequencies <= high)
        frequencies = frequencies[mask]
        psd = psd[mask]

    if frequencies.size < 2:
        raise ValueError(
            "At least two frequency bins are required. "
            "Check nperseg and frequency_range."
        )

    return frequencies.astype(float), psd.astype(float)


def _validate_spectrum(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate frequency bins and their corresponding PSD values."""
    f = np.asarray(frequencies_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)

    if f.ndim != 1 or p.ndim != 1:
        raise ValueError("Frequencies and PSD must be one-dimensional.")

    if f.size != p.size or f.size < 2:
        raise ValueError(
            "Frequencies and PSD must have equal length >= 2."
        )

    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(p)):
        raise ValueError("Spectrum must contain only finite values.")

    if np.any(f < 0) or np.any(np.diff(f) <= 0):
        raise ValueError("Frequencies must be non-negative and increasing.")

    if np.any(p < 0):
        raise ValueError("PSD values must be non-negative.")

    if trapezoid(p, f) <= 0:
        raise ValueError("PSD must contain positive total spectral power.")

    return f, p


def calc_total_power(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
) -> float:
    """Calculate total spectral power by integrating the PSD."""
    f, p = _validate_spectrum(frequencies_hz, psd)
    return float(trapezoid(p, f))


def calc_mnf(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
) -> float:
    """
    Calculate Mean Frequency (MNF).

    MNF = integral(f * PSD(f)) / integral(PSD(f))
    """
    f, p = _validate_spectrum(frequencies_hz, psd)
    return float(trapezoid(f * p, f) / trapezoid(p, f))


def calc_mdf(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
) -> float:
    """
    Calculate Median Frequency (MDF).

    Returns the frequency dividing the integrated PSD into two
    equal-power halves, using linear interpolation.
    """
    f, p = _validate_spectrum(frequencies_hz, psd)

    # Trapezoidal power contribution between adjacent frequency bins.
    segment_power = (
        0.5 * (p[:-1] + p[1:]) * np.diff(f)
    )
    cumulative_power = np.concatenate(
        ([0.0], np.cumsum(segment_power))
    )

    half_power = cumulative_power[-1] / 2.0
    index = int(
        np.searchsorted(cumulative_power, half_power, side="left")
    )

    if index == 0:
        return float(f[0])

    power_before = cumulative_power[index - 1]
    power_after = cumulative_power[index]

    if power_after == power_before:
        return float(f[index])

    fraction = (
        (half_power - power_before) / (power_after - power_before)
    )

    return float(f[index - 1] + fraction * (f[index] - f[index - 1]))


def calc_peak_frequency(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
) -> float:
    """Return the frequency bin with maximum PSD."""
    f, p = _validate_spectrum(frequencies_hz, psd)
    return float(f[np.argmax(p)])


def calc_frequency_variance(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
) -> float:
    """Calculate power-weighted variance around the mean frequency."""
    f, p = _validate_spectrum(frequencies_hz, psd)
    weights = p / np.sum(p)
    mean_frequency = np.sum(f * weights)

    return float(np.sum(weights * (f - mean_frequency) ** 2))


def calc_frequency_std(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
) -> float:
    """Calculate power-weighted frequency standard deviation in Hz."""
    return float(np.sqrt(calc_frequency_variance(frequencies_hz, psd)))


def calc_spectral_entropy(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
    *,
    normalize: bool = True,
) -> float:
    """
    Calculate Shannon entropy of the normalized PSD distribution.

    Normalized entropy lies between 0 and 1 when at least two
    frequency bins are available.
    """
    _, p = _validate_spectrum(frequencies_hz, psd)

    probabilities = p / np.sum(p)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))

    if normalize:
        return float(entropy / np.log2(p.size))

    return float(entropy)


def calc_spectral_edge_frequency(
    frequencies_hz: ArrayLike,
    psd: ArrayLike,
    *,
    percentile: float = 95.0,
) -> float:
    """
    Calculate the frequency below which the specified percentage
    of integrated spectral power lies.

    percentile must be strictly between 0 and 100.
    """
    f, p = _validate_spectrum(frequencies_hz, psd)

    try:
        percentile = float(percentile)
    except (TypeError, ValueError) as exc:
        raise ValueError("percentile must be numeric.") from exc

    if not np.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100.")

    segment_power = 0.5 * (p[:-1] + p[1:]) * np.diff(f)
    cumulative_power = np.concatenate(
        ([0.0], np.cumsum(segment_power))
    )

    target = cumulative_power[-1] * percentile / 100.0
    index = int(np.searchsorted(cumulative_power, target, side="left"))

    if index == 0:
        return float(f[0])

    power_before = cumulative_power[index - 1]
    power_after = cumulative_power[index]

    if power_after == power_before:
        return float(f[index])

    fraction = (target - power_before) / (power_after - power_before)

    return float(f[index - 1] + fraction * (f[index] - f[index - 1]))


def extract_frequency_domain_features(
    signal: ArrayLike,
    *,
    sampling_rate_hz: float = 4000.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str = "constant",
    frequency_range: tuple[float, float] | None = None,
    edge_percentile: float = 95.0,
) -> dict[str, float]:
    """
    Estimate the PSD and extract all implemented frequency features.

    Returns:
        A dictionary of scalar frequency-domain features.
    """
    frequencies, psd = calculate_psd(
        signal,
        sampling_rate_hz=sampling_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
        frequency_range=frequency_range,
    )

    return {
        "total_power": calc_total_power(frequencies, psd),
        "mean_frequency": calc_mnf(frequencies, psd),
        "median_frequency": calc_mdf(frequencies, psd),
        "peak_frequency": calc_peak_frequency(frequencies, psd),
        "frequency_variance": calc_frequency_variance(frequencies, psd),
        "frequency_std": calc_frequency_std(frequencies, psd),
        "spectral_entropy": calc_spectral_entropy(frequencies, psd),
        "spectral_edge_frequency": calc_spectral_edge_frequency(
            frequencies, psd, percentile=edge_percentile
        ),
    }