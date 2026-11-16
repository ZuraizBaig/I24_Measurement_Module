"""
Time-domain feature extraction for EMG signals.

All functions operate on a single, finite, one-dimensional EMG segment.
The input signal is not automatically filtered, rectified, or normalized.

Features implemented:
    - Integrated EMG (IEMG)
    - Mean Absolute Value (MAV)
    - Modified Mean Absolute Value 1 (MMAV1)
    - Modified Mean Absolute Value 2 (MMAV2)
    - Simple Square Integral (SSI)
    - Variance
    - V-order
    - Root Mean Square (RMS)
    - Waveform Length (WL)
    - Willison Amplitude (WAMP)
    - Zero Crossings (ZC)
    - Slope Sign Changes (SSC)
    - Log Detector (LOG)

Dependencies:
    numpy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_signal(signal: ArrayLike) -> NDArray[np.float64]:
    """Convert input to a validated, finite, non-empty 1D array."""
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
    """Validate a finite, strictly positive numeric parameter."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc

    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive number.")

    return value


def _validate_non_negative(value: float, name: str) -> float:
    """Validate a finite, non-negative numeric parameter."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be non-negative.") from exc

    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")

    return value


def calc_iemg(signal: ArrayLike) -> float:
    """
    Calculate Integrated EMG as the discrete sum of absolute amplitudes.

    IEMG = sum(|x[n]|)

    Note:
        This discrete feature is not multiplied by the sampling interval.
    """
    x = _validate_signal(signal)
    return float(np.sum(np.abs(x)))


def calc_mav(signal: ArrayLike) -> float:
    """
    Calculate Mean Absolute Value.

    MAV = (1/N) * sum(|x[n]|)
    """
    x = _validate_signal(signal)
    return float(np.mean(np.abs(x)))


def calc_mmav1(signal: ArrayLike) -> float:
    """
    Calculate Modified Mean Absolute Value 1.

    Samples in the first and last 25% of the segment receive weight 0.5.
    Samples in the middle 50% receive weight 1.0.

    MMAV1 = (1/N) * sum(w[n] * |x[n]|)
    """
    x = _validate_signal(signal)
    n = x.size

    weights = np.ones(n, dtype=np.float64)
    edge_length = int(np.floor(0.25 * n))

    if edge_length > 0:
        weights[:edge_length] = 0.5
        weights[-edge_length:] = 0.5

    return float(np.mean(weights * np.abs(x)))


def calc_mmav2(signal: ArrayLike) -> float:
    """
    Calculate Modified Mean Absolute Value 2.

    Uses symmetric edge weighting:
        - First 25%: linearly increasing from 0.5 toward 1.0.
        - Middle 50%: weight 1.0.
        - Last 25%: linearly decreasing toward 0.5.

    The weighted sum is normalized by the number of samples.
    """
    x = _validate_signal(signal)
    n = x.size

    positions = np.arange(n, dtype=np.float64)
    relative_position = positions / max(n - 1, 1)

    weights = np.ones(n, dtype=np.float64)

    left = relative_position < 0.25
    right = relative_position > 0.75

    weights[left] = 0.5 + 2.0 * relative_position[left]
    weights[right] = 0.5 + 2.0 * (1.0 - relative_position[right])

    return float(np.mean(weights * np.abs(x)))


def calc_ssi(signal: ArrayLike) -> float:
    """
    Calculate Simple Square Integral.

    SSI = sum(x[n] ** 2)
    """
    x = _validate_signal(signal)
    return float(np.dot(x, x))


def calc_var(signal: ArrayLike, *, ddof: int = 0) -> float:
    """
    Calculate signal variance.

    ddof=0 computes population variance.
    ddof=1 computes sample variance.

    Raises:
        ValueError: if ddof is negative or there are insufficient samples.
    """
    x = _validate_signal(signal)

    if not isinstance(ddof, (int, np.integer)) or isinstance(ddof, bool):
        raise ValueError("ddof must be a non-negative integer.")

    if ddof < 0:
        raise ValueError("ddof must be non-negative.")

    if x.size <= ddof:
        raise ValueError("Not enough samples for the requested ddof.")

    return float(np.var(x, ddof=ddof))


def calc_vorder(signal: ArrayLike, *, order: float = 2.0) -> float:
    """
    Calculate the V-order feature.

    V-order = (mean(|x[n]| ** order)) ** (1/order)

    For order=1, this equals MAV.
    For order=2, this equals RMS.
    """
    x = _validate_signal(signal)
    order = _validate_positive(order, "order")

    return float(np.mean(np.abs(x) ** order) ** (1.0 / order))


def calc_rms(signal: ArrayLike) -> float:
    """
    Calculate Root Mean Square.

    RMS = sqrt(mean(x[n] ** 2))
    """
    x = _validate_signal(signal)
    return float(np.sqrt(np.mean(x**2)))


def calc_wl(signal: ArrayLike) -> float:
    """
    Calculate Waveform Length.

    WL = sum(|x[n] - x[n-1]|)
    """
    x = _validate_signal(signal)

    if x.size < 2:
        return 0.0

    return float(np.sum(np.abs(np.diff(x))))


def calc_wamp(
    signal: ArrayLike,
    *,
    threshold: float = 0.0,
) -> int:
    """
    Calculate Willison Amplitude.

    Counts adjacent sample differences whose absolute value is
    strictly greater than threshold.
    """
    x = _validate_signal(signal)
    threshold = _validate_non_negative(threshold, "threshold")

    if x.size < 2:
        return 0

    return int(np.count_nonzero(np.abs(np.diff(x)) > threshold))


def calc_zc(
    signal: ArrayLike,
    *,
    threshold: float = 0.0,
) -> int:
    """
    Count Zero Crossings.

    A crossing is counted when adjacent samples have opposite signs
    and their absolute difference is greater than threshold.

    Transitions involving an exact zero are not counted.
    """
    x = _validate_signal(signal)
    threshold = _validate_non_negative(threshold, "threshold")

    if x.size < 2:
        return 0

    x1 = x[:-1]
    x2 = x[1:]

    opposite_signs = (x1 * x2) < 0
    sufficient_difference = np.abs(x2 - x1) > threshold

    return int(np.count_nonzero(opposite_signs & sufficient_difference))


def calc_ssc(
    signal: ArrayLike,
    *,
    threshold: float = 0.0,
) -> int:
    """
    Count Slope Sign Changes.

    A slope sign change is counted at sample n when the adjacent
    differences have opposite signs and both exceed threshold
    in absolute magnitude.
    """
    x = _validate_signal(signal)
    threshold = _validate_non_negative(threshold, "threshold")

    if x.size < 3:
        return 0

    previous_difference = x[1:-1] - x[:-2]
    next_difference = x[2:] - x[1:-1]

    opposite_slopes = (
        previous_difference * next_difference
    ) < 0

    sufficient_slopes = (
        (np.abs(previous_difference) > threshold)
        & (np.abs(next_difference) > threshold)
    )

    return int(np.count_nonzero(opposite_slopes & sufficient_slopes))


def calc_log_detector(
    signal: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> float:
    """
    Calculate the Log Detector using absolute sample amplitudes.

    LOG = exp(mean(log(max(|x[n]|, epsilon))))

    Epsilon prevents log(0). The result is in the same amplitude
    units as the input signal.
    """
    x = _validate_signal(signal)
    epsilon = _validate_positive(epsilon, "epsilon")

    amplitudes = np.maximum(np.abs(x), epsilon)

    return float(np.exp(np.mean(np.log(amplitudes))))


def extract_time_domain_features(
    signal: ArrayLike,
    *,
    wamp_threshold: float = 0.0,
    zc_threshold: float = 0.0,
    ssc_threshold: float = 0.0,
    vorder: float = 2.0,
    variance_ddof: int = 0,
) -> dict[str, float | int]:
    """
    Extract all implemented time-domain features from one EMG segment.

    Returns:
        Dictionary mapping feature names to scalar values.
    """
    x = _validate_signal(signal)

    return {
        "iemg": calc_iemg(x),
        "mav": calc_mav(x),
        "mmav1": calc_mmav1(x),
        "mmav2": calc_mmav2(x),
        "ssi": calc_ssi(x),
        "variance": calc_var(x, ddof=variance_ddof),
        "vorder": calc_vorder(x, order=vorder),
        "rms": calc_rms(x),
        "waveform_length": calc_wl(x),
        "willison_amplitude": calc_wamp(
            x, threshold=wamp_threshold
        ),
        "zero_crossings": calc_zc(
            x, threshold=zc_threshold
        ),
        "slope_sign_changes": calc_ssc(
            x, threshold=ssc_threshold
        ),
        "log_detector": calc_log_detector(x),
    }