
"""
Positive peak and negative trough detection.

Uses scipy.signal.find_peaks.

No filtering is performed in this module.
"""

import numpy as np

from scipy.signal import find_peaks

from .models import DetectedEvent


def validate_signal(signal, sampling_rate):
    """Validate and convert the input signal."""

    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(x) < 3:
        raise ValueError("Signal must contain at least three samples.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite values.")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    return x


def estimate_prominence(signal, prominence_factor=0.35):
    """
    Estimate a prominence threshold using robust signal dispersion.

    Uses median absolute deviation (MAD).
    """

    if prominence_factor < 0:
        raise ValueError("Prominence factor cannot be negative.")

    median = np.median(signal)

    mad = np.median(np.abs(signal - median))

    robust_sigma = 1.4826 * mad

    fallback = np.ptp(signal) * 0.05

    threshold = max(
        robust_sigma * prominence_factor,
        fallback,
        np.finfo(float).eps
    )

    return float(threshold)


def detect_positive_peaks(
    signal,
    sampling_rate,
    min_distance_s=1.0,
    prominence=None,
    prominence_factor=0.35,
):
    """
    Detect positive peaks in the respiratory waveform.

    Returns
    -------
    list[DetectedEvent]
    """

    x = validate_signal(signal, sampling_rate)

    if min_distance_s <= 0:
        raise ValueError("Minimum peak distance must be positive.")

    distance_samples = max(
        1,
        int(round(min_distance_s * sampling_rate))
    )

    if prominence is None:
        prominence = estimate_prominence(
            x,
            prominence_factor
        )

    if prominence < 0:
        raise ValueError("Prominence cannot be negative.")

    peak_indices, properties = find_peaks(
        x,
        distance=distance_samples,
        prominence=prominence
    )

    events = []

    for i, index in enumerate(peak_indices):

        events.append(
            DetectedEvent(
                event_type="positive_peak",
                sample_index=int(index),
                time_s=float(index / sampling_rate),
                amplitude=float(x[index]),
                prominence=float(properties["prominences"][i])
            )
        )

    return events


def detect_negative_troughs(
    signal,
    sampling_rate,
    min_distance_s=1.0,
    prominence=None,
    prominence_factor=0.35,
):
    """
    Detect negative troughs in the respiratory waveform.

    Negative trough detection is implemented by applying
    find_peaks() to the inverted signal.
    """

    x = validate_signal(signal, sampling_rate)

    if min_distance_s <= 0:
        raise ValueError("Minimum trough distance must be positive.")

    distance_samples = max(
        1,
        int(round(min_distance_s * sampling_rate))
    )

    if prominence is None:
        prominence = estimate_prominence(
            x,
            prominence_factor
        )

    if prominence < 0:
        raise ValueError("Prominence cannot be negative.")

    trough_indices, properties = find_peaks(
        -x,
        distance=distance_samples,
        prominence=prominence
    )

    events = []

    for i, index in enumerate(trough_indices):

        events.append(
            DetectedEvent(
                event_type="negative_trough",
                sample_index=int(index),
                time_s=float(index / sampling_rate),
                amplitude=float(x[index]),
                prominence=float(properties["prominences"][i])
            )
        )

    return events
