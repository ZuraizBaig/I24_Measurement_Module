
"""Signal-processing utilities for I24 EMG recordings.

All functions operate on one-dimensional signals and leave the input
array unchanged.

Sampling rates are expressed in Hz and window durations in milliseconds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import (
    butter,
    detrend as scipy_detrend,
    filtfilt,
    iirnotch,
    sosfiltfilt,
)


def _validate_signal(
    signal: np.ndarray,
    sampling_rate_hz: float | None = None,
) -> tuple[np.ndarray, float | None]:
    """Validate and copy a signal and optionally its sampling rate."""
    values = np.asarray(signal, dtype=float)

    if values.ndim != 1:
        raise ValueError("Signal must be a one-dimensional array.")

    if values.size == 0:
        raise ValueError("Signal must not be empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Signal must contain only finite values.")

    if sampling_rate_hz is None:
        return values.copy(), None

    fs = float(sampling_rate_hz)

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "Sampling rate must be finite and greater than zero."
        )

    return values.copy(), fs


def _validate_cutoff(
    cutoff_hz: float,
    sampling_rate_hz: float,
    name: str = "Cutoff frequency",
) -> float:
    """Validate a frequency against the Nyquist frequency."""
    cutoff = float(cutoff_hz)
    nyquist = sampling_rate_hz / 2.0

    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(
            f"{name} must be finite and greater than zero."
        )

    if cutoff >= nyquist:
        raise ValueError(
            f"{name} must be below the Nyquist frequency "
            f"({nyquist:g} Hz)."
        )

    return cutoff


def _validate_order(order: int) -> int:
    """Validate a Butterworth filter order."""
    if isinstance(order, bool) or not isinstance(
        order, (int, np.integer)
    ):
        raise ValueError("Filter order must be a positive integer.")

    if order < 1:
        raise ValueError("Filter order must be a positive integer.")

    return int(order)


def _apply_sos_filter(
    values: np.ndarray,
    sos: np.ndarray,
    filter_name: str,
) -> np.ndarray:
    """Apply zero-phase filtering with second-order sections."""
    try:
        return sosfiltfilt(sos, values)
    except ValueError as exc:
        raise ValueError(
            f"Signal is too short for the {filter_name} filter. "
            "Use a longer signal or a lower filter order."
        ) from exc


def lowpass_filter(
    signal: np.ndarray,
    sampling_rate_hz: float,
    cutoff_hz: float = 450.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter."""
    values, fs = _validate_signal(signal, sampling_rate_hz)
    assert fs is not None

    cutoff = _validate_cutoff(cutoff_hz, fs)
    order = _validate_order(order)

    sos = butter(
        order,
        cutoff,
        btype="lowpass",
        fs=fs,
        output="sos",
    )

    return _apply_sos_filter(values, sos, "low-pass")


def highpass_filter(
    signal: np.ndarray,
    sampling_rate_hz: float,
    cutoff_hz: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter."""
    values, fs = _validate_signal(signal, sampling_rate_hz)
    assert fs is not None

    cutoff = _validate_cutoff(cutoff_hz, fs)
    order = _validate_order(order)

    sos = butter(
        order,
        cutoff,
        btype="highpass",
        fs=fs,
        output="sos",
    )

    return _apply_sos_filter(values, sos, "high-pass")


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate_hz: float,
    lowcut_hz: float = 20.0,
    highcut_hz: float = 450.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter."""
    values, fs = _validate_signal(signal, sampling_rate_hz)
    assert fs is not None

    lowcut = _validate_cutoff(
        lowcut_hz, fs, "Low cutoff frequency"
    )
    highcut = _validate_cutoff(
        highcut_hz, fs, "High cutoff frequency"
    )
    order = _validate_order(order)

    if lowcut >= highcut:
        raise ValueError(
            "Low cutoff frequency must be lower than "
            "high cutoff frequency."
        )

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return _apply_sos_filter(values, sos, "band-pass")


def notch_filter(
    signal: np.ndarray,
    sampling_rate_hz: float,
    notch_hz: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Suppress narrowband interference, such as 50 Hz mains noise."""
    values, fs = _validate_signal(signal, sampling_rate_hz)
    assert fs is not None

    notch = _validate_cutoff(notch_hz, fs, "Notch frequency")
    q = float(quality_factor)

    if not np.isfinite(q) or q <= 0:
        raise ValueError(
            "Quality factor must be finite and greater than zero."
        )

    b, a = iirnotch(notch, q, fs=fs)

    try:
        return filtfilt(b, a, values)
    except ValueError as exc:
        raise ValueError(
            "Signal is too short for the notch filter."
        ) from exc


def detrend_signal(
    signal: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Remove a constant offset or linear trend.

    Args:
        signal: One-dimensional input signal.
        method: 'constant' removes the mean; 'linear' removes
            the best-fit linear trend.
    """
    values, _ = _validate_signal(signal)

    if method not in {"constant", "linear"}:
        raise ValueError("method must be 'constant' or 'linear'.")

    return scipy_detrend(values, type=method)


def full_wave_rectify(signal: np.ndarray) -> np.ndarray:
    """Full-wave rectify a signal using its absolute value."""
    values, _ = _validate_signal(signal)
    return np.abs(values)


def moving_average_envelope(
    signal: np.ndarray,
    sampling_rate_hz: float,
    window_ms: float = 50.0,
    rectify: bool = True,
) -> np.ndarray:
    """Calculate a same-length moving-average envelope.

    By default, the absolute signal is averaged. Set rectify=False
    to average the original signed values instead.
    """
    values, fs = _validate_signal(signal, sampling_rate_hz)
    assert fs is not None

    window = float(window_ms)

    if not np.isfinite(window) or window <= 0:
        raise ValueError(
            "Window duration must be finite and positive."
        )

    if not isinstance(rectify, (bool, np.bool_)):
        raise ValueError("rectify must be a boolean.")

    window_samples = max(1, int(round(window * fs / 1000.0)))
    working = np.abs(values) if rectify else values

    cumulative = np.concatenate(
        ([0.0], np.cumsum(working, dtype=float))
    )
    indices = np.arange(values.size)

    half_left = window_samples // 2
    half_right = window_samples - half_left

    starts = np.maximum(0, indices - half_left)
    ends = np.minimum(values.size, indices + half_right)

    sums = cumulative[ends] - cumulative[starts]
    counts = ends - starts

    return sums / counts


def rms_envelope(
    signal: np.ndarray,
    sampling_rate_hz: float,
    window_ms: float = 50.0,
) -> np.ndarray:
    """Calculate a centered moving RMS envelope.

    The output has the same length as the input. Windows are shortened
    at signal boundaries instead of adding artificial samples.
    """
    values, fs = _validate_signal(signal, sampling_rate_hz)
    assert fs is not None

    window = float(window_ms)

    if not np.isfinite(window) or window <= 0:
        raise ValueError(
            "Window duration must be finite and positive."
        )

    window_samples = max(1, int(round(window * fs / 1000.0)))
    squared = values * values

    cumulative = np.concatenate(
        ([0.0], np.cumsum(squared, dtype=float))
    )
    indices = np.arange(values.size)

    half_left = window_samples // 2
    half_right = window_samples - half_left

    starts = np.maximum(0, indices - half_left)
    ends = np.minimum(values.size, indices + half_right)

    sums = cumulative[ends] - cumulative[starts]
    counts = ends - starts

    return np.sqrt(np.maximum(sums / counts, 0.0))


def detect_artifacts(
    signal: np.ndarray,
    threshold_sigma: float = 6.0,
    *,
    method: str = "mad",
    absolute_threshold: float | None = None,
) -> np.ndarray:
    """Flag potentially abnormal samples without modifying the signal.

    Args:
        signal: One-dimensional input signal.
        threshold_sigma: Detection threshold in estimated standard
            deviations.
        method: 'mad' for robust median absolute deviation or
            'zscore' for mean and standard deviation.
        absolute_threshold: Optional additional absolute-amplitude
            threshold in the signal's original units.

    Returns:
        Boolean array with True at flagged sample positions.

    A flagged sample is not necessarily an actual acquisition artifact.
    """
    values, _ = _validate_signal(signal)

    threshold = float(threshold_sigma)

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "threshold_sigma must be finite and positive."
        )

    if method not in {"mad", "zscore"}:
        raise ValueError("method must be 'mad' or 'zscore'.")

    if method == "mad":
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        robust_sigma = 1.4826 * mad

        if robust_sigma == 0:
            mask = values != center
        else:
            mask = (
                np.abs(values - center) > threshold * robust_sigma
            )
    else:
        center = float(np.mean(values))
        spread = float(np.std(values))

        if spread == 0:
            mask = np.zeros(values.shape, dtype=bool)
        else:
            mask = (
                np.abs(values - center) > threshold * spread
            )

    if absolute_threshold is not None:
        limit = float(absolute_threshold)

        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(
                "absolute_threshold must be finite and positive."
            )

        mask = mask | (np.abs(values) > limit)

    return np.asarray(mask, dtype=bool)


def signal_quality_metrics(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    clipping_threshold: float | None = None,
    flat_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Calculate descriptive signal-quality metrics.

    No medical or diagnostic quality grade is assigned. Clipping
    metrics are meaningful only if the threshold corresponds to a
    known acquisition or ADC limit.
    """
    values, fs = _validate_signal(signal, sampling_rate_hz)
    assert fs is not None

    tolerance = float(flat_tolerance)

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "flat_tolerance must be finite and non-negative."
        )

    if values.size > 1:
        differences = np.diff(values)
        flat_count = int(
            np.count_nonzero(np.abs(differences) <= tolerance)
        )
        flat_fraction = flat_count / differences.size
    else:
        flat_count = 0
        flat_fraction = 0.0

    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(values * values)))

    if rms == 0:
        crest_factor = 0.0 if peak == 0 else float("inf")
    else:
        crest_factor = peak / rms

    metrics: dict[str, Any] = {
        "sample_count": int(values.size),
        "duration_seconds": float(values.size / fs),
        "sampling_rate_hz": fs,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": rms,
        "peak_absolute": peak,
        "peak_to_peak": float(np.ptp(values)),
        "crest_factor": float(crest_factor),
        "flat_difference_count": flat_count,
        "flat_difference_fraction": float(flat_fraction),
    }

    if clipping_threshold is not None:
        limit = float(clipping_threshold)

        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(
                "clipping_threshold must be finite and positive."
            )

        clipped = np.abs(values) >= limit

        metrics["clipping_threshold"] = limit
        metrics["clipped_sample_count"] = int(
            np.count_nonzero(clipped)
        )
        metrics["clipped_sample_fraction"] = float(
            np.mean(clipped)
        )

    return metrics
