
"""
Respiratory Event Detection Pipeline
=====================================

Combines the individual event detection components:

1. Signal validation
2. Positive peak detection
3. Negative trough detection
4. Respiratory cycle segmentation
5. Event validation

Returns a complete EventDetectionResult.
"""

from .models import EventDetectionResult

from .peak_detection import (
    validate_signal,
    detect_positive_peaks,
    detect_negative_troughs,
)

from .cycle_segmentation import (
    segment_respiratory_cycles,
)

from .event_validation import (
    validate_respiratory_cycles,
)


def detect_respiratory_events(
    signal,
    sampling_rate,
    min_event_distance_s=1.0,
    prominence_factor=0.35,
    positive_prominence=None,
    negative_prominence=None,
    min_cycle_duration_s=1.0,
    max_cycle_duration_s=10.0,
    min_peak_to_peak_amplitude=None,
    min_positive_peak_prominence=None,
):
    """
    Execute the complete respiratory event detection pipeline.

    Parameters
    ----------
    signal : array-like
        Raw respiratory waveform.

    sampling_rate : float
        Sampling frequency in Hz.

    min_event_distance_s : float
        Minimum separation between events of the same type.

    prominence_factor : float
        Automatic prominence estimation factor.

    positive_prominence : float, optional
        Custom positive peak prominence threshold.

    negative_prominence : float, optional
        Custom negative trough prominence threshold.

    min_cycle_duration_s : float
        Minimum acceptable respiratory cycle duration.

    max_cycle_duration_s : float
        Maximum acceptable respiratory cycle duration.

    min_peak_to_peak_amplitude : float, optional
        Minimum acceptable cycle amplitude.

    min_positive_peak_prominence : float, optional
        Minimum acceptable positive peak prominence.

    Returns
    -------
    EventDetectionResult
        Positive peaks, negative troughs and validated cycles.
    """

    # -----------------------------------------------------
    # 1. Validate input signal
    # -----------------------------------------------------

    x = validate_signal(
        signal,
        sampling_rate
    )

    # -----------------------------------------------------
    # 2. Positive peak detection
    # -----------------------------------------------------

    positive_peaks = detect_positive_peaks(
        signal=x,
        sampling_rate=sampling_rate,
        min_distance_s=min_event_distance_s,
        prominence=positive_prominence,
        prominence_factor=prominence_factor,
    )

    # -----------------------------------------------------
    # 3. Negative trough detection
    # -----------------------------------------------------

    negative_troughs = detect_negative_troughs(
        signal=x,
        sampling_rate=sampling_rate,
        min_distance_s=min_event_distance_s,
        prominence=negative_prominence,
        prominence_factor=prominence_factor,
    )

    # -----------------------------------------------------
    # 4. Respiratory cycle segmentation
    # -----------------------------------------------------

    cycles = segment_respiratory_cycles(
        signal=x,
        sampling_rate=sampling_rate,
        positive_peaks=positive_peaks,
        negative_troughs=negative_troughs,
    )

    # -----------------------------------------------------
    # 5. Event and cycle validation
    # -----------------------------------------------------

    validated_cycles = validate_respiratory_cycles(
        cycles=cycles,
        min_cycle_duration_s=min_cycle_duration_s,
        max_cycle_duration_s=max_cycle_duration_s,
        min_peak_to_peak_amplitude=min_peak_to_peak_amplitude,
        min_positive_peak_prominence=min_positive_peak_prominence,
    )

    # -----------------------------------------------------
    # 6. Return detection result
    # -----------------------------------------------------

    return EventDetectionResult(
        positive_peaks=positive_peaks,
        negative_troughs=negative_troughs,
        cycles=validated_cycles,
    )
