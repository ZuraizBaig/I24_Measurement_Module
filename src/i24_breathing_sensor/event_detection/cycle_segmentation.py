
"""
Respiratory Cycle Segmentation
==============================

Segments a respiratory waveform into candidate cycles using
successive negative troughs as cycle boundaries.

Each cycle is assigned the strongest positive peak occurring
between its two boundary troughs.
"""

import numpy as np

from .models import DetectedEvent, RespiratoryCycle


def segment_respiratory_cycles(
    signal,
    sampling_rate,
    positive_peaks,
    negative_troughs,
):
    """
    Segment a respiratory signal into trough-to-trough cycles.

    Parameters
    ----------
    signal : array-like
        Raw respiratory waveform.

    sampling_rate : float
        Sampling frequency in Hz.

    positive_peaks : list[DetectedEvent]
        Detected positive peak events.

    negative_troughs : list[DetectedEvent]
        Detected negative trough events.

    Returns
    -------
    list[RespiratoryCycle]
        Candidate respiratory cycles.
    """

    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(x) == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains invalid values.")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    # Sort events chronologically.
    troughs = sorted(
        negative_troughs,
        key=lambda event: event.sample_index
    )

    peaks = sorted(
        positive_peaks,
        key=lambda event: event.sample_index
    )

    cycles = []

    # Two successive troughs define one candidate cycle.
    for i in range(len(troughs) - 1):

        start = troughs[i]
        end = troughs[i + 1]

        start_idx = start.sample_index
        end_idx = end.sample_index

        # Validate cycle boundaries.
        if not (
            0 <= start_idx < end_idx < len(x)
        ):
            continue

        # Find positive peaks inside this cycle.
        cycle_peaks = [
            peak
            for peak in peaks
            if start_idx < peak.sample_index < end_idx
        ]

        # Select the strongest positive peak.
        positive_peak = (
            max(cycle_peaks, key=lambda event: event.amplitude)
            if cycle_peaks
            else None
        )

        # Calculate cycle duration.
        duration_s = (
            end_idx - start_idx
        ) / sampling_rate

        # Calculate phase durations.
        if positive_peak is not None:

            phase_1_duration_s = (
                positive_peak.sample_index - start_idx
            ) / sampling_rate

            phase_2_duration_s = (
                end_idx - positive_peak.sample_index
            ) / sampling_rate

        else:

            phase_1_duration_s = None
            phase_2_duration_s = None

        # Extract cycle amplitude characteristics.
        cycle_signal = x[start_idx:end_idx + 1]

        peak_to_peak = float(np.ptp(cycle_signal))

        cycle = RespiratoryCycle(
            cycle_index=i + 1,
            start_trough=start,
            positive_peak=positive_peak,
            end_trough=end,
            duration_s=float(duration_s),
            phase_1_duration_s=phase_1_duration_s,
            phase_2_duration_s=phase_2_duration_s,
            peak_to_peak_amplitude=peak_to_peak,
        )

        cycles.append(cycle)

    return cycles
