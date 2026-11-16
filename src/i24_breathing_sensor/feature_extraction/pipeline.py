
"""
Respiratory Feature Extraction Pipeline
========================================

Combines:
    1. Temporal feature extraction.
    2. Frequency-domain feature extraction.

Input:
    Raw respiratory waveform.
    Sampling frequency.
    EventDetectionResult from event_detection.

Output:
    FeatureExtractionResult.
"""

from .models import FeatureExtractionResult

from .temporal_features import (
    extract_temporal_features,
)

from .frequency_features import (
    extract_frequency_features,
)


def extract_respiratory_features(
    signal,
    sampling_rate,
    detection_result,
    inhalation_phase=None,
    pause_threshold_s=None,
    include_rejected=False,
    respiratory_band_hz=(0.05, 1.0),
    spectral_bands_hz=None,
    welch_window_s=16.0,
    stft_window_s=16.0,
    stft_hop_s=2.0,
):
    """
    Run temporal and frequency-domain feature extraction.

    Parameters
    ----------
    signal:
        Raw Channel 4 respiratory signal.

    sampling_rate:
        Actual acquisition sampling rate in Hz.

    detection_result:
        EventDetectionResult returned by event detection.

    inhalation_phase:
        "phase_1", "phase_2", or None.

    pause_threshold_s:
        Optional threshold for prolonged inter-trough intervals.

    include_rejected:
        Include rejected cycles in temporal calculations.

    respiratory_band_hz:
        Frequency range for respiratory spectral analysis.

    spectral_bands_hz:
        Dictionary defining spectral bands.

    welch_window_s:
        Welch analysis window in seconds.

    stft_window_s:
        STFT analysis window in seconds.

    stft_hop_s:
        STFT time step in seconds.

    Returns
    -------
    FeatureExtractionResult
    """

    # ---------------------------------------------------------
    # 1. Temporal features
    # ---------------------------------------------------------

    temporal_result = extract_temporal_features(
        detection_result=detection_result,
        inhalation_phase=inhalation_phase,
        pause_threshold_s=pause_threshold_s,
        include_rejected=include_rejected,
    )

    # ---------------------------------------------------------
    # 2. Frequency-domain features
    # ---------------------------------------------------------

    frequency_result = extract_frequency_features(
        signal=signal,
        sampling_rate=sampling_rate,
        respiratory_band_hz=respiratory_band_hz,
        spectral_bands_hz=spectral_bands_hz,
        welch_window_s=welch_window_s,
        stft_window_s=stft_window_s,
        stft_hop_s=stft_hop_s,
    )

    # ---------------------------------------------------------
    # 3. Combined result
    # ---------------------------------------------------------

    return FeatureExtractionResult(
        temporal=temporal_result,
        frequency=frequency_result,
    )
