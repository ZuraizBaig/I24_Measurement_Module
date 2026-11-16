
"""
Data models for I24 respiratory feature extraction.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TemporalFeatureResult:
    """Cycle-wise temporal features and summary statistics."""

    per_cycle: list[dict] = field(default_factory=list)

    summary: dict = field(default_factory=dict)

    pause_candidates: list[dict] = field(default_factory=list)


@dataclass
class FrequencyFeatureResult:
    """Frequency-domain respiratory features and spectra."""

    dominant_frequency_hz: float | None = None

    dominant_respiratory_rate_bpm: float | None = None

    fft_resolution_hz: float | None = None

    respiratory_band_power: float | None = None

    spectral_centroid_hz: float | None = None

    spectral_entropy: float | None = None

    spectral_band_power: dict = field(default_factory=dict)

    # FFT data
    fft_frequencies_hz: np.ndarray | None = None
    fft_amplitude: np.ndarray | None = None

    # Welch PSD data
    psd_frequencies_hz: np.ndarray | None = None
    psd_values: np.ndarray | None = None

    # STFT data
    stft_times_s: np.ndarray | None = None
    stft_frequencies_hz: np.ndarray | None = None
    stft_magnitude: np.ndarray | None = None

    # Dominant respiratory frequency per STFT window
    time_varying_frequency_hz: np.ndarray | None = None


@dataclass
class FeatureExtractionResult:
    """Combined output of respiratory feature extraction."""

    temporal: TemporalFeatureResult

    frequency: FrequencyFeatureResult
